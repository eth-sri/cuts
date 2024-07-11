import pickle
import torch
from .models import ResidualDenoiser, FullyConnectedDenoiser
from .loss_functions import total_variation_loss, jensen_shannon_divergence, mean_squared_error_loss
import numpy as np
from utils import Timer
from query import query_marginal, get_all_marginals
from differential_privacy import gaussian_mechanism, exponential_mechanism, cdp_rho


class Denoiser:
    
    def __init__(self, num_features, one_hot_index_map, in_size=100, layout=None, architecture_type='residual',
                 head='gumbel', noise_type='gaussian', device=None):
        """
        Constructor of the class.

        :param num_features: (int) Number of features the dataset has when full one-hot encoded.
        :param one_hot_index_map: (dict) A dictionary containing the feature names and the corresponding indices of the
            in the one-hot representation, in the same order as the features appear in the data.
        :param in_size: (int) The dimension of the input noise.
        :param layout: (list) The layout of the denoising neural network. It is to be passed as a list, where the first
            l entries are the dimensions of the l hidden layers in the network, and the last entry specifies the output
            dimension. Note that the output dimension should match the dimension of the fully one-hot encoded data.
        :param architecture_type: (str) Choose the architecture type for the denoising network. Currently, the available
            choices are 'residual' in which case an architecture similar to the CTGAN generator is used, or in any other
            case, a fully connected neural network with ReLU activations is used.
        :param head: (str) Choose the out-head of the network from the following options:
            - 'gumbel': Applies a straight-through gumbel softmax at the out-head of the network.
            - 'softmax': Applies a simple softmax at the output.
            - 'hard_softmax': Applies a straight-through softmax estimator at the output.
        :param noise_type: (str) Choose the type of noise injected at the input, available choices are 'gaussian',
            resulting in N(0, 1) Gaussian noise at the input, and 'uniform' for (0, 1] uniform noise.
        :param device: (str) The name of the devices on which the tensors are stored.
        """
        assert head in ['softmax', 'gumbel', 'hard_softmax'], f'Please choose the head from: softmax, gumbel, ' \
                                                              f'or hard_softmax'

        self.num_features = num_features
        self.one_hot_index_map = one_hot_index_map
        self.in_size = in_size
        self.architecture_type = architecture_type
        self.head = head
        if layout is None:
            layout = [100, 200, 200, num_features]
        self.layout = layout
        self.noise_type = noise_type
        if noise_type == 'categorical':
            self.in_size = num_features
            in_size = num_features

        # initialize the denoiser network
        if architecture_type == 'residual':
            self.generator = ResidualDenoiser(input_size=in_size, layout=layout, one_hot_index_map=one_hot_index_map,
                                              head=head)
        else:
            self.generator = FullyConnectedDenoiser(input_size=in_size, layout=layout,
                                                    one_hot_index_map=one_hot_index_map, head=head)
        self.device = device
        self.to(self.device)
        self.fitted = False
        self.dp, self.epsilon, self.delta = False, np.inf, 1e-9

    def __str__(self):
        fitted = 'not' if not self.fitted else ''
        ed = str((self.epsilon, self.delta))+'-DP' if self.dp else ''
        return f'{self.architecture_type} Denoiser {ed} {fitted} fitted'

    def to(self, device):
        """
        Send the parameters of the denoising network to the given device, and change the default device of the object.

        :param device: (str) Name of the target device.
        :return: self
        """
        self.device = device
        self.generator.to(device)
        return self

    def reinitialize(self):
        """
        Reinitialize the network.

        :return:
        """
        if self.architecture_type == 'residual':
            self.generator = ResidualDenoiser(input_size=self.in_size, layout=self.layout,
                                              one_hot_index_map=self.one_hot_index_map, head=self.head)
        else:
            self.generator = FullyConnectedDenoiser(input_size=self.in_size, layout=self.layout,
                                                    one_hot_index_map=self.one_hot_index_map, head=self.head)
        self.to(self.device)
        return self

    def _sample_noise(self, size):
        """
        Private method to sample noise for the input.

        :param size: (int) The number of rows to sample. Note that inputting this number of rows will result in this
            number of rows of output data.
        :return: (torch.tensor) The noise that can be used as input to the denoising network to generate synthetic
            data.
        """
        if self.noise_type == 'gaussian':
            z = torch.normal(mean=torch.zeros((size, self.in_size), device=self.device),
                             std=torch.ones((size, self.in_size), device=self.device)).to(self.device)
        elif self.noise_type == 'uniform_0_1':
            z = torch.rand((size, self.in_size)).to(self.device)
        elif self.noise_type == 'uniform_m1_1':
            z = 2 * torch.rand((size, self.in_size)).to(self.device) - 1
        elif self.noise_type == 'categorical':
            z = torch.cat([torch.nn.functional.gumbel_softmax(torch.ones((size, len(idx))).to(self.device), hard=True)
                           for idx in self.one_hot_index_map.values()], dim=-1).to(self.device)
        else:
            raise ValueError(f'{self.noise_type} is not a valid choice for the generator input noise, please choose '
                             f'from gaussian or uniform')
        return z

    def _sample_per_feature(self, data):
        """
        In case the generator outputs soft probabilities in each feature field, instead of one-hot vectors, we
        postsample the data to produce discrete features. We do this by sampling each field independently in relation
        to its contained soft probabilities.

        :param data: (torch.tensor) The soft encoded data.
        :return: (torch.tensor) The sample one-hot encoded data.
        """
        out_data = torch.zeros_like(data)
        pointer = 0
        for i, (feature_name, feature_index) in enumerate(self.one_hot_index_map.items()):
            locs = torch.multinomial(data[:, feature_index], num_samples=1).squeeze(-1)
            out_data[np.arange(len(locs)), pointer+locs] = 1.
            pointer += len(feature_index)
        return out_data

    def generate_data(self, size, sample=False):
        """
        We generate synthetic data of length size using this method. If the outputs of the denoising network are soft,
        and you wish to generate one-hot encoded discrete data, set the sample argument to True. This is necessary if
        the output layer of the network contains softmax instead of gumbel-softmax activations.

        :param size: (int) The length of the dataset we want to create.
        :param sample: (bool) Toggle if the outputs of the denoising network are soft and you want to return a
            hard-sampled dataset in one-hot encoding. Note that in the case of a gumbel-softmax activation at the output
            the resulting dataset will already be in one-hot encoding.
        :return: (torch.tensor) The resulting dataset sampled from the denoising network.
        """
        z = self._sample_noise(size)
        data = self.generator(z)
        if sample:
            data = self._sample_per_feature(data)
        return data

    def _fit(self, target_marginals, optimizer, scheduler, n_epochs=1000, batch_size=1000, subsample=None,
             loss_to_use='total_variation', max_slice=1000, constraint_compiler=None, save=None,
             verbose=False):
        """
        The private method used to fit the denoising model given a set of measured marginals. These marginals can be
        measured in a private manner as well, in which case the resulting model will produce data with differential
        privacy guarantees.

        :param target_marginals: (dict) A dictionary containing the measured normalized marginals (sums to 1). The
            structure of the dictionary for an N-way marginal should be:
            {('feature1', 'feature2', ..., 'featureN'): torch.tensor},
            where the torch.tensor contains the measured normalized marginal corresponding to the features in the key
            tuple.
        :param optimizer: (torch.optim.Optimizer) A torch optimizer containing the parameters of the denoising network.
            This optimizer is going to be used to train the denoising network.
        :param scheduler: (torch.optim.Scheduler) A learning rate scheduler containing the optimizer.
        :param n_epochs: (int) The number of epochs the training should be run for.
        :param batch_size: (int) The size of the input noise, i.e., the intermediate tabular_datasets created at each update to
            estimate the current performance of the generative model.
        :param subsample: (int) If given, at each update we will only consider subsample amount of marginals randomly
            selected from the target marginals. Note that in each epoch we still update at exactly once over each
            marginal.
        :param loss_to_use: (str) Name of the loss function to be used for training. Available are:
            - 'total_variation': the mean Total Variation loss,
            - 'jensen_shannon': the Jensen-Shannon divergence,
            - 'squared_error': The mean squared error loss.
        :param: max_slice: (int) The maximum size of a slice processed by the GPU at once when calculating marginals.
        :param constraint_compiler: (ConstraintCompiler) Instantiated ConstraintCompiler method that, if given, will be used
            to regularize towards respecting constraints on the generated data. 
        :param save: (list) List of iterations at which the current model is to be saved.
        :param verbose: (bool) Toggle to show live progress during training.
        :return: self
        """

        loss_function = {
            'total_variation': total_variation_loss,
            'jensen_shannon': jensen_shannon_divergence,
            'squared_error': mean_squared_error_loss
        }

        marginal_names = np.array(list(target_marginals.keys()), dtype='object')
        n_marginals = len(marginal_names)

        # prepare query batched training
        if subsample is None:
            subsample = len(marginal_names)
        n_splits = np.ceil(n_marginals/subsample).astype(int)

        timer = Timer(n_epochs)

        for it in range(n_epochs):

            # diagnostics
            timer.start()
            losses_in_epoch = []

            # randomly permute for training
            split_marginal_names = np.array_split(np.random.permutation(marginal_names), n_splits)

            for sub_it, names_in_split in enumerate(split_marginal_names):
                optimizer.zero_grad()

                # get the data
                synthetic_data_batch = self.generate_data(batch_size)

                # iterate over all marginals to be compared and cumulate the loss
                total_loss = torch.tensor([0.], device=self.device)
                for marginal_name in names_in_split:
                    # query the synthetic data
                    synthetic_marginal = query_marginal(synthetic_data_batch, marginal_name, self.one_hot_index_map,
                                                        input_torch=True, normalize=True, max_slice=max_slice)
                    # calculate the individual loss
                    _loss = loss_function[loss_to_use](target_marginals[tuple(marginal_name)], synthetic_marginal)
                    total_loss += 1/(len(names_in_split)) * _loss

                losses_in_epoch.append(total_loss.item())

                # constraint regularizer
                if constraint_compiler is not None:
                    reg = constraint_compiler.compile_regularizer(synthetic_data_batch)
                    total_loss += reg

                # calculate the gradients and make step
                total_loss.backward()
                optimizer.step()

            scheduler.step()
            timer.end()

            if save is not None:
                if it + 1 in save[0]:
                    with open(f'{save[1]}_{it+1}.pickle', 'wb') as f:
                        pickle.dump(self, f)

            reg_val = 0. if constraint_compiler is None else reg.item()
            if verbose:
                print(f'Epoch: {it+1}/{n_epochs}          Loss: {np.mean(losses_in_epoch):.5f}    Regularizer: {reg_val:.3f}          {timer}',
                      end='\r')
        if verbose:
            timer.duration()

        return self

    def _compile_candidates_aim(self, workload, horizontal=True):
        """
        This method is required for running the AIM mechanism. It calculates the downward closure of the
        workload and weights each entry by how much those marginals' feature sets overlap with the features contained
        in the workload. Those entries that do not overlap are removed as they are irrelevant for the workload.

        :param workload: (list of tuples) The list of the marginal names in the workload.
        :return: (list of tuples) List of (weight, marginal name) of the downward closure of the workload.
        """
        # get the max degree of marginal in the workload
        max_deg = np.max([len(m) for m in workload]).astype(int)

        # get all submarginals of this degree or one lower if no horizontal extension is allowed
        if not horizontal:
            all_submarginals = get_all_marginals(list(self.one_hot_index_map.keys()), max(1, max_deg-1), downward_closure=True)
            all_submarginals += workload
        else:
            all_submarginals = get_all_marginals(list(self.one_hot_index_map.keys()), max_deg, downward_closure=True)

        # calculate the weight for all submarginals
        weights = np.array([sum([len(set(sub_m) & set(m)) for m in workload]) for sub_m in all_submarginals])

        return [(w, m) for w, m in zip(weights, all_submarginals) if w > 0]

    def _calculate_domain_size(self, marginal_name):
        """
        A method to calculate the domain size of a given marginal.

        :param marginal_name: (tuple of str) Name of the marginal.
        :return: (np.int) The size of the product-space of the features constituting the marginal.
        """
        lens = []
        for feature in marginal_name:
            lens.append(len(self.one_hot_index_map[feature]))
        return np.prod(lens)

    @staticmethod
    def _anneal_privacy_budget_adaptive(measured_marginal, new_measurement_result, sigma, epsilon, domain_size,
                                        verbose, threshold=np.sqrt(2)):
        """
        Adapts the privacy budget used for selecting new candidates. Concretely, we either increase the privacy budget
        used in each round, or decrease it, depending on the improvements made on the current measurement. The goal is
        to always hover around an improvement of which the error is explained fully by the expected error on the current
        DP level.

        :param measured_marginal: (torch.tensor) The marginal measured under the previous model.
        :param new_measurement_result: (torch.tensor) The marginal measured under the new model.
        :param sigma: (float) The current standard deviation for the Gaussian mechanism.
        :param epsilon: (float) The current temperature for the exponential mechanism.
        :param domain_size: (int) The size of the product space of the features in the marginal.
        :param verbose: (bool) Toggle to print the decisions.
        :param threshold: (float) The threshold for the maximum adaptation step. Default is sqrt(2). This will result in
            an at most double / half usage of the zCDP budget in the next round.
        :return: epsilon, sigma (tuple of floats) The updated privacy budgets.
        """
        learning_gain = (measured_marginal - new_measurement_result).abs().sum()
        if learning_gain <= np.sqrt(2 / np.pi) * sigma * domain_size:
            if verbose:
                print('Reducing Sigma')
            dcr_factor = (learning_gain / domain_size * np.sqrt(np.pi / 2)).cpu().item() / sigma
            dcr_factor = np.max([dcr_factor, 1 / threshold])
            sigma = sigma * dcr_factor
            epsilon = epsilon / dcr_factor
        else:
            if verbose:
                print('Increasing Sigma')
            incr_factor = (learning_gain / domain_size * np.sqrt(np.pi / 2)).cpu().item() / sigma
            incr_factor = np.min([incr_factor, threshold])
            sigma = sigma * incr_factor
            epsilon = epsilon / incr_factor
        return sigma, epsilon

    @staticmethod
    def _anneal_privacy_budget_aim(measured_marginal, new_measurement_result, sigma, epsilon, domain_size,
                                   verbose):
        """
        Anneals the privacy budget using the simple method from AIM. Namely, if the improvement in the measured marginal
        is below than what could be explained by the DP noise, the budget allocated for the next step is quadrupled.

        :param measured_marginal: (torch.tensor) The marginal measured under the previous model.
        :param new_measurement_result: (torch.tensor) The marginal measured under the new model.
        :param sigma: (float) The current standard deviation for the Gaussian mechanism.
        :param epsilon: (float) The current temperature for the exponential mechanism.
        :param domain_size: (int) The size of the product space of the features in the marginal.
        :param verbose: (bool) Toggle to print the decisions.
        :return: epsilon, sigma (tuple of floats) The updated privacy budgets.
        """
        learning_gain = (measured_marginal - new_measurement_result).abs().sum()
        if learning_gain <= np.sqrt(2 / np.pi) * sigma * domain_size:
            if verbose:
                print('Reducing Sigma')
            epsilon = 2 * epsilon
            sigma = sigma / 2
        else:
            epsilon = epsilon
            sigma = sigma
        return sigma, epsilon

    def _fit_aim(self, optimizer_parameters, scheduler_parameters, full_one_hot_dataset, workload, rho, T=None,
                 alpha=0.9, n_epochs=1000, batch_size=1000, subsample=None, anneal='adaptive', horizontal=True,
                 loss_to_use='total_variation', data_len=None, keep_running_average=True, max_slice=1000,
                 return_measurements=False, verbose=False):
        """
        Method to fit the data synthesizer in a differentially private way using the selection and measurement method
        of AIM (https://arxiv.org/pdf/2201.12677.pdf).

        :param optimizer_parameters: (dict) A dictionary containing the specification of the optimizer. Minimally, it
            should contain: 'name': name of the optimizer.
        :param scheduler_parameters: (dict) A dictionary containing the specification of the learning rate scheduler.
            Minimally, it should contain: 'name': name of the scheduler, 'eta_min': the minimal learning rate.
        :param full_one_hot_dataset: (torch.tensor) The fully one-hot encoded, discretized true dataset.
        :param workload: (list of tuples) The list of the marginal names in the workload.
        :param rho: (float) Zero-Concentrated Differential Privacy parameter.
        :param T: (int) Hyper parameter, it matters only at initialization when measuring the 1-way marginals. It should
            be at least larger than the number of features in the dataset. Default is 16 times the number of features.
        :param alpha: (float) Hyper parameter, has to be in (0, 1), it divides the budget between measuring and
            selecting. Namely, higher alpha corresponds to spending more privacy budget in measuring.
        :param n_epochs: (int) The number of epochs for which the training of the denoiser model will be ran.
        :param batch_size: (int) The size of the input noise, i.e., the intermediate tabular_datasets created at each update to
            estimate the current performance of the generative model.
        :param subsample: (int) If given, at each update we will only consider subsample amount of marginals randomly
            selected from the target marginals. Note that in each epoch we still update at exactly once over each
            marginal.
        :param anneal: (str) Choose the budget annealing method. Available are:
            - 'aim': will use the privacy budget annealing of AIM, as in the original publication,
            - 'adaptive': will use our improved budgeting, where we adapt each step such that it retrospectively
                would have been accepted, inspired by adaptive differential equation solvers.
        :param horizontal: (bool) Allow horizontal extension of the workload. If set to False, the workload will only be extended
            with the strictly downward closure of it, i.e., only lower dimensional marginals will be added.
        :param loss_to_use: (str) Name of the loss function to be used for training. Available are:
            - 'total_variation': the mean Total Variation loss,
            - 'jensen_shannon': the Jensen-Shannon divergence.
        :param data_len: (int) The length of the dataset we generate when measuring the current error for the selection
            step of the algorithm. The default is 10 * batch_size.
        :param keep_running_average: (bool) Toggle to keep the running average of measurements, or to replace each
            measurement by the newest one.
        :param: max_slice: (int) The maximum size of a slice processed by the GPU at once when calculating marginals.
        :param return_measurements: (bool) Toggle to return the reference marginals used for fitting. Note that this
            will alter what the method returns, and the object can not be used anymore in chained expressions as it
            does not return self anymore.
        :param verbose: (bool) Toggle to print information during fitting.
        :return: if return_measurements: (dict) Returns the dictionary of the measured reference marginals based on which the
                    model is fitted.
                else:
                    self.
        """

        available_optimizers = {'adam': torch.optim.Adam}
        available_schedulers = {'cosine': torch.optim.lr_scheduler.CosineAnnealingLR}

        orig_data_len = data_len

        if T is None:
            T = 16 * len(self.one_hot_index_map)

        timer = Timer(T)

        # set up, initialize, and prepare everything
        sigma = np.sqrt(T / (2 * alpha * rho))  # budget we use each time for measurement
        epsilon = np.sqrt(8 * (1 - alpha) * rho / T)  # budget we use each time for selection
        rho_used = 0.
        all_dp_measurements = {}
        all_non_dp_measurements = {}
        weighted_candidates = self._compile_candidates_aim(workload, horizontal)
        domain_sizes = {marginal_name: self._calculate_domain_size(marginal_name) for _, marginal_name in weighted_candidates}
        times_measured = {marginal_name: 0. for _, marginal_name in weighted_candidates}
        largest_weight = np.max([w for w, _ in weighted_candidates])
        with torch.no_grad():
            real_answers = {m: query_marginal(full_one_hot_dataset, m, self.one_hot_index_map, normalize=False, 
                                            input_torch=True, max_slice=max_slice) for _, m in weighted_candidates}
        errors = [np.inf]
        all_noisy_lines = []

        # ------------- Initialize the generator ------------- #
        # initialize the generator -- this is done by fitting all 1-way marginals
        if verbose:
            print('Initialize')
        optimizer = available_optimizers[optimizer_parameters['name']](self.generator.parameters())
        scheduler = available_schedulers[scheduler_parameters['name']](optimizer, n_epochs, scheduler_parameters['eta_min'])
        all_1_way_marginal_names = get_all_marginals(list(self.one_hot_index_map.keys()), 1)
        for marginal_name in all_1_way_marginal_names:
            queried_true_marginal = query_marginal(full_one_hot_dataset, marginal_name, self.one_hot_index_map,
                                                   normalize=False, input_torch=True, max_slice=max_slice)
            # add DP noise
            queried_true_marginal = gaussian_mechanism(queried_true_marginal, sigma)
            # normalize the perturbed marginal to be used with the distances we use
            all_dp_measurements[marginal_name] = queried_true_marginal / queried_true_marginal.sum()
            all_noisy_lines.append(queried_true_marginal.sum().item())
            times_measured[marginal_name] += 1.
        self._fit(target_marginals=all_dp_measurements,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  n_epochs=n_epochs,
                  batch_size=batch_size,
                  subsample=subsample,
                  loss_to_use=loss_to_use,
                  max_slice=max_slice,
                  verbose=False)

        # account the budget used for initialization
        rho_used = rho_used + len(all_1_way_marginal_names) / (2 * sigma**2)

        # ------------- Main loop of select, measure, and fit ------------- #
        # enter the loop where we adaptively select and measure
        terminate = False
        it = 1
        while rho_used < rho and not terminate:
            timer.start()

            if verbose:
                print('\n')
                print(f'Iteration: {it}    Error: {np.mean(errors):.3f}    Sigma: {sigma:.2f}    Rho Used: {100 * rho_used/rho:.1f}%')

            # ------ Check budget ------ #
            # we check how much budget is left, and if we would potentially overuse it in two rounds then we terminate
            rho_remaining = rho - rho_used
            if rho_remaining < 2 * (1/8 * epsilon**2 + 1/(2 * sigma**2)):
                if verbose:
                    print('Using up the remaining budget')
                terminate = True
                sigma = np.sqrt(1 / (2 * alpha * rho_remaining))
                epsilon = np.sqrt(8 * (1 - alpha) * rho_remaining)

            # ------ Estimate Data Len ------ #
            if orig_data_len is None:
                data_len = int(np.mean(all_noisy_lines) + 0.5)
            else:
                data_len = orig_data_len
            self.data_len = data_len  # save it in self

            # ------ Select ------ #
            with torch.no_grad():
                # generate data with the current model
                fake_data = self.generate_data(size=data_len, sample=(self.head == 'softmax')).detach().clone()
                # measure the score function
                scores = torch.zeros(len(weighted_candidates), device=self.device)
                errors = []
                if verbose:
                    print('Select')
                subtimer = Timer(len(weighted_candidates))
                for idx, (w, m) in enumerate(weighted_candidates):
                    subtimer.start()
                    if verbose:
                        print(subtimer, end='\r')
                    fake_answer = query_marginal(fake_data, m, self.one_hot_index_map, normalize=False, 
                                                input_torch=True, max_slice=max_slice)
                    real_answer = real_answers[m]
                    # real_answer = query_marginal(full_one_hot_dataset, m, self.one_hot_index_map, normalize=False, 
                    #                              input_torch=True, max_slice=max_slice)
                    error = (fake_answer - real_answer).abs().sum()
                    scores[idx] = w * (error - np.sqrt(2/np.pi) * sigma * domain_sizes[m])
                    errors.append(error.item() / data_len)
                    subtimer.end()

            # apply the exponential mechanism for selection from the scores
            selected_candidate_index = exponential_mechanism(scores=scores, epsilon=epsilon, sensitivity=largest_weight)
            selected_candidate_name = [m for _, m in weighted_candidates][selected_candidate_index]
            if verbose:
                print(f'{selected_candidate_name} selected for measurement')

            # ------ Measure ------ #
            if verbose:
                print('Measure')
            measured_marginal = query_marginal(full_one_hot_dataset, selected_candidate_name, 
                                               self.one_hot_index_map, normalize=False, input_torch=True, 
                                               max_slice=max_slice)
            dp_measured_candidate = gaussian_mechanism(measured_marginal, sigma)
            all_noisy_lines.append(dp_measured_candidate.sum().item())
            if keep_running_average:
                # keep a running average of the measured marginal
                tm = times_measured[selected_candidate_name]
                if tm > 0:
                    prev_measurement_deavg = tm * all_dp_measurements[selected_candidate_name]
                else:
                    prev_measurement_deavg = torch.zeros_like(dp_measured_candidate)
                curr_measurement_normalized = dp_measured_candidate / dp_measured_candidate.sum()
                all_dp_measurements[selected_candidate_name] = 1/(tm+1) * (prev_measurement_deavg + curr_measurement_normalized)
                times_measured[selected_candidate_name] += 1.
            else:
                curr_measurement_normalized = dp_measured_candidate / dp_measured_candidate.sum()
                all_dp_measurements[selected_candidate_name] = curr_measurement_normalized

            # ------ Fit ------ #
            if verbose:
                print('Fit')
            optimizer = available_optimizers[optimizer_parameters['name']](self.generator.parameters())
            scheduler = available_schedulers[scheduler_parameters['name']](optimizer, n_epochs,
                                                                           scheduler_parameters['eta_min'])
            self._fit(target_marginals=all_dp_measurements,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      n_epochs=n_epochs,
                      batch_size=batch_size,
                      subsample=subsample,
                      loss_to_use=loss_to_use,
                      max_slice=max_slice,
                      verbose=False)

            # ------ Account ------ #
            rho_used = rho_used + 1/8 * epsilon**2 + 1/(2 * sigma**2)

            # ------ Anneal Budget ------ #
            new_fake_data = self.generate_data(size=data_len, sample=(self.head == 'softmax')).detach().clone()
            new_measurement_result = query_marginal(new_fake_data, selected_candidate_name, self.one_hot_index_map,
                                                    normalize=False, input_torch=True)
            if anneal == 'aim':
                sigma, epsilon = self._anneal_privacy_budget_aim(measured_marginal=measured_marginal,
                                                                 new_measurement_result=new_measurement_result,
                                                                 sigma=sigma,
                                                                 epsilon=epsilon,
                                                                 domain_size=domain_sizes[selected_candidate_name],
                                                                 verbose=verbose)
            elif anneal == 'adaptive':
                sigma, epsilon = self._anneal_privacy_budget_adaptive(measured_marginal=measured_marginal,
                                                                      new_measurement_result=new_measurement_result,
                                                                      sigma=sigma,
                                                                      epsilon=epsilon,
                                                                      domain_size=domain_sizes[selected_candidate_name],
                                                                      verbose=verbose)
            else:
                raise ValueError('Only aim and adaptive privacy budgeting methods are available.')

            it += 1
            timer.end()

        if verbose:
            timer.duration()
        
        if return_measurements:
            return all_dp_measurements
        else:
            return self

    def fit(self, epsilon=np.inf, delta=1e-9, algorithm='input', optimizer=None, scheduler=None,
            optimizer_parameters=None, scheduler_parameters=None, full_one_hot_dataset=None, workload=2, T=None,
            alpha=0.9, anneal='adaptive', horizontal=True, keep_running_average=True, data_len=None, target_marginals=None,
            n_epochs=1000, batch_size=1000, subsample=None, loss_to_use='total_variation', max_slice=1000,
            constraint_compiler=None, save=None, return_measurements=False, verbose=False):
        """
        The general method to fit the denoiser. It is both possible to fit it in a private and non-private manner.
        Currently, two modes are available for fitting:
            - 'input': Using precalculated marginals (measured either privately or non-privately) we fit the denoiser
                only to these marginals.
            - 'aim': We use the AIM mechanisms to fit the denoiser privately to a given workload and at a given privacy
                level of epsilon, delta.

        :param epsilon: (float) The epsilon privacy parameter for approximate DP.
        :param delta: (float) The delta privacy parameter for approximate DP.
        :param algorithm: (str) The fitting method, see the general description above for the available modes.
        :param optimizer: (torch.optim.Optimizer) Only relevant for 'input': A torch optimizer containing the parameters
            of the denoising network. This optimizer is going to be used to train the denoising network.
        :param scheduler: (torch.optim.Scheduler) Only relevant for 'input': A learning rate scheduler containing the
            optimizer.
        :param optimizer_parameters: (dict) Only relevant for 'aim': A dictionary containing the specification of the
            optimizer. Minimally, it should contain: 'name': name of the optimizer.
        :param scheduler_parameters: (dict) Only relevant for 'aim': A dictionary containing the specification of the
            learning rate scheduler. Minimally, it should contain: 'name': name of the scheduler, 'eta_min': the minimal
            learning rate.
        :param full_one_hot_dataset: (torch.tensor) Only relevant for 'aim': The fully one-hot encoded, discretized true
            dataset.
        :param workload: (int or list of tuples) Only relevant for 'aim': The list of the marginal names in the
            workload. Instead, if an integer n is given, this list is generated from all the n-way marginals of the
            dataset.
        :param T: (int) Only relevant for 'aim': Hyper parameter, it matters only at initialization when measuring
            the 1-way marginals. It should be at least larger than the number of features in the dataset. Default is 16
            times the number of features.
        :param alpha: (float) Only relevant for 'aim': Hyper parameter, has to be in (0, 1), it divides the budget
            between measuring and selecting. Namely, higher alpha corresponds to spending more privacy budget in
            measuring.
        :param anneal: (str) Only relevant for 'aim': Choose the budget annealing method. Available are:
            - 'aim': will use the privacy budget annealing of AIM, as in the original publication,
            - 'adaptive': will use our improved budgeting, where we adapt each step such that it retrospectively
                would have been accepted, inspired by adaptive differential equation solvers.
        :param horizontal: (bool) Allow horizontal extension of the workload. If set to False, the workload will only be extended
            with the strictly downward closure of it, i.e., only lower dimensional marginals will be added.
        :param keep_running_average: (bool) Only relevant for 'aim': Toggle to keep the running average of measurements,
            or to replace each measurement by the newest one.
        :param data_len: (int) Only relevant for 'aim': The length of the dataset we generate when measuring the current
            error for the selection step of the algorithm. The default is 10 * batch_size.
        :param target_marginals: (dict) Only relevant for 'input': A dictionary containing the measured normalized
            marginals (sums to 1). The structure of the dictionary for an N-way marginal should be:
                {('feature1', 'feature2', ..., 'featureN'): torch.tensor},
            where the torch.tensor contains the measured normalized marginal corresponding to the features in the key
            tuple.
        :param n_epochs: (int) The number of epochs for which the training of the denoiser model will be ran.
        :param batch_size: (int) The size of the input noise, i.e., the intermediate tabular_datasets created at each update to
            estimate the current performance of the generative model.
        :param subsample: (int) If given, at each update we will only consider subsample amount of marginals randomly
            selected from the target marginals. Note that in each epoch we still update at exactly once over each
            marginal.
        :param loss_to_use: (str) Name of the loss function to be used for training. Available are:
            - 'total_variation': the mean Total Variation loss,
            - 'jensen_shannon': the Jensen-Shannon divergence.
        :param max_slice: (int) The maximum size of a slice processed by the GPU at once when calculating marginals.
        :param constraint_compiler: (ConstraintCompiler) Instantiated ConstraintCompiler method that, if given, will be used
            to regularize towards respecting constraints on the generated data. 
        :param save: (list, path) If this argument is given, we save the model at the iterations designated in the list,
            under the given path + the iteration. Only applies in non_dp training.
        :param return_measurements: (bool) Toggle to return the reference marginals used for fitting. Note that this
            will alter what the method returns, and the object can not be used anymore in chained expressions as it
            does not return self anymore.
        :param verbose: (bool) Toggle to print information during fitting.
        :return: if return_measurements: (dict) Returns the dictionary of the measured reference marginals based on which the
                    model is fitted.
                else:
                    self.
        """

        # assert if we can train the desired way
        implemented_algorithms = ['input', 'aim']
        assert algorithm in implemented_algorithms, f'Algorithm {algorithm} is not implemented, please choose from: ' \
                                                    f'{implemented_algorithms}.'

        # train by the input based simple algorithm where the marginals to be fitted are user defined
        if algorithm == 'input':

            # take care of default arguments
            optimizer_was_none = False
            if optimizer is None:
                optimizer_was_none = True
                optimizer = torch.optim.Adam(self.generator.parameters())
            if scheduler is None or optimizer_was_none:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-8)

            # make sure that the input is correct
            assert target_marginals is not None or workload is not None, 'In this mode either the target marginals or ' \
                                                                         'the workload has to be provided as input'

            # if there was no concrete target given, then we train on the downward closure of the workload
            if target_marginals is None:
                assert full_one_hot_dataset is not None, 'For fitting without precomputed marginals, we require the dataset'
                if isinstance(workload, int):
                    workload = get_all_marginals(list(self.one_hot_index_map.keys()), workload, downward_closure=False)
                # actually calculate all targets now
                target_marginals = {m: query_marginal(full_one_hot_dataset, m, self.one_hot_index_map, normalize=True,
                                                      input_torch=True, max_slice=max_slice) for m in workload}
            
            self._fit(optimizer=optimizer,
                      scheduler=scheduler,
                      target_marginals=target_marginals,
                      n_epochs=n_epochs,
                      batch_size=batch_size,
                      subsample=subsample,
                      loss_to_use=loss_to_use,
                      max_slice=max_slice,
                      constraint_compiler=constraint_compiler,
                      save=save,
                      verbose=verbose)
            
            returned = target_marginals  # align for returning

        elif algorithm == 'aim':

            # make sure we have the dataset
            assert full_one_hot_dataset is not None, 'For fitting with AIM, we require the dataset'

            # make sure that we have some valid privacy budget
            assert 0 < epsilon < np.inf, 'Set epsilon to a finite positive value'

            # set defaults
            if optimizer_parameters is None:
                optimizer_parameters = {'name': 'adam'}
            if scheduler_parameters is None:
                scheduler_parameters = {'name': 'cosine', 'eta_min': 1e-8}
            if isinstance(workload, int):
                workload = get_all_marginals(list(self.one_hot_index_map.keys()), workload, downward_closure=False)

            # calculate the zCDP privacy budget given epsilon and delta
            rho = cdp_rho(epsilon, delta)

            returned = self._fit_aim(
                optimizer_parameters=optimizer_parameters,
                scheduler_parameters=scheduler_parameters,
                full_one_hot_dataset=full_one_hot_dataset,
                workload=workload,
                rho=rho,
                T=T,
                alpha=alpha,
                n_epochs=n_epochs,
                batch_size=batch_size,
                subsample=subsample,
                anneal=anneal,
                horizontal=horizontal,
                loss_to_use=loss_to_use,
                data_len=data_len,
                keep_running_average=keep_running_average,
                max_slice=max_slice,
                return_measurements=return_measurements,
                verbose=verbose
            )

        self.epsilon, self.delta = epsilon, delta
        self.dp = True if 0 < self.epsilon < np.inf else False
        self.fitted = True

        if return_measurements:
            return returned
        else:
            return self
