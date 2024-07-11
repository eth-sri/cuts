# CuTS: Customizable Tabular Synthetic Data Generation <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

This is the official codebase of the ICML 2024 paper [CuTS: Customizable Tabular Synthetic Data Generation](https://arxiv.org/abs/2307.03577v4). In case of any questions, feel free to raise an issue on GitHub or contact the corresponding author per email: Mark Vero, mark.vero@inf.ethz.ch.

## Installation

For ease of installation, we provide a [conda](https://docs.conda.io/en/latest/) environment in `environment.yml`, which can be installed with the following command:

```
conda env create -f environment.yml
```

The environment can be activated using the command:

```
conda activate cuts
```

## Example usage of CuTS

We provide a minimal example of a CuTS synthetization, executable by running the provided `example.py` script.

```python
from programmable_synthesizer import CuTS


program = '''
SYNTHESIZE: Adult;

    MINIMIZE: STATISTICAL:  
        E[age|age > 30] == 40;
    
END;
'''    

cuts = CuTS(program)
cuts.fit(verbose=True)

syndata = cuts.generate_data(30000)

```

## Reproducing the experiments in the paper

### Datasets

All raw data files are included in the repository, except for the files of the Health Heritage Prize dataset, as it is over the size limit of GitHub. The required raw data for the Health Heritage dataset can be downloaded from [here](https://files.sri.inf.ethz.ch/tableak/Health_Heritage/). Please, download the files and place them on the path `tabular_datasets/Health_Heritage`.

To fix the train-test split of the datasets, run
```
python split_datasets.py --dataset <dataset_name>
```
for each dataset, keeping other parameters fixed. For the Health Heritage Prize and the Compas datasets, run the above command additionally also with the flag `--binary_fairness`.

### Experiments

All single-constraint experiments can be reproduced by running the python script `run_constraint_program_benchmark.py`, specifying the `--dataset` (see list of dataset names in the script), `--workload` (default for non-DP and `all_three` for DP experiments), and `--epsilon` (default for non-DP and `1.0` for DP experiments) arguments.

All constraint-combination experiments can be reproduced by running the python script `run_ablation.py`, specifying the `--dataset`, and `--option` (`1` for mixed constraints `2` for only logical constraints) arguments.

Each of the above setups can be also run with the flag `--baseline_mode`, where the raw, unconstrained model is benchmarked.

Note that on the first execution at specific privacy level on a given dataset, the backbone model is trained. This may take up to a few hours. However, afterwards, this backbone is saved, and any other experiments on the same dataset and privacy level will load and fine tune this model, not retraining it, saving considerable computation time. Note that a slight random difference in the backbone model may cause your results to non-significantly deviate from the ones presented in the paper.

All experimental results are saved in `.npy` files in the folder `experiment_data`, and include 6 metrics (three statistical similarity metrics and 3 downstream classifier performance metrics). The layout of the `.npy` tensor can be inspected in the corresponding experimental scripts. To reproduce the results shown in the paper, average over the `sample` and `resample` dimensions of the mean results for each metric.

## Citation

```
@inproceedings{vero2024cuts,
    title={Cu{TS}: Customizable Tabular Synthetic Data Generation},
    author={Mark Vero and Mislav Balunovic and Martin Vechev},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024}
}
```