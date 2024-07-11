import os
import re


# ------ Change these to create the constraint ------ #
filename = 'fairness_EO_downstream_sex_desired'

constraint = """
SYNTHESIZE: ADULT;

    MINIMIZE: BIAS: PARAM <param1>:
        EQUALIZED_ODDS(protected=sex, target=salary, desired_outcome=>50K, lr=0.1, n_epochs=15, batch_size=256);
        
END;
"""

# ------ Automatically creates necessary files for eval  ------ #
dp_boiler = """

    ENSURE: DIFFERENTIAL PRIVACY:
        EPSILON=<epsilon>, DELTA=1e-9;"""

split_constraint = constraint.split(';')[:-1]

# extract dataset for path
dataset = split_constraint[0].split(':')[-1].strip()

# create dp version
dp_split = [split_constraint[0] + ';'] + [dp_boiler] + [s + ';' for s in split_constraint[1:]]
dp_constraint = ''.join(dp_split)

# create eval versions
pattern = r'PARAM <param\d+>:'    
eval_constraint = re.sub(pattern, '', constraint)
dp_eval_constraint = re.sub(pattern, '', dp_constraint)

# saving
paths = {
    'non_dp_train': (f'{dataset}/non_dp_constraints/training_constraints/', filename + '.txt', constraint),
    'non_dp_eval': (f'{dataset}/non_dp_constraints/evaluation_constraints/', filename + '_eval.txt', eval_constraint),
    'dp_train': (f'{dataset}/dp_constraints/training_constraints/', filename + '_dp.txt', dp_constraint),
    'dp_eval': (f'{dataset}/dp_constraints/evaluation_constraints/', filename + '_dp_eval.txt', dp_eval_constraint)
}

for path, fn, cstr in paths.values():
    os.makedirs(path, exist_ok=True)
    with open(path + fn, 'w') as f:
        f.write(cstr)
