import numpy as np
import os

ab_path = os.getcwd() + '/datasets/'

np.random.seed(1)
#######################################
# Declare Parameters
SAMPLES_PARA = {}

SAMPLES_PARA['num_phantoms'] = len(os.listdir(ab_path + 'CCTA_GT/'))

# 75% training data, 15% validation data and 10% test data
# random_index = np.random.choice((np.arange(SAMPLES_PARA['num_phantoms'])+1),int(SAMPLES_PARA['num_phantoms']*25/100),False).tolist()

# SAMPLES_PARA['validation_index'] = random_index[:int(SAMPLES_PARA['num_phantoms']*15/100)]
# SAMPLES_PARA['test_index'] = random_index[:(int(SAMPLES_PARA['num_phantoms']*15/100)-1):-1]
# SAMPLES_PARA['train_index'] = [i for i in ((np.arange(SAMPLES_PARA['num_phantoms'])+1).tolist())
#                     if (i not in SAMPLES_PARA['validation_index']) & (i not in SAMPLES_PARA['test_index'])]

# SAMPLES_PARA['num_train_data'] = len(SAMPLES_PARA['train_index'])
# SAMPLES_PARA['num_validation_data'] = len(SAMPLES_PARA['validation_index'])
# SAMPLES_PARA['num_test_data'] = len(SAMPLES_PARA['test_index'])


# MY EDIT #

# SAMPLES_PARA['train_index'] = list(range(0,38))
# SAMPLES_PARA['validation_index'] = list(range(38,45))
# SAMPLES_PARA['test_index'] = list(range(45,50))

# Get the 50 OG filenames (without extension)
bp_og_folder = ab_path + 'CCTA_BP_OG/'
gt_og_folder = ab_path + 'CCTA_GT_OG/'
og_filenames = sorted([f.split('_')[0] for f in os.listdir(bp_og_folder) if f.endswith('.npy')])[:50]

# Split indices: first 38 train, next 7 validation, last 5 test
train_og = og_filenames[:38]
val_og = og_filenames[38:45]
test_og = og_filenames[45:50]

# Now, for each cropped file, assign to split based on its OG origin
bp_folder = ab_path + 'CCTA_BP/'
gt_folder = ab_path + 'CCTA_GT/'

bp_files = sorted([f for f in os.listdir(bp_folder) if f.endswith('.npy')])
gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.npy')])

def assign_indices(files, og_list):
    indices = []
    for i, og_name in enumerate(og_list):
        # Find all files that start with this OG name
        matches = [j for j, f in enumerate(files) if f.startswith(og_name)]
        indices.extend(matches)
    return indices

SAMPLES_PARA['train_index'] = assign_indices(bp_files, train_og)
SAMPLES_PARA['validation_index'] = assign_indices(bp_files, val_og)
SAMPLES_PARA['test_index'] = assign_indices(bp_files, test_og)

SAMPLES_PARA['num_train_data'] = len(SAMPLES_PARA['train_index'])
SAMPLES_PARA['num_validation_data'] = len(SAMPLES_PARA['validation_index'])
SAMPLES_PARA['num_test_data'] = len(SAMPLES_PARA['test_index'])

# print(
#     'Number of training data: ', SAMPLES_PARA['num_train_data'],
#     '\nNumber of validation data: ', SAMPLES_PARA['num_validation_data'], 
#     '\nNumber of test data: ', SAMPLES_PARA['num_test_data']
# )