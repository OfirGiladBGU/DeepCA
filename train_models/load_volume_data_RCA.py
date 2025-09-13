import numpy as np
import os
import torch
import pathlib

ab_path = os.getcwd() + '/datasets/'

# class Dataset(torch.utils.data.Dataset):
#     'Characterizes a dataset for PyTorch'
#     def __init__(self, list_IDs):
#         # 'Initialization'
#         self.list_IDs = list_IDs

#     def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.list_IDs)

#     def __getitem__(self, index):
#         # 'Generates one sample of data'
#         # Select sample
#         ID = self.list_IDs[index]

#         # Load data and get label
#         data_file = ab_path + 'CCTA_BP/recon_' + str(ID) + '.npy'
#         data = np.transpose(np.load(data_file)[:,:,:,np.newaxis])
#         label_file = ab_path + 'CCTA_GT/' + str(ID) + '.npy'
#         label = np.transpose(np.load(label_file)[:,:,:,np.newaxis])

#         return torch.from_numpy(data), torch.from_numpy(label)
  

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs
        self.data_paths = [ab_path + 'CCTA_BP', ab_path + 'CCTA_GT']

        self.data_files1 = pathlib.Path(self.data_paths[0]).rglob("*.*")
        self.data_files1 = sorted(self.data_files1)
        self.data_files1 = [f for i, f in enumerate(self.data_files1) if i in self.list_IDs]

        self.data_files2 = pathlib.Path(self.data_paths[1]).rglob("*.*")
        self.data_files2 = sorted(self.data_files2)
        self.data_files2 = [f for i, f in enumerate(self.data_files2) if i in self.list_IDs]

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, index):
        data_file1 = str(self.data_files1[index])
        numpy_3d_data1 = np.load(str(data_file1))
        numpy_3d_data1 = numpy_3d_data1.astype(np.float32)
        numpy_3d_data1 = torch.from_numpy(numpy_3d_data1).clone().unsqueeze(0)

        # 1 3D input + 1 3D target
        data_file2 = str(self.data_files2[index])
        numpy_3d_data2 = np.load(str(data_file2))
        numpy_3d_data2 = numpy_3d_data2.astype(np.float32)
        numpy_3d_data2 = torch.from_numpy(numpy_3d_data2).clone().unsqueeze(0)

        return numpy_3d_data1, numpy_3d_data2


class DatasetV2(torch.utils.data.Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs
        self.data_paths = [ab_path + 'CCTA_BP', ab_path + 'CCTA_GT']

        self.data_files1 = pathlib.Path(self.data_paths[0]).rglob("*.*")
        self.data_files1 = sorted(self.data_files1)
        self.data_files1 = [f for i, f in enumerate(self.data_files1) if i in self.list_IDs]

        self.data_files2 = pathlib.Path(self.data_paths[1]).rglob("*.*")
        self.data_files2 = sorted(self.data_files2)
        self.data_files2 = [f for i, f in enumerate(self.data_files2) if i in self.list_IDs]

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, index):
        data_file1 = str(self.data_files1[index])
        numpy_3d_data1 = np.load(str(data_file1))
        numpy_3d_data1 = numpy_3d_data1.astype(np.float32)
        numpy_3d_data1 = torch.from_numpy(numpy_3d_data1).clone().unsqueeze(0)

        # 1 3D input + 1 3D target
        data_file2 = str(self.data_files2[index])
        numpy_3d_data2 = np.load(str(data_file2))
        numpy_3d_data2 = numpy_3d_data2.astype(np.float32)
        numpy_3d_data2 = torch.from_numpy(numpy_3d_data2).clone().unsqueeze(0)

        return numpy_3d_data1, numpy_3d_data2, pathlib.Path(data_file1).stem
