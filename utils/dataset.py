from torch.utils.data import Dataset
import utils.data_utils as Util
import numpy as np


class GMOD_Dataset(Dataset):
    def __init__(self, dataroot, l_resolution=16, r_resolution=64, split='train', data_len=-1):
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.split = split

        self.hr_path = Util.get_paths_from_images(
            '{}/{}/HR_{}_{}_{}'.format(dataroot, split, l_resolution, r_resolution, split))
        
        self.lr_path = Util.get_paths_from_images(
            '{}/{}/LR_{}_{}_{}'.format(dataroot, split, l_resolution, r_resolution, split))

        self.sr3_path = Util.get_paths_from_images('{}/{}/SR3_{}_{}_{}'.format(dataroot.replace('LR_HR', 'PCA_LR_HR'), split, l_resolution, r_resolution, split))
        
        self.dataset_len = len(self.hr_path)
        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)
        

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        img_HR = Util.transform2tensor(np.load(self.hr_path[index]).astype(np.float32))
        img_LR = Util.transform2tensor(np.load(self.lr_path[index]).astype(np.float32))
        img_SR = Util.transform2tensor(np.load(self.sr3_path[index]).astype(np.float32))
    
        return img_LR, img_HR, img_SR