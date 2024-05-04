import os
from torch.utils.data import Dataset
import json

import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE))

class PLR(Dataset):
    def __init__(self, data_root, split):
        assert split in ['train', 'val', 'trainval', 'test']
        self.data_root = data_root
        self.split = split
        self.data_files = self._load_data_files()
        self.data = self._load_data()

    def _load_data_files(self):
        # Assuming the directory structure separates training and validation data
        split_path = os.path.join(self.data_root, self.split)
        data_files = [os.path.join(split_path, fname) for fname in os.listdir(split_path) if fname.endswith('.json')]
        return data_files

    def _load_data(self):
        data = []
        for file_path in self.data_files:
            with open(file_path, 'r') as f:
                data_dict = json.load(f)
                data.append(data_dict)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Implement this method to return a single data point at index `idx`.
        # This might involve transforming the raw data into a format suitable for your model.
        data_item = self.data[idx]
        pts = data_item['pc_full']
        gt_2d_pts = data_item['label_pc_2d_full']
        gt_labels = data_item['label_pc_2d_segmentation']
        data_dict = {
            'pts': pts,
            'gt_2d_pts': gt_2d_pts,
            'gt_labels': gt_labels
        }

        return data_dict