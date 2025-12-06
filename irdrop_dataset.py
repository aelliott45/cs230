import os.path as osp
import copy
import numpy as np


class IRDropDataset(object):
    def __init__(self, ann_file, dataroot, test_mode=False, **kwargs):
        super().__init__()
        self.ann_file = ann_file
        self.dataroot = dataroot
        self.test_mode = test_mode
        self.data_infos = self.load_annotations()

    def load_annotations(self):
        data_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                feature, label = line.strip().split(',')
                if self.dataroot is not None:
                    feature_path = osp.join(self.dataroot, feature)
                    label_path = osp.join(self.dataroot, label)
                data_infos.append(dict(feature_path=feature_path, label_path=label_path))
        return data_infos


    def prepare_data(self, idx):
        HOTSPOT_THRESHOLD = 0.98  #
        results = copy.deepcopy(self.data_infos[idx])
        
        feature = np.load(results['feature_path']).transpose(2, 0, 1).astype(np.float32)
        feature = np.expand_dims(feature, axis=0)  # [1, C, H, W]

        reg_label = np.load(results['label_path']).transpose(2, 0, 1).astype(np.float32).squeeze()
        # reg_label: [H, W] (IR-drop in volts)

        # classification target: hotspot if vdrop > 0.05 V
        cls_label = (reg_label > HOTSPOT_THRESHOLD).astype(np.float32)  # [H, W]

        # return both as a tuple
        targets = (reg_label, cls_label)

        return feature, targets, results['label_path']

    def __len__(self):
        return len(self.data_infos)


    def __getitem__(self, idx):
        return self.prepare_data(idx)