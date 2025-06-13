from os import path as osp
import sys
import cv2
import torch.nn.functional as F
import json

sys.path.insert(0, "/rds/general/user/aem21/home/FYP/compression_indigo_plus")
from basicsr.data import build_dataloader, build_dataset
from basicsr.utils import imwrite, tensor2img

import data

if __name__ == '__main__':

    qf = range(1,20)

    for factor in qf:

        dataset_opt = {
            'name': 'FFHQJPEGTest',
            'type': 'FFHQJPEGDataset',
            'phase': 'test',
            'gt_size': 512,
            'scale': 4,
            'qf': factor,
            'dataroot_gt': 'datasets/ffhq/test',
            'io_backend':
                {'type': 'disk'} 
        }

        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
                    test_set,
                    dataset_opt,
                    num_gpu=1,
                    dist=False,
                    sampler=None,
                    seed=0)
    
        test_dir = f'testdata/down_jpeg_qf{factor}/'

        encsizes = dict()
    
        for idx, val_data in enumerate(test_loader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            img_lq = tensor2img([val_data['lq']])[0]

            encsizes[img_name] = val_data['encsize'].item()

            save_img_path = osp.join(test_dir, f'{img_name}.png')
            imwrite(img_lq, save_img_path)

    
        save_json_path =  osp.join(test_dir, 'encsizes.json')
        with open(save_json_path, "w") as f:
            json.dump(encsizes, f, indent=4)

        
