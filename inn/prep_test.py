from os import path as osp
import cv2
import torch.nn.functional as F

from basicsr.data import build_dataloader, build_dataset
from basicsr.utils import imwrite, tensor2img

import data

if __name__ == '__main__':

    qf = 10

    dataset_opt = {
        'name': 'FFHQJPEGTest',
        'type': 'FFHQJPEGDataset',
        'phase': 'test',
        'gt_size': 512,
        'scale': 4,
        'qf': qf,
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
    
    test_dir = f'testdata/jpeg_qf{qf}/'
    
    for idx, val_data in enumerate(test_loader):
        img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
        img_hq = tensor2img([val_data['gt']])[0]

        save_img_path = osp.join(test_dir, f'{img_name}.jpg')
        imwrite(img_hq, save_img_path, [cv2.IMWRITE_JPEG_QUALITY, qf])