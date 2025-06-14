import cv2
import os
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.degradations import add_jpg_compression
from basicsr.data.transforms import augment, mod_crop, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class FFHQJPEGDataset(data.Dataset):
    """Example dataset.

    1. Read GT image
    2. Generate LQ (Low Quality) image with cv2 bicubic downsampling and JPEG compression

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(FFHQJPEGDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.qf = opt['qf']
        self.scale = self.opt['scale']
        self.gt_size = self.opt['gt_size']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt']
        # it now only supports folder mode, for other modes such as lmdb and meta_info file, please see:
        # https://github.com/xinntao/BasicSR/blob/master/basicsr/data/
        self.paths = [os.path.join(self.gt_folder, v) for v in list(scandir(self.gt_folder))]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = mod_crop(img_gt, self.scale)
        
        # Adjusting gt size
        h_old,w_old = img_gt.shape[0:2] 
        if not (h_old == self.gt_size and w_old == self.gt_size):
            img_gt = cv2.resize(img_gt, (self.gt_size,) * 2, interpolation=cv2.INTER_CUBIC)
            

        # Downsampling to match effect of forward INN (set scale to 1 to avoid)
        img_lq = cv2.resize(img_gt, (self.gt_size // self.scale,) *2, interpolation=cv2.INTER_CUBIC)

        # Degrade image through JPEG compression
        img_lq, encsize = add_jpg_compression(img_lq, quality=self.qf, size=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, self.gt_size, self.scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # Limit image to range [0,1]
        img_gt = torch.clamp((img_gt * 255.0).round(), 0, 255) / 255.
        img_lq = torch.clamp((img_lq * 255.0).round(), 0, 255) / 255.

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path, 'encsize': encsize}

    def __len__(self):
        return len(self.paths)