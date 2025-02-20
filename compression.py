from utils import util_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image, make_grid


def compress_images(in_path, out_path, qf):
    dataset = BaseDataFolder(
        dir_path=in_path,
        transform_type='default',
        transform_kwargs={'mean':0, 'std':1.0},
        need_path=True,
        im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
    )
    print(f'Number of testing images: {len(dataset)}', flush=True)


compress_images()