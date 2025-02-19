from utils import util_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image, make_grid


def compress_images(in_path, out_path, qf):
    train_dataset = datasets.ImageFolder(in_path)
    loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Looping through it, get a batch on each loop
    # for images, labels in dataloader:
    #     pass

    # Get one batch
    hq_images, labels = next(iter(loader_train))
    lq_images = torch.empty(hq_images.shape)
    for i in hq_images.size(dim=0):
        lq_images[i] = util_image.jpeg_compress(hq_images[i],qf)
        save_image(lq_images[i], out_path+'img'+str(i)+'.png')


compress_images()