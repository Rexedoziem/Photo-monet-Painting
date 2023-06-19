import torchvision.transforms as transforms
from config import config
from torchvision.utils import make_grid
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from dataset import ImageDataset

img_ds = ImageDataset(config.monet_dir, config.photo_dir)
img_dl = DataLoader(img_ds, config.batch_size, num_workers=3, shuffle=True, pin_memory=True)
photo_img, monet_img = next(iter(img_dl))

trans = transforms.Compose([transforms.Normalize(mean=[-1], std=[2])])

photo_img = trans(photo_img)
monet_img = trans(monet_img)

def denorm(img_tensors):
    return img_tensors * config.stats[1][0] + config.stats[0][0]

def show_images(images, nmax=2):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=2).permute(1,2,0))
    
def show_batch(dl, nmax=2):
    for images, _ in dl:
        show_images(images, nmax)
        break