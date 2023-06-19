from config import config
import matplotlib.pyplot as plt
from train import *

plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.plot(gan.gen_stats.losses, 'r', label='Generator Loss')
plt.plot(gan.desc_stats.losses, 'b', label='Descriminator Loss')
plt.legend()
plt.show()

fig, axes = plt.subplots(5, 2)

for i in range(5):
    photo_img, _ = next(iter(img_dl))
    pred_monet = gan.gen_ptm(photo_img.to(config.device)).cpu().detach()
    photo_img = trans(photo_img)
    pred_monet = trans(pred_monet)
    
    axes[i, 0].axis('off')
    axes[i, 0].imshow(photo_img[0].numpy().transpose(1, 2, 0))
    axes[i, 0].set_title("Input Photo")

    axes[i, 1].axis('off')
    axes[i, 1].imshow(pred_monet[0].numpy().transpose(1, 2, 0))
    axes[i, 1].set_title("Monet-esque Photo")

fig.set_figwidth(12)   
fig.set_figheight(28) 
plt.show()