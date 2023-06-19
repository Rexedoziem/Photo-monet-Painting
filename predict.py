from train import *
import shutil
import os
class PhotoDataset(data.Dataset):
    def __init__(self, photo_dir, size=(256, 256), normalize=True):
        super().__init__()
        self.photo_dir = photo_dir
        self.photo_idx = dict()
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()                               
            ])
        for i, fl in enumerate(os.listdir(self.photo_dir)):
            self.photo_idx[i] = fl

    def __getitem__(self, idx):
        photo_path = os.path.join(self.photo_dir, self.photo_idx[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        return photo_img
    
    def __len__(self):
        return len(self.photo_idx.keys())
    

ph_ds = PhotoDataset('../input/gan-getting-started/photo_jpg/')
ph_dl = DataLoader(ph_ds, batch_size=1, pin_memory=True)

pil_trans = transforms.ToPILImage()

def unnorm(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
        
    return img
        
    return img

t = tqdm(ph_dl, leave=False, total=ph_dl.__len__())
for i, photo in enumerate(t):
    with torch.no_grad():
        pred_monet = gan.gen_ptm(photo.to(config.device)).cpu().detach()
    pred_monet = unnorm(pred_monet)
    img = pil_trans(pred_monet[0]).convert("RGB")
    img.save("../images/" + str(i+1) + ".jpg")


shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")