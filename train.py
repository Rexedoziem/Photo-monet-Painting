from config import config
import torch.nn as nn
import time
import config
import numpy as np
import torch
from tqdm import tqdm
from utils import SampleFake, AvgStats, update_req_grad
from models import init_weights
import itertools
from models import Discriminator, Generator
from config import config
import torch.optim as optim
import torchvision.utils as vutils
from image_transforms import *
import gc
from tqdm.notebook import tqdm

def save_checkpoint(state, save_path):
    torch.save(state, save_path)

class CycleGAN(object):
    def __init__(self,in_ch, out_ch, epochs, device, start_lr=2e-4, lmbda=10, idt_coef=0.5, decay_epoch=0):
        self.epochs = epochs
        self.decay_epoch = decay_epoch if decay_epoch > 0 else int(self.epochs/2)
        self.lmbda = lmbda
        self.idt_coef = idt_coef
        self.device = device
        self.gen_mtp = Generator(in_ch, out_ch)
        self.gen_ptm = Generator(in_ch, out_ch)
        self.desc_m = Discriminator(in_ch)
        self.desc_p = Discriminator(in_ch)
        self.init_models()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adam_gen = torch.optim.RAdam(
            itertools.chain(self.gen_mtp.parameters(), self.gen_ptm.parameters()), 
            lr = start_lr, 
            betas=(0.5, 0.999)
        )
        self.adam_desc = torch.optim.RAdam(
            itertools.chain(self.desc_m.parameters(), self.desc_p.parameters()), 
            lr = start_lr, 
            betas=(0.5, 0.999)
        )
        self.sample_monet = SampleFake()
        self.sample_photo = SampleFake()

        self.gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_gen, 
                                                              lr_lambda=lambda epoch: 1 - max(0, epoch-decay_epoch) / (epochs-decay_epoch))
        self.desc_lr_sched = torch.optim.lr_scheduler.LambdaLR(self.adam_desc, 
                                                               lr_lambda=lambda epoch: 1 - max(0, epoch-decay_epoch) / (epochs-decay_epoch))
        self.gen_stats = AvgStats()
        self.desc_stats = AvgStats()
        
    def init_models(self):
        init_weights(self.gen_mtp)
        init_weights(self.gen_ptm)
        init_weights(self.desc_m)
        init_weights(self.desc_p)
        self.gen_mtp = self.gen_mtp.to(self.device)
        self.gen_ptm = self.gen_ptm.to(self.device)
        self.desc_m = self.desc_m.to(self.device)
        self.desc_p = self.desc_p.to(self.device)
        
    def train(self, photo_dl):
        for epoch in range(self.epochs):
            start_time = time.time()
            avg_gen_loss = 0.0
            avg_desc_loss = 0.0
            t = tqdm(photo_dl, leave=False, total=photo_dl.__len__())
            
            for i, (photo_real, monet_real) in enumerate(t):
                  # gc.collect()
                photo_img, monet_img = photo_real.to(config.device), monet_real.to(config.device)
                # Disable gradient computation for Generator
                update_req_grad([self.desc_m, self.desc_p], False)

                    
                self.adam_gen.zero_grad()

                    # Forward pass through generator
                fake_photo = self.gen_mtp(monet_img)
                fake_monet = self.gen_ptm(photo_img)

                cycl_monet = self.gen_ptm(fake_photo)
                cycl_photo = self.gen_mtp(fake_monet)

                id_monet = self.gen_ptm(monet_img)
                id_photo = self.gen_mtp(photo_img)

                # generator losses - identity, Adversarial, cycle consistency
                idt_loss_monet = self.l1_loss(id_monet, monet_img) * self.lmbda * self.idt_coef
                idt_loss_photo = self.l1_loss(id_photo, photo_img) * self.lmbda * self.idt_coef

                cycle_loss_monet = self.l1_loss(cycl_monet, monet_img) * self.lmbda
                cycle_loss_photo = self.l1_loss(cycl_photo, photo_img) * self.lmbda

                monet_desc = self.desc_m(fake_monet)
                photo_desc = self.desc_p(fake_photo)

                real = torch.ones(monet_desc.size()).to(self.device)

                adv_loss_monet = self.mse_loss(monet_desc, real)
                adv_loss_photo = self.mse_loss(photo_desc, real)

                # total generator loss
                total_gen_loss = cycle_loss_monet + adv_loss_monet\
                              + cycle_loss_photo + adv_loss_photo\
                              + idt_loss_monet + idt_loss_photo

                avg_gen_loss += total_gen_loss.item()

                # backward pass
                total_gen_loss.backward()
                self.adam_gen.step()

                # Forward pass through Descriminator
                update_req_grad([self.desc_m, self.desc_p], True)

    
                self.adam_desc.zero_grad()

                fake_monet = self.sample_monet([fake_monet.cpu().data.numpy()])[0]
                fake_photo = self.sample_photo([fake_photo.cpu().data.numpy()])[0]
                fake_monet = torch.tensor(fake_monet).to(self.device)
                fake_photo = torch.tensor(fake_photo).to(self.device)

               
                monet_desc_real = self.desc_m(monet_img)
                
                
                monet_desc_real = torch.unsqueeze(monet_desc_real, 2)  # Add an extra dimension at index 2
            
                monet_desc_fake = self.desc_m(fake_monet)
                
                monet_desc_fake = torch.unsqueeze(monet_desc_fake, 2)  # Add an extra dimension at index 2
            
                photo_desc_real = self.desc_p(photo_img)
                
                photo_desc_real = torch.unsqueeze(photo_desc_real, 2)  # Add an extra dimension at index 2
               
                photo_desc_fake = self.desc_p(fake_photo)
                
                photo_desc_fake = torch.unsqueeze(photo_desc_fake, 2)  # Add an extra dimension at index 2

                real = torch.full_like(monet_desc_real, 1., device=self.device)
            
                fake = torch.full_like(monet_desc_fake, 1., device=self.device)

                # Descriminator losses
                # --------------------
                monet_desc_real_loss = self.mse_loss(monet_desc_real, real)
                monet_desc_fake_loss = self.mse_loss(monet_desc_fake, fake)
                photo_desc_real_loss = self.mse_loss(photo_desc_real, real)
                photo_desc_fake_loss = self.mse_loss(photo_desc_fake, fake)

                monet_desc_loss = (monet_desc_real_loss + monet_desc_fake_loss) / 2
                photo_desc_loss = (photo_desc_real_loss + photo_desc_fake_loss) / 2
                total_desc_loss = monet_desc_loss + photo_desc_loss
                avg_desc_loss += total_desc_loss.item()

                # Backward
                monet_desc_loss.backward()
                photo_desc_loss.backward()
                self.adam_desc.step()

                t.set_postfix(gen_loss=total_gen_loss.item(), desc_loss=total_desc_loss.item())

            save_dict = {
                'epoch': epoch+1,
                'gen_mtp': gan.gen_mtp.state_dict(),
                'gen_ptm': gan.gen_ptm.state_dict(),
                'desc_m': gan.desc_m.state_dict(),
                'desc_p': gan.desc_p.state_dict(),
                'optimizer_gen': gan.adam_gen.state_dict(),
                'optimizer_desc': gan.adam_desc.state_dict()
            }

            save_checkpoint(save_dict, 'current.ckpt')
            
            avg_gen_loss /= photo_dl.__len__()
            avg_desc_loss /= photo_dl.__len__()
            time_req = time.time() - start_time
            
            self.gen_stats.append(avg_gen_loss, time_req)
            self.desc_stats.append(avg_desc_loss, time_req)
            
            print("Epoch: (%d) | Generator Loss:%f | Discriminator Loss:%f" % 
                                                (epoch+1, avg_gen_loss, avg_desc_loss))
      
            self.gen_lr_sched.step()
            self.desc_lr_sched.step()
    torch.cuda.empty_cache()
gan = CycleGAN(3, 3, 100, config.device)

# Save before train
save_dict = {
    'epoch': 0,
    'gen_mtp': gan.gen_mtp.state_dict(),
    'gen_ptm': gan.gen_ptm.state_dict(),
    'desc_m': gan.desc_m.state_dict(),
    'desc_p': gan.desc_p.state_dict(),
    'optimizer_gen': gan.adam_gen.state_dict(),
    'optimizer_desc': gan.adam_desc.state_dict()
}

save_checkpoint(save_dict, 'init.ckpt')
gan.train(img_dl)