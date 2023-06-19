import numpy as np
import torch

class SampleFake:
    def __init__(self, max_imgs=50):
        assert max_imgs > 0, 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_imgs = max_imgs
        self.cur_img = 0
        self.imgs = []

    def __call__(self, imgs):
        output = []
        for img in imgs:
            if self.cur_img < self.max_imgs:
                self.imgs.append(img)
                output.append(img)
                self.cur_img += 1
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_imgs)
                    output.append(self.imgs[idx])
                    self.imgs[idx] = img
                else:
                    output.append(img)
        return output
    
def update_req_grad(models, requires_grad=True):
    """
    We need gradients to be computed for Discriminator and not to be computed for Generator
    """
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad
            
    
class AvgStats(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.losses =[]
        self.its = []
        
    def append(self, loss, it):
        self.losses.append(loss)
        self.its.append(it)