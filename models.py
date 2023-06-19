import torch.nn as nn
import torch

def Upsample(in_ch, out_ch, dropout_ratio=0.5):
    return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.Dropout(dropout_ratio),
            nn.SiLU()
        )

def Convlayer(in_ch, out_ch, kernel_size=3, stride=2, use_pad=True):
    if use_pad:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 1, bias=True)
    else:
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, 0, bias=True)

    actv = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    norm = nn.InstanceNorm2d(out_ch) #

    return nn.Sequential(conv, norm, actv)

class Resblock(nn.Module):
    def __init__(self, in_features, use_dropout=True, dropout_ratio=0.5):
        super().__init__()
        layers = list()
        layers.append(nn.ReflectionPad2d(1))    # making 1-layer padding near input tensor
        layers.append(Convlayer(in_features, in_features, 3, 1, use_pad=False))
        layers.append(nn.Dropout(dropout_ratio))
        layers.append(nn.ReflectionPad2d(1))
        layers.append(nn.Conv2d(in_features, in_features, 3, 1, padding=0, bias=True))
        layers.append(nn.InstanceNorm2d(in_features))
        layers.append(nn.SiLU())
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.res(x)
    
class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, num_res_blocks=6):
        super().__init__()
        model = list()
        model.append(nn.ReflectionPad2d(3))    # making 3-layer padding near input tensor
        model.append(Convlayer(in_ch, 64, 7, 1, use_pad=False))
        model.append(Convlayer(64, 128, 3, 2))
        model.append(Convlayer(128, 256, 3, 2))
        for _ in range(num_res_blocks):
            model.append(Resblock(256))
        model.append(Upsample(256, 128))
        model.append(Upsample(128, 64))
        model.append(nn.ReflectionPad2d(3))
        model.append(nn.Conv2d(64, out_ch, kernel_size=7, padding=0)) # taking away old padding
        model.append(nn.Tanh()) # activation function

        self.gen = nn.Sequential(*model)

    def forward(self, x):
        return self.gen(x) # returning our ready generator model
    

class Discriminator(nn.Module):
    def __init__(self, in_ch, num_layers=4):
        super().__init__()
        model = list()
        model.append(nn.Conv2d(in_ch, 64, 4, stride=2, padding=1))
        model.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        for i in range(1, num_layers):
            in_chs = 64 * 2**(i-1)
            out_chs = in_chs * 2
            if i == num_layers-1:
                model.append(Convlayer(in_chs, out_chs, 4, 1)) # for the penultimate case stride will be 1
            else:
                model.append(Convlayer(in_chs, out_chs, 4, 2)) # stride 2
        model.append(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)) # and the same situation for the last case
        self.disc = nn.Sequential(*model)

    def forward(self, x):
        return self.disc(x) # returning our ready dicriminator model
    

def init_weights(net, gain=0.02):
    def init_func(m):
        """
        Checking if our classes Generator and Discriminatore have methods weight or bias, in this case we fill
        the input weight tensor from weight.data or bias.data with values drawn from the normal distribution 
        N(mean, std**2) and the input bias tensor we fill with 0
        """
        classname = m.__class__.__name__ # here we're getting our class name (Generator or Discriminator)
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1): 
            torch.nn.init.normal_(m.weight.data, 0.0, gain) # initalizing weight tensor
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias.data) # filling bias tensor with zeroes
        elif classname.find('BatchNorm2d') != -1:
            torch.nninit.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.zeros_(m.bias.data)
    net.apply(init_func) # applying function to every submodule of our hand-made net