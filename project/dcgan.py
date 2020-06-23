# -*- coding: utf-8 -*-

import numpy as np
import os
from pathlib import Path
import random
import shlex
import subprocess
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor as pil_to_tensor
from torchvision.transforms.functional import to_pil_image as tensor_to_pil

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import PIL
import matplotlib.pyplot as plt

from IPython.display import display
import pytorch_lightning as pl

# Turn off the skip of loading large images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



PIXEL_SIZE = 256
BATCH_SIZE = 16
VERSION = 1.4


# 1. Load png images

images = []

for i in tqdm(range(10000)):
    img = PIL.Image.open(f'data/{i:05d}.jpg')
    img.load()  # Eplicit execution to avoid 'Too many open files' OSError 
    images.append(img)


# 2. DataLoader

class RealCXRDataSet(torch.utils.data.Dataset):
    def __init__(self, images, pixel_size=64):
        """
        Input
        -----
        images: list of PIL.Image.Image
        """
        self.images = images
        self.pixel_size = pixel_size
        
        # Image transformation
        self.norm_param = ((0.5,), (0.5,))    # (mean, std)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*self.norm_param)
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        """
        PIL.Image.Image -> torch.Tensor
        """
        img = self.images[index]
        img = torchvision.transforms.Resize(self.pixel_size)(img)
        img = self.transform(img)    # Normalize pixel values to the range [-1, 1] & PIL.Image -> torch.tensor

        return img

ds = RealCXRDataSet(images, PIXEL_SIZE)
dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)




# 3. DCGAN

### 3-1. Generator

class Generator(nn.Module):
    def __init__(self, z_dim=20, hidden_channels=64, leaky_relu=True):
        """
        inputs
        ------
        z_dim (int):
            The dimensionality of noise vector.
        hidden_channels (int):
            The hidden channels size.
        leaky_relu (bool):
            Whether to use LeakyReLU as activation function.
            If set False, ReLU will be used instead, although LeakyReLU is recommended for better performance.
        """
        super().__init__()
        self.z_dim = z_dim
        self.hidden_channels = hidden_channels
        self.leaky = leaky_relu
        
        def g_body_layer(in_channel, out_channel):
            layer = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.LeakyReLU(0.1, inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )
            return layer


        # 1x1 image -> 4x4 image
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, hidden_channels * 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(hidden_channels * 32),
            nn.LeakyReLU(0.1, inplace=True) if self.leaky else nn.ReLU(inplace=True)
        )

        # 4x4 image -> 8x8 image
        self.layer2 = g_body_layer(hidden_channels * 32, hidden_channels * 16)

        # 8x8 image -> 16x16 image
        self.layer3 = g_body_layer(hidden_channels * 16, hidden_channels * 8)

        # 16x16 image -> 32x32 image
        self.layer4 = g_body_layer(hidden_channels * 8, hidden_channels * 4)
        
        # 32x32 image -> 64x64 image
        self.layer5 = g_body_layer(hidden_channels * 4, hidden_channels * 2)
        
        # 64x64 image -> 128x128 image
        self.layer6 = g_body_layer(hidden_channels * 2, hidden_channels)
        
        # 128x128 image -> 256x256 image & normalize to [-1, 1]
        self.last = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh())
        
        # Initialize parameters
        self._initialize()
        

    def _initialize(self):
        """
        Caution: This method will reset all the weights of this network. Do not use this after training.
        """
        
        def _initialize_module(module):
            conv_mean, conv_std, bn_mean, bn_std, bias = 0.0, 0.02, 1.0, 0.02, 0
        
            if module.__class__.__name__.find('Conv') != -1:
                # initial params for Conv2d & ConvTranspose2d
                nn.init.normal_(module.weight.data, conv_mean, conv_std)
                nn.init.constant_(module.bias.data, bias)
                
            elif module.__class__.__name__.find('BatchNorm') != -1:
                # initial params for BatchNorm2d
                nn.init.normal_(module.weight.data, bn_mean, bn_std)
                nn.init.constant_(module.bias.data, bias)

        self.apply(_initialize_module)
        
        
    def get_device(self):
        """
        Returns the device on which this module is located.
        """
        return self.state_dict()['layer1.0.weight'].device
                

    def forward(self, z):
        """
        Generate image(s) given a noise vector z.
        
        inputs
        ------
        z (torch.Tensor):
            Noise vector.
            But in practice, this must be given as 1x1 pixel images with self.z_dim channels.
            So z must be a 4D tensor in shape (n_batch, self.z_dim, 1, 1).

        outputs
        -------
        out (torch.Tensor):
            4D tensor for output image(s), in shape (n_batch, 1, 256, 256).
        """
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.last(out)
        return out

    
    def generate_something(self, n, device=None):
        """
        End-to-end image generation with no need to give a noise vector explicitly.

        inputs
        ------
        n (int):
            The number of images desired.
        device (torch.device, optional):
            The device to run this method.
            Leave this None to use the device on which this module is located.

        outputs
        -------
        out (torch.Tensor):
            4D tensor for output image(s), in shape (n, 1, 256, 256).
        """
        input_z = torch.randn(n, self.z_dim).unsqueeze(-1).unsqueeze(-1)
        
        if device is None:
            input_z = input_z.to(self.get_device())
        else:
            input_z = input_z.to(device)

        return self.forward(input_z)





### 3-2. Discriminator

class Discriminator(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        def d_body_layer(in_channel, out_channel):
            layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )
            return layer

        
        # pixel: 256x256 -> 128x128
        self.layer1 = d_body_layer(1, hidden_channels)
        
        # pixel: 128x128 -> 64x64
        self.layer2 = d_body_layer(hidden_channels, hidden_channels * 2)

        # pixel: 64x64 -> 32x32
        self.layer3 = d_body_layer(hidden_channels * 2, hidden_channels * 4)

        # pixel: 32x32 -> 16x16
        self.layer4 = d_body_layer(hidden_channels * 4, hidden_channels * 8)

        # pixel: 16x16 -> 8x8
        self.layer5 = d_body_layer(hidden_channels * 8, hidden_channels * 16)

        # pixel: 8x8 -> 4x4
        self.layer6 = d_body_layer(hidden_channels * 16, hidden_channels * 32)

        # pixel: 4x4 -> 1x1 (logits)
        self.last = nn.Conv2d(hidden_channels * 32, 1, kernel_size=4, stride=1)
        
        # Initialize params
        self._initialize()


    def _initialize(self):
        """
        Caution: This method will reset all the weights of this network. Do not use this after training.
        """
        
        def _initialize_module(module):
            conv_mean, conv_std, bn_mean, bn_std, bias = 0.0, 0.02, 1.0, 0.02, 0
        
            if module.__class__.__name__.find('Conv') != -1:
                # initial params for Conv2d & ConvTranspose2d
                nn.init.normal_(module.weight.data, conv_mean, conv_std)
                nn.init.constant_(module.bias.data, bias)
                
            elif module.__class__.__name__.find('BatchNorm') != -1:
                # initial params for BatchNorm2d
                nn.init.normal_(module.weight.data, bn_mean, bn_std)
                nn.init.constant_(module.bias.data, bias)

        self.apply(_initialize_module)
        

    def forward(self, img):
        """
        Discriminate whether input images are Real (1) or Fake (0).
        
        inputs
        ------
        img (torch.Tensor):
            4D tensor for gray scale images, in size (n_batch, 1, 256, 256).

        outputs
        ------
        out (torch.Tensor):
            Vector for logits for real images, in size (n_batch).
            Pass this output to sigmoid function for conversion to probability.
        """
        out = self.layer1(img)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.last(out)
        return out.squeeze()



### 3-3. DCGAN

class DCGANModule(pl.LightningModule):
    def __init__(
        self, 
        discriminator,
        generator, 
        loss='hinge',
        label_smoothing=True,
        real_label_range=(0.8, 1.0),
        fake_label_range=(0.0, 0.2)
    ):
        """
        Combined DCGAN Module.

        inputs
        ------
        a
        """
        super().__init__()
        self.D = discriminator
        self.G = generator
        self.label_smoothing = label_smoothing
        self.real_label_range = real_label_range
        self.fake_label_range = fake_label_range
        
        self.loss = loss
        assert(self.loss in ('bce', 'hinge'))

        self.lossfunc = nn.BCELoss(reduction='mean')


    def get_device(self):
        return self.G.state_dict()['layer1.0.weight'].device


    def forward(self, inputs):
        # Run Discriminator
        logits = self.D(inputs)
        return logits

    
    def training_step(self, batch, batch_nb, optimizer_idx):
        # Process on individual mini-batchs
        """
        (batch) -> (dict or OrderedDict)
        # Caution: key for loss function must exactly be 'loss'.
        """

        n_batch = batch.size()[0]

        
        # Avoid n_batch == 1 because it will cause an error in batch normalization in the generator
        if n_batch == 1:
            batch = torch.cat([batch, batch])
            n_batch = 2
        
        
        # Prepare real & fake samples
        real_sample = batch
        real_sample = real_sample.to(self.get_device())

        fake_sample = self.G.generate_something(n_batch, device=self.get_device())
        fake_sample = fake_sample.to(self.get_device())
        

        # Run Discriminator
        logits_real = self.forward(real_sample)
        logits_fake = self.forward(fake_sample)
        

        # Prepare labels
        if self.label_smoothing:
            # Smoothed labels from uniform distribution
            T_real = torch.rand(n_batch) * (self.real_label_range[1] - self.real_label_range[0]) + self.real_label_range[0]
            T_fake = torch.rand(n_batch) * (self.fake_label_range[1] - self.fake_label_range[0]) + self.fake_label_range[0]
        else:
            T_real = torch.tensor([1.] * n_batch)
            T_fake = torch.tensor([0.] * n_batch)
            
        T_real = T_real.to(self.get_device())
        T_fake = T_fake.to(self.get_device())
        

        # Calculate loss
        if self.loss == 'bce':
            loss_d_real = self.lossfunc(nn.Sigmoid()(logits_real), T_real)
            loss_d_fake = self.lossfunc(nn.Sigmoid()(logits_fake), T_fake)
            loss_d = loss_d_real + loss_d_fake
            loss_g = self.lossfunc(nn.Sigmoid()(logits_fake), T_real)

        elif self.loss == 'hinge':
            loss_d_real = torch.min(logits_real - 1, torch.zeros_like(logits_real)) * (-1.0)
            loss_d_fake = torch.min(logits_fake * (-1.0) - 1, torch.zeros_like(logits_fake)) * (-1.0)
            loss_d = loss_d_real + loss_d_fake
            loss_g = logits_fake * (-1.0)

        if optimizer_idx == 0:
            # Discriminator
            loss = loss_d
            
        elif optimizer_idx == 1:
            # Generator
            loss = loss_g

        returns = {'loss':loss, 'loss_d':loss_d, 'loss_g':loss_g}
        return returns
    

    def training_step_end(self, outputs):
        """
        outputs(dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
        loss = torch.mean(outputs['loss'])
        loss_d = torch.mean(outputs['loss_d']) 
        loss_g = torch.mean(outputs['loss_g'])
        
        progress_bar = {'train_loss_d':loss_d, 'train_loss_g':loss_g}
        returns = {'loss':loss, 'loss_d':loss_d, 'loss_g':loss_g, 'progress_bar':progress_bar}
        return returns
    
    
    def training_epoch_end(self, outputs):
        """
        outputs(list of dict) -> loss(dict or OrderedDict)
        # Caution: key must exactly be 'loss'.
        """
        if len(outputs) > 1:
            loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
            loss_d = torch.mean(torch.tensor([output['loss_d'] for output in outputs]))
            loss_g = torch.mean(torch.tensor([output['loss_g'] for output in outputs]))
            # PY_real = torch.cat([output['PY_real'] for output in outputs])
            # PY_fake = torch.cat([output['PY_fake'] for output in outputs])
        else:
            loss = torch.mean(outputs[0]['loss'])
            loss_d = torch.mean(outputs[0]['loss_d'])
            loss_g = torch.mean(outputs[0]['loss_g'])
            # PY_real = outputs[0]['PY_real']
            # PY_fake = outputs[0]['PY_fake']
        
        # PY_real_arr = PY_real.detach().cpu().numpy()
        # PY_fake_arr = PY_fake.detach().cpu().numpy() 
        # acc = accuracy_score(np.concatenate([PY_real_arr >= 0.5, PY_fake_arr < 0.5]), np.array([1] * (len(PY_real_arr) + len(PY_fake_arr))))
        
        # progress_bar = {'train_loss':loss, 'acc':acc}
        progress_bar = {'train_loss_d':loss_d, 'train_loss_g':loss_g}
        
        returns = {'loss':loss, 'progress_bar':progress_bar}
        return returns

    
    def configure_optimizers(self):
        LR_G, LR_D, BETA_1, BETA_2 = 1e-4, 4e-4, 0.0, 0.9
        optim_d = optim.Adam(self.D.parameters(), LR_D, [BETA_1, BETA_2])
        optim_g = optim.Adam(self.G.parameters(), LR_G, [BETA_1, BETA_2])
        return [optim_d, optim_g]
    
    def train_dataloader(self):
        return dl

device = torch.device('cuda:3')
G = Generator(leaky_relu=True)
D = Discriminator(G.hidden_channels)
dcgan = DCGANModule(D, G, loss='hinge', label_smoothing=True).to(device)


trainer = pl.Trainer(
    max_epochs=1,
    num_sanity_val_steps=0,
    train_percent_check=1
)

def denormalize(tensor):
    mean = 0.5
    std = 0.5
    return (tensor * std) + mean

def show_fake(row, col):
    """
    (int, int) -> PIL.Image.Image
    """
    n = row * col
    img_tensors = dcgan.G.generate_something(n).squeeze().cpu()
    
    # Split into rows
    rows = torch.chunk(img_tensors, row)
    
    # Concat within each row
    img_tensors = torch.cat([torch.cat([tensor for tensor in row], axis=1) for row in rows])

    # Result
    return tensor_to_pil(denormalize(img_tensors))

for i in range(15):
    trainer.fit(dcgan)
    show_fake(1, 6).save(f'dcgan_v1.4_epoch{i+1:02d}.jpg', quality=95)
