import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import os

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.init_size = opt.img_size // 4
        self.cn1 = 32
        self.img_size = opt.img_size
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1*(self.init_size ** 2)))
        self.l1p = nn.Sequential(nn.Linear(opt.latent_dim, self.cn1*(opt.img_size**2)))

        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),
        )

        self.conv_blocks_02p = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(1, self.cn1//4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )

        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, opt.channels, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.l1p(z) # 100 -> 32 * 128 * 128
        out = out.view(out.shape[0], self.cn1, self.img_size, self.img_size)
        out01 = self.conv_blocks_01p(out)

        out1 = self.conv_blocks_1(out01)
        return out1

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.cn1 = 32
        self.down_size0 = 64
        self.down_size = 32

        self.pre = nn.Sequential(
            nn.Linear(opt.img_size ** 2, self.down_size0 ** 2),
        )

        self.down = nn.Sequential(
            nn.Conv2d(opt.channels, self.cn1, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, self.cn1//2, 3, 1, 1),
            nn.BatchNorm2d(self.cn1//2),
            nn.ReLU(),
        )

        self.conv_block02p = nn.Sequential(
            nn.Upsample(scale_factor=self.down_size),
            nn.Conv2d(1, self.cn1//4, 3, stride=1, padding=1),
            nn.BatchNorm2d(self.cn1//4),
            nn.ReLU(),
        )

        down_dim = 16 * (self.down_size) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 16),
            nn.BatchNorm1d(16, 0.8),
            nn.ReLU(),
            nn.Linear(16, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU()
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(16, 16, 3, 1, 1), # didnt use label here
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 1, 1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, opt.channels, 2, 1, 0),
            nn.Sigmoid(),
        )

    def forward(self, img):
        b,c,_,_ =img.shape
        img = img.flatten(2,3)
        out00 = self.pre(img) # 128 * 128 -> 64 * 64 
        out00 = out00.view(b,c,self.down_size0, self.down_size0)
        # print(out00.shape)
        out01 = self.down(out00)
        # print(out01.shape)

        out = self.fc(out01.view(out01.size(0), -1))
        # print(out.shape)
        out = self.up(out.view(out.size(0), 16, self.down_size, self.down_size))
        return out

class sciGANModel():
    def __init__(self, opt):
        self.opt = opt
        self.g = Generator(opt)
        self.d = Discriminator(opt)
        self.k = 0
        self.lambda_k = 0.001
        self.gamma = 0.95
        self.optim_G = torch.optim.Adam(self.g.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optim_D = torch.optim.Adam(self.d.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
        if len(opt.gpu_ids) > 1:
            self.g = nn.DataParallel(self.g).cuda()
            self.d = nn.DataParallel(self.d).cuda()
        
        elif len(opt.gpu_ids) > 0:
            self.g = self.g.cuda()
            self.d = self.d.cuda()
        
        self.loss_stat = {}

    def set_input(self, input, mode='train'):
        self.z = Variable(input['z']).float().cuda()
        if mode == 'train':
            self.img = Variable(input['img']).cuda()
    
    def forward(self):
        self.g.train()
        self.fake_img = self.g(self.z)

    def inference(self):
        self.g.eval()
        fake_img = self.g(self.z)
        return fake_img
    
    def update_optimzer(self):
        self.optim_G.zero_grad()
        fake_img = self.g(self.z)
        g_loss = torch.mean(torch.abs(self.d(fake_img)-fake_img))
        # g_loss = torch.mean(torch.abs(self.d(fake_img) - self.img))
        g_loss.backward()
        self.optim_G.step()

        self.optim_D.zero_grad()
        d_real = self.d(self.img)
        d_fake = self.d(fake_img.detach())
        d_loss_real = torch.mean(torch.abs(d_real - self.img))
        d_loss_fake = torch.mean(torch.abs(d_fake - fake_img.detach()))

        d_loss = d_loss_real - self.k * d_loss_fake # the default kt in the code is set to 0
        d_loss.backward()
        self.optim_D.step()

        diff = torch.mean(self.gamma * d_loss_real - d_loss_fake)
        self.k = self.k + self.lambda_k * np.asscalar(diff.detach().data.cpu().numpy())
        self.k = min(max(self.k, 0), 1)

        self.loss_stat['gloss'] = g_loss
        self.loss_stat['dloss_real'] = d_loss_real
        self.loss_stat['dloss_fake'] = d_loss_fake

    def get_cur_loss(self):
        return self.loss_stat
    
    def save(self, epoch_name):
        epoch_name = str(epoch_name)
        Gpath = os.path.join(self.opt.checkpoint_dir, self.opt.name, 'G_' + epoch_name + '.pth')
        Dpath = os.path.join(self.opt.checkpoint_dir, self.opt.name, 'D_' + epoch_name + '.pth')
        torch.save(self.g.state_dict(), Gpath)
        torch.save(self.d.state_dict(), Dpath)

    def load(self, Gpath, Dpath):

        # Gpath = os.path.join(self.opt.checkpoint_dir, self.opt.name, 'G_' + epoch_name)
        # Dpath = os.path.join(self.opt.checkpoint_dir, self.opt.name, 'D_' + epoch_name)
        # print(Gpath)
        self.g.load_state_dict(torch.load(Gpath))
        self.d.load_state_dict(torch.load(Dpath))
