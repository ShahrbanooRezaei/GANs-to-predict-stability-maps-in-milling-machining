from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from Dataloader import MultiEpochsDataLoader
import scipy.io




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='SLD', help='SLD')
parser.add_argument('--dataroot',default = './data', help='path to dataset')
parser.add_argument('--Train_dataset_root', default='/data/SLD/GAN_Map_normal_Setup_1/SLDMap.mat', help='path to dataset LL')
parser.add_argument('--n', type=int, default=4000, help='training size')
parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize1', type=int, default=200, help='the height / width of the input image to network')
parser.add_argument('--imageSize2', type=int, default=100, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=700, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda',default=1, action='store_true', help='enables cuda')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netE', default='', help="path to netE (to continue training)")
parser.add_argument('--outf', default='./DCGAN_SLD_out_normal_Setup_1_dit2', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
####################################################################
root = '/SLD/GAN_Map_normal_Setup_1_Test/TrueSLD_0.csv'
    
data = np.zeros((64, 1, opt.imageSize1, opt.imageSize2))

df = pd.read_csv(root, delimiter=',', header=None)

SLD =  np.array(df)
x_tru = []
y_tru = []

for i in range(200):
    b = True
    t = 0
    while (t < 100) & (b == True):
        
        if SLD[i,t] ==1:
            b = False
            x_tru.append(i)
            y_tru.append(t)
        t = t+1
plt.plot(x_tru,y_tru,label = 'True', color = 'red')
plt.show()

df = np.array(df)*255
data[:,0,:,:] = (df-127.5)/127.5 #.T
# data[0,0,:,:] = np.loadtxt(root)

Label = np.zeros((data.shape[0],1))
tensor_x_tru = torch.Tensor(data) # transform to torch tensor
t_data = tensor_x_tru.to(device)


# #########################################

  
data = torch.zeros([opt.n, 1, opt.imageSize1, opt.imageSize2], dtype=torch.float32) #np.zeros((opt.n, 1, opt.imageSize1, opt.imageSize2))
root = opt.Train_dataset_root
# root = opt.Train_dataset_root1
# root = opt.Train_dataset_root2
# root = opt.Train_dataset_root4

train_data = scipy.io.loadmat(root)
    
maps = train_data['maps']
sp = train_data['sampledVals']

# tp = 0
# tt = 0
# while (tp < 10000) & (tt < 4000):
#     if sp[0,tp][0][0,4] > 0.0009:
        
#         data[tt,0,:,:] = torch.tensor(maps[:,:,tp]).to(torch.float32)
#         tt = tt+1
#     tp = tp+1


fs0 = 0
tp = 0
tt = 0
while (tp < 10000) & (tt < 4000):
    if sp[0,tp][0][0,4] > 0.0009:
        if tt%20 == 0:
            fs0 = fs0 +1
            data[tt,0,:,:] = torch.tensor(maps[:,:,2569]).to(torch.float32)
        else:
            data[tt,0,:,:] = torch.tensor(maps[:,:,tp]).to(torch.float32)
        
        
        tt = tt+1
    tp = tp+1

# for i in range(opt.n):
    
#     data[i,0,:,:] = torch.tensor(maps[:,:,i]).to(torch.float32)
 

## Normalize the images to [-1, 1]
data = data*255 #(data - 0.5)/0.5 #!!!!!!!!!!!!!
data = (data - 127.5)/127.5

Label =  torch.zeros([data.shape[0],1],dtype=torch.float32)



# tensor_x = torch.Tensor(data) # transform to torch tensor
# tensor_y = torch.Tensor(Label)

dataset = TensorDataset(data,Label) # create your datset


dataloader = MultiEpochsDataLoader(dataset=dataset,
                                             batch_size=opt.batchSize,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=int(opt.workers),
                                             worker_init_fn=(None if opt.manualseed == -1
                                             else lambda x: np.random.seed(opt.manualseed)))


nc = 1
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(nz, ngf*64),
                        nn.ConvTranspose2d(nz, ngf*64, (6,3), 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(ngf*64),
                        nn.BatchNorm2d(ngf*64))
        main.add_module('initial-{0}-relu'.format(ngf*64),
                        nn.ReLU(True))
        
        
        main.add_module('pyramid-{0}-{1}-convt'.format(ngf*64, ngf*32),
                        nn.ConvTranspose2d(ngf*64, ngf*32 ,(2,2), (1,1), (0,0), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(ngf*32),
                        nn.BatchNorm2d(ngf*32))
        main.add_module('pyramid-{0}-relu'.format(ngf*32),
                            nn.ReLU(True))
        
        main.add_module('pyramid-{0}-{1}-convt'.format(ngf*32, ngf*16),
                        nn.ConvTranspose2d(ngf*32, ngf*16 ,(2,2), (1,1), (0,0), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(ngf*16),
                        nn.BatchNorm2d(ngf*16))
        main.add_module('pyramid-{0}-relu'.format(ngf*16),
                            nn.ReLU(True))
        
        
        main.add_module('pyramid-{0}-{1}-convt'.format(ngf*16, ngf*8),
                        nn.ConvTranspose2d(ngf*16, ngf*8 ,(2,2), (2,2), (1,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(ngf*8),
                        nn.BatchNorm2d(ngf*8))
        main.add_module('pyramid-{0}-relu'.format(ngf*8),
                            nn.ReLU(True))
        
        
        main.add_module('pyramid-{0}-{1}-convt'.format(ngf*8, ngf*4),
                        nn.ConvTranspose2d(ngf*8, ngf*4 ,(2,2), (2,2), (1,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(ngf*4),
                        nn.BatchNorm2d(ngf*4))
        main.add_module('pyramid-{0}-relu'.format(ngf*4),
                            nn.ReLU(True))
        
        main.add_module('pyramid-{0}-{1}-convt'.format(ngf*4, ngf*2),
                        nn.ConvTranspose2d(ngf*4, ngf*2 ,(3,2), (2,2), (1,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(ngf*2),
                        nn.BatchNorm2d(ngf*2))
        main.add_module('pyramid-{0}-relu'.format(ngf*2),
                            nn.ReLU(True))
        
        main.add_module('pyramid-{0}-{1}-convt'.format(ngf*2, ngf),
                        nn.ConvTranspose2d(ngf*2, ngf ,(3,3), (2,2), (1,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format(ngf),
                        nn.BatchNorm2d(ngf))
        main.add_module('pyramid-{0}-relu'.format(ngf),
                            nn.ReLU(True))
        

        main.add_module('final-{0}-{1}-convt'.format(ngf, nc),
                        nn.ConvTranspose2d(ngf, nc, (2,2), (2,2), (1,1), bias=False))
        main.add_module('final-{0}-tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

class Encoder(nn.Module):
    def __init__(self, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        main = nn.Sequential()
        # input is nc x isize1 x isize2 (1 x 200 x 100)
        main.add_module('initial-conv-{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, (2,2), (2,2), (1,1), bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        
        # state size. (ndf) x 101 x 51
        
        main.add_module('pyramid-{0}-{1}-conv'.format(ndf, ndf*2),
                        nn.Conv2d(ndf, ndf*2, (3,3), (2,2), (1,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format( ndf*2),
                        nn.BatchNorm2d( ndf*2))
        main.add_module('pyramid-{0}-relu'.format( ndf*2),
                        nn.LeakyReLU(0.2, inplace=True))
        
        # state size. (ndf) x 51 x 26
        
        main.add_module('pyramid-{0}-{1}-conv'.format(ndf*2, ndf*4),
                        nn.Conv2d(ndf*2, ndf*4, (3,2), (2,2), (1,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format( ndf*4),
                        nn.BatchNorm2d( ndf*4))
        main.add_module('pyramid-{0}-relu'.format( ndf*4),
                        nn.LeakyReLU(0.2, inplace=True))
        
        # state size. (ndf) x 26 x 14
        
        main.add_module('pyramid-{0}-{1}-conv'.format(ndf*4, ndf*8),
                        nn.Conv2d(ndf*4, ndf*8, (2,2), (2,2), (1,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format( ndf*8),
                        nn.BatchNorm2d( ndf*8))
        main.add_module('pyramid-{0}-relu'.format( ndf*8),
                        nn.LeakyReLU(0.2, inplace=True))
        
        
        # state size. (ndf) x 14 x 8
        
        main.add_module('pyramid-{0}-{1}-conv'.format(ndf*8, ndf*16),
                        nn.Conv2d(ndf*8, ndf*16, (2,2), (2,2), (1,1), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format( ndf*16),
                        nn.BatchNorm2d( ndf*16))
        main.add_module('pyramid-{0}-relu'.format( ndf*16),
                        nn.LeakyReLU(0.2, inplace=True))
        
        # state size. (ndf) x 8 x 5
        
        main.add_module('pyramid-{0}-{1}-conv'.format(ndf*16, ndf*32),
                        nn.Conv2d(ndf*16, ndf*32, (2,2), (1,1), (0,0), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format( ndf*32),
                        nn.BatchNorm2d( ndf*32))
        main.add_module('pyramid-{0}-relu'.format( ndf*32),
                        nn.LeakyReLU(0.2, inplace=True))
        
        # state size. (ndf) x 7 x 4
        
        main.add_module('pyramid-{0}-{1}-conv'.format(ndf*32, ndf*64),
                        nn.Conv2d(ndf*32, ndf*64, (2,2), (1,1), (0,0), bias=False))
        main.add_module('pyramid-{0}-batchnorm'.format( ndf*64),
                        nn.BatchNorm2d( ndf*64))
        main.add_module('pyramid-{0}-relu'.format( ndf*64),
                        nn.LeakyReLU(0.2, inplace=True))
        
        # state size. (ndf*8) x 6 x 3
        
        
        main.add_module('final-{0}-{1}-conv'.format(ndf*64, 1),
                            nn.Conv2d(ndf*64, nz, (6,3), 1, 0, bias=False))
        
        self.main = main

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

netE = Encoder(ngpu).to(device)
netE.apply(weights_init)
if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))
print(netE)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 200 x 100
            nn.Conv2d(nc, ndf, (100,50), (14,4), (1,1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            
            # state size. (ndf*8) x 8 x 14
            
            nn.Conv2d(ndf, 1, (8,14), 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)




criterion = nn.BCELoss()
L1loss = nn.L1Loss()

fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=0.002, betas=(opt.beta1, 0.5))
optimizerG = optim.Adam(netG.parameters(), lr=0.002, betas=(opt.beta1, 0.5))

if opt.dry_run:
    opt.niter = 1
def train():
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            for p in netD.parameters():
                p.requires_grad = True  # to avoid computation
                
            for p in netG.parameters():
                p.requires_grad = True  # to avoid computation
                
            for d_iter in range(2):
                netD.zero_grad()
                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), real_label,
                                   dtype=real_cpu.dtype, device=device)
                # label = label + 0.05*torch.rand(batch_size).to(device)
                output = netD(real_cpu)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()
        
                # train with fake
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                output = netD(fake.detach())
                errD_fake = criterion(output, label)
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD.step()
            # if errD.item() < 1e-2: 
            #     netD.apply(weights_init)
            #     print('   Reloading net d')
    
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            for p in netD.parameters():
                    p.requires_grad = False
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            ############################
            # (3) Update E network: Minimize E(G(z))-z
            ###########################
            for p in netG.parameters():
                    p.requires_grad = False
            netE.zero_grad()
            # noise2 = torch.randn(batch_size, nz, 1, 1, device=device)
            # fake2 = netG(noise2)
            # latent = netE(fake2)
            # fake3 = netG(latent)
            
            # errE1 = L1loss(fake2, fake3)
            # errE1.backward()
            # # errE2 = L1loss(noise2, latent)
            # # errE2.backward()
            # errE = errE1 #+ errE2
            
            # latent = netE(real_cpu)
            # fake2 = netG(latent)
            # errE = L1loss(fake2, real_cpu)
            
            latent = netE(fake.detach())
            errE = L1loss(latent, noise)
            errE.backward()
            E_G_z = latent.mean().item()
            optimizerE.step()
            
            #########
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_E: %.4f D(x): %.4f D(G(z)): %.4f / %.4f E(G(z)): %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), errG.item(), errE.item(), D_x, D_G_z1, D_G_z2, E_G_z))
            # with torch.no_grad():
            if i % 500 == 0:
                vutils.save_image(1-real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                # noise = torch.randn(batch_size, nz,1,1, device=device)
                fake = netG(fixed_noise)
                vutils.save_image(1-fake.detach(),
                        '%s/fake_samples_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                
                
                
                
                data2 = np.zeros((100,200))
                
                fake_tru= ((fake *127.5)+127.5)/255
                for i in range(100):
                    for j in range(200):
                        data2[99-i,199-j] = fake_tru[0,0,j,i]
                
                data3 = np.zeros((100,200))
                for i in range(100):
                    for j in range(200):
                        data3[i,199-j] = data2[i,j]
                data3 = (1-data3)*255
                tensor_tru = torch.Tensor(data3) # transform to torch tensor
                fake_truu = tensor_tru.to(device)
                vutils.save_image(fake_truu.detach(),
                        '%s/A_fake_E_G_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                
                
                latent = netE(t_data.detach())
                fake_tru = netG(latent)
                ll = torch.mean(torch.abs(fake_tru-t_data))
                print('recons_loss:', ll)
                
                data2 = np.zeros((100,200))
                
                fake_tru= ((fake_tru *127.5)+127.5)/255
                for i in range(100):
                    for j in range(200):
                        data2[99-i,199-j] = fake_tru[0,0,j,i]
                
                data3 = np.zeros((100,200))
                for i in range(100):
                    for j in range(200):
                        data3[i,199-j] = data2[i,j]
                data3 = (1-data3)*255
                tensor_tru = torch.Tensor(data3) # transform to torch tensor
                fake_truu = tensor_tru.to(device)
                vutils.save_image(fake_truu.detach(),
                        '%s/A_test_E_G_%03d.png' % (opt.outf, epoch),
                        normalize=True)
                
            if opt.dry_run:
                break
        # do checkpointing
        # do checkpointing
        # torch.save({'state_dict': netG.state_dict()},
        #                '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        # torch.save({'state_dict': netE.state_dict()},
        #                '%s/netE_epoch_%d.pth' % (opt.outf, epoch))
        # torch.save({'state_dict': netD.state_dict()},
        #                '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        
        if epoch>= 50:
        
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
            torch.save(netE.state_dict(), '%s/netE_epoch_%d.pth' % (opt.outf, epoch))


plt.close()
if __name__ == '__main__':
    train()

#256, 265
