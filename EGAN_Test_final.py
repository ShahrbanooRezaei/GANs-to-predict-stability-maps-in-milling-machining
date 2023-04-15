from __future__ import print_function
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
# import pandas as pd
from numpy import genfromtxt

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
from matplotlib import cm

#%% Test on data > Numerical experiments

tp = 5

for n in [1]:
    for sample_num in [1]:
        
        #### Select range of spindle speed and axial depth of cut for Setup 1
        tool = 'Setup 1'
        tr = '/target_1/'
        method = "_dit2/"
        s_model_min = 2000
        s_model_max = 20000
        b_model_min = 0
        b_model_max = 10
        N1 = 100 #Chosee trained N1th to N2th trained encoder and decoderto make prediction
        N2 = 200
        
        
        
        if n !=0:
            rot = '/data/SLD/GAN_Map_'+tool+'_Test/'+tr+str(tp)+'_test_points/S'+str(sample_num)+'/'+str(n)
            if not os.path.isdir(rot):
                os.makedirs(rot)
            
        else:
            
            rot = '/data/SLD/GAN_Map_'+tool+'_Test/'+tr+str(tp)+'_test_points/S'+str(sample_num)
        
        #%%
        s_n = 200
        b_n = 100
        aps_lim = b_model_max
        
        
        #### Input parameters
        rpms = np.linspace(s_model_min, s_model_max, s_n) 
        aps = np.linspace(b_model_min, b_model_max, b_n)
        
        #### import True SLD
        True_sld = np.array(pd.read_csv('data/SLD/GAN_Map_'+tool+'_Test'+tr+'TrueSLD_0.csv', header = None))
        trueSLD = 1-True_sld
        
        #%% Model
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='SLD', help='SLD')
        parser.add_argument('--dataroot',default = './data', help='path to dataset')
        parser.add_argument('--n', type=int, default=10000, help='training size')
        parser.add_argument('--manualseed', default=-1, type=int, help='manual seed')
        parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
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
        parser.add_argument('--outf', default='./DCGAN_SLD_out_Tool_D_sheet1_uniform', help='folder to output images and model checkpoints')
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
        
        
        
        
        #%%# Bayesian input to GAN
        
        root = rot+'/GAN_Input_Bayes_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv' #!!!!!!!!!!!!!
        df = genfromtxt(root, delimiter=',')
        
        fig, ax = plt.subplots()
        
        angle = 90 # in degrees
        new_data = ndimage.rotate(1-df, angle, reshape=True)
        im = ax.imshow(new_data,cmap='gray')
        #bar = plt.colorbar(im)
        #bar.set_label('Expected probability of stability')
        
        ## True Stability Map
        x_G_t = []
        y_G_t = []
        
        for i in range(200):
            bb = 0
            t = 0
            
            while (t < 100) & (bb == 0):
                if (t == 99) & (bb == 0):
                    x_G_t.append(i)
                    y_G_t.append(t)
                    
                if trueSLD[i,t] == 0:
                    bb = 1
                    x_G_t.append(i)
                    y_G_t.append(t)
                t = t+1
        
        ax.plot(x_G_t, 100-np.array(y_G_t), color = 'red')
        plt.show()
        
        
        df = np.array(df)*255
        dataa = np.zeros((1, 1, 200, 100))
        dataa[0,0,:,:] = ((df-127.5)/127.5)
        
        tensor_xa = torch.Tensor(dataa)
        t_dataa = tensor_xa.to(device)
        
        ##### Output of the last 400 trained GAN
        
        z_all = []
        it = 0
        SLD_GAN_list = 0
        for i in range(N1,N2): #!!!!!!!!! DCGAN_SLD_out_sheet4_uniform_Tool_E
            it = it+1
            print(it)
            path_G = "/DCGAN_SLD_out_"+tool+method+"netG_epoch_"+str(i)+".pth" #!!!!!!!!!!
            path_E = "/DCGAN_SLD_out_"+tool+method+"netE_epoch_"+str(i)+".pth" #!!!!!!!!!!
            netG.load_state_dict(torch.load(path_G))
            netE.load_state_dict(torch.load(path_E))
            
        
            latent = netE(t_dataa.detach())
            fake = netG(latent)
            
            
            
            
            fake= ((fake *127.5)+127.5)/255
            SLD_GAN_list = SLD_GAN_list + fake[0,0,:,:].detach().cpu()
            
            z_all.append(1-fake[0,0,:,:].detach().cpu())
        
        #%%
        a_all = np.zeros((N2-N1,200))
        for l in range(N2-N1): #!!!!!!!!!!! 
            sld_TR = z_all[l]
            
            # TR
            SLD =  sld_TR < 0.5
            x_TR = []
            y_TR = []
            
            for i in range(200):
                bt = True
                t = 0
                for t in range(100):
                    
                    if (SLD[i,t] == True) & (bt==True):
                        bt = False
                        x_TR.append(i)
                        y_TR.append(t)
                    if (t == 99) & (bt==True):
                        x_TR.append(i)
                        y_TR.append(t)
                    
                    
            # fig, ax = plt.subplots()
            # ax.plot(x_TR,y_TR, label = 'TR', color = 'green')
            
            a_all[l,:] = y_TR
        
        
        a_TR_tru = []
        
        for i in range(200):
            bb = 0
            t = 0
            while (t < 100) & (bb == 0):
                
                if trueSLD[i,t] == 0:
                    bb = 1
                    
                    a_TR_tru.append(aps[t])
                t = t+1
            if (t == 100) & (bb == 0):
                a_TR_tru.append(aps[-1])
        
        # truncated of labels in each spindle speed
        a_all2 = np.zeros((N2-N1,200))
        for i in range(N2-N1):
            for j in range(200): 
                a_all2[i,j] = aps[int(a_all[i,j])]
            
        a_all_sort = np.sort(a_all2, axis=0)
        
        k_exc = 0.25
        k_exc = int(0.25*(N2-N1))
        a_TR = np.sum(a_all_sort[k_exc:-k_exc,:], axis=0)/((N2-N1)-2*k_exc)
        fig, ax = plt.subplots()
        ax.plot(rpms, a_TR, label = 'TR', color = 'green')
        ax.plot(rpms, a_TR_tru, color = 'red')
        
        
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        plt.savefig(rot+'/GAN_bound_output.png')
        plt.show()
        
        sld_GAN_ave = 1-SLD_GAN_list/it
        
        fig, ax = plt.subplots()
        
        angle = 90 # in degrees
        new_data = ndimage.rotate(sld_GAN_ave, angle, reshape=True)
        im = ax.imshow(new_data,cmap='gray')
        #bar = plt.colorbar(im)
        #bar.set_label('Expected probability of stability')
        
        ## True Stability Map
        x_G_t = []
        y_G_t = []
        
        for i in range(200):
            bb = 0
            t = 0
            
            while (t < 100) & (bb == 0):
                if (t == 99) & (bb == 0):
                    x_G_t.append(i)
                    y_G_t.append(t)
                    
                if trueSLD[i,t] == 0:
                    bb = 1
                    x_G_t.append(i)
                    y_G_t.append(t)
                t = t+1
        
        ax.plot(x_G_t, 100-np.array(y_G_t), color = 'red')
        
        ## Fig labels
        x = []
        y = []
        xlab = []
        ylab = []
        for i in range(rpms.shape[0]):
            if i in [0, 39, 79, 119, 159, 199]:
                x.append(i)
                xlab.append(str(int(round(rpms[i]))))
        for i in range(aps.shape[0]):
            if i in [0, 19, 39, 59, 79, 99]:
                y.append(i)
                ylab.append(str(int(aps_lim-round(aps[i]))))
        
        ax.set_xlabel('n [rpm]')
        ax.set_xticks(x)
        ax.set_xticklabels(xlab)
        
        ax.set_ylabel('b [mm]')
        ax.set_yticks(y)
        ax.set_yticklabels(ylab)
        
        plt.savefig(rot+'/GAN_prob_output_input_Bayes.png')
        plt.show()
        
        #### Save
        df = pd.DataFrame(np.array((1-sld_GAN_ave)))
        df.to_csv(rot+'/sld_EGAN_ave_Bayes_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'_vae_inside.csv', header=False,index = False)
        
        sld_GAN_ave_Bayes = 1-sld_GAN_ave
        
        
        ################################################################################### new figure for GAN
        Z = 1-sld_GAN_ave_Bayes
        
        
        b = aps
        s = rpms
        XX, YY = np.meshgrid(b,s)
        
        test_final_aft_aug = np.array(pd.read_csv(rot+'/test_aft_aug_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=None))
        
        s_test = test_final_aft_aug[:,1].reshape(-1,1)
        b_test = test_final_aft_aug[:,0].reshape(-1,1)
        Y_test = test_final_aft_aug[:,2].reshape(-1,1)
        
        Y = Y_test
        
        
        X_plot = np.concatenate((s_test, b_test), axis = 1)
        X_train_plot, Y_train_plot = X_plot, Y
        
        
        
        X0_train = X_train_plot[np.where(Y== 0)[0],:]
        X1_train = X_train_plot[np.where(Y == 1)[0],:]
        
        b_model = []
        for j in range(s_n):
            if np.sum(np.where(True_sld[j,:]==1)[0]) == 0:
                ind = -1
            else:
                ind = np.where(True_sld[j,:]==1)[0][0]
            b_model.append(b[ind])
        b_model = np.array(b_model)
        s_model = s
        
        
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        #plt.scatter(X0[:,0], X0[:,1], color = 'blue', marker = 'o', label = 'stable')
        #plt.scatter(X1[:,0], X1[:,1], color = 'red', marker = 'x', label = 'unstable')
        plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # plt.scatter(X0_test[:,0], X0_test[:,1], color = 'blue', marker = 'o')
        # plt.scatter(X1_test[:,0], X1_test[:,1], color = 'red', marker = 'x')
        # plt.plot(ss_sim, bsim)
        cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        ax1.plot(s_model, b_model, 'r')
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        #ax1.legend(loc = 'best',fontsize = 'large', prop={'size': 8})
        ax1.set_xlim(s[0],s[-1])
        ax1.set_ylim(b[0],b[-1])
        ax1.tick_params(labelsize=14)
        # ax1.set_xticks(np.arange(s_model_min,s_model_max,1000))
        cbar = fig.colorbar(cs, ticks = cticks)
        plt.legend(loc = 'best', fontsize = 'large')
        cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot+'/GAN_prob_output_input_Bayes_w_points.png')
        plt.show()
        
        ######
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        #plt.scatter(X0[:,0], X0[:,1], color = 'blue', marker = 'o', label = 'stable')
        #plt.scatter(X1[:,0], X1[:,1], color = 'red', marker = 'x', label = 'unstable')
        plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # plt.scatter(X0_test[:,0], X0_test[:,1], color = 'blue', marker = 'o')
        # plt.scatter(X1_test[:,0], X1_test[:,1], color = 'red', marker = 'x')
        # plt.plot(ss_sim, bsim)
        cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        ax1.plot(s_model, b_model, 'r')
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        #ax1.legend(loc = 'best',fontsize = 'large', prop={'size': 8})
        ax1.set_xlim(s[0],s[-1])
        ax1.set_ylim(b[0],b[-1])
        ax1.tick_params(labelsize=14)
        # ax1.set_xticks(np.arange(s_model_min,s_model_max,1000))
        #cbar = fig.colorbar(cs, ticks = cticks)
        plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot+'/GAN_prob_output_input_Bayes_w_points_2.png')
        plt.show()
        
        ######
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        #plt.scatter(X0[:,0], X0[:,1], color = 'blue', marker = 'o', label = 'stable')
        #plt.scatter(X1[:,0], X1[:,1], color = 'red', marker = 'x', label = 'unstable')
        # plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        # plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # plt.scatter(X0_test[:,0], X0_test[:,1], color = 'blue', marker = 'o')
        # plt.scatter(X1_test[:,0], X1_test[:,1], color = 'red', marker = 'x')
        # plt.plot(ss_sim, bsim)
        cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        ax1.plot(s_model, b_model, 'r')
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        #ax1.legend(loc = 'best',fontsize = 'large', prop={'size': 8})
        ax1.set_xlim(s[0],s[-1])
        ax1.set_ylim(b[0],b[-1])
        ax1.tick_params(labelsize=14)
        # ax1.set_xticks(np.arange(s_model_min,s_model_max,1000))
        #cbar = fig.colorbar(cs, ticks = cticks)
        plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot+'/GAN_prob_output_input_Bayes_without_points_2.png')
        plt.show()
        
        ######
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        #plt.scatter(X0[:,0], X0[:,1], color = 'blue', marker = 'o', label = 'stable')
        #plt.scatter(X1[:,0], X1[:,1], color = 'red', marker = 'x', label = 'unstable')
        # plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        # plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # plt.scatter(X0_test[:,0], X0_test[:,1], color = 'blue', marker = 'o')
        # plt.scatter(X1_test[:,0], X1_test[:,1], color = 'red', marker = 'x')
        # plt.plot(ss_sim, bsim)
        cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        #ax1.plot(s_model, b_model, 'r')
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        #ax1.legend(loc = 'best',fontsize = 'large', prop={'size': 8})
        ax1.set_xlim(s[0],s[-1])
        ax1.set_ylim(b[0],b[-1])
        ax1.tick_params(labelsize=14)
        # ax1.set_xticks(np.arange(s_model_min,s_model_max,1000))
        #cbar = fig.colorbar(cs, ticks = cticks)
        plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot+'/GAN_prob_output_input_Bayes_without_points_sld.png')
        plt.show()
        
        
        
        
        #%%
        ####################################################################### new figure for BL
        root = rot+'/GAN_Input_Bayes_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv' #!!!!!!!!!!!!!
        df = genfromtxt(root, delimiter=',')
        
        Z = 1-df
        
        
        b = aps
        s = rpms
        XX, YY = np.meshgrid(b,s)
        
        test_final_aft_aug = np.array(pd.read_csv(rot+'/test_aft_aug_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=None))
        
        s_test = test_final_aft_aug[:,1].reshape(-1,1)
        b_test = test_final_aft_aug[:,0].reshape(-1,1)
        Y_test = test_final_aft_aug[:,2].reshape(-1,1)
        
        Y = Y_test
        
        
        X_plot = np.concatenate((s_test, b_test), axis = 1)
        X_train_plot, Y_train_plot = X_plot, Y
        
        
        
        X0_train = X_train_plot[np.where(Y== 0)[0],:]
        X1_train = X_train_plot[np.where(Y == 1)[0],:]
        
        b_model = []
        for j in range(s_n):
            if np.sum(np.where(True_sld[j,:]==1)[0]) == 0:
                ind = -1
            else:
                ind = np.where(True_sld[j,:]==1)[0][0]
            b_model.append(b[ind])
        b_model = np.array(b_model)
        s_model = s
        
        
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        #plt.scatter(X0[:,0], X0[:,1], color = 'blue', marker = 'o', label = 'stable')
        #plt.scatter(X1[:,0], X1[:,1], color = 'red', marker = 'x', label = 'unstable')
        plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # plt.scatter(X0_test[:,0], X0_test[:,1], color = 'blue', marker = 'o')
        # plt.scatter(X1_test[:,0], X1_test[:,1], color = 'red', marker = 'x')
        # plt.plot(ss_sim, bsim)
        cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        ax1.plot(s_model, b_model, 'r')
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        #ax1.legend(loc = 'best',fontsize = 'large', prop={'size': 8})
        ax1.set_xlim(s[0],s[-1])
        ax1.set_ylim(b[0],b[-1])
        ax1.tick_params(labelsize=14)
        # ax1.set_xticks(np.arange(s_model_min,s_model_max,1000))
        cbar = fig.colorbar(cs, ticks = cticks)
        plt.legend(loc = 'best', fontsize = 'large')
        cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot+'/BL_prob_output_w_points.png')
        plt.show()
        
        ######
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        #plt.scatter(X0[:,0], X0[:,1], color = 'blue', marker = 'o', label = 'stable')
        #plt.scatter(X1[:,0], X1[:,1], color = 'red', marker = 'x', label = 'unstable')
        plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # plt.scatter(X0_test[:,0], X0_test[:,1], color = 'blue', marker = 'o')
        # plt.scatter(X1_test[:,0], X1_test[:,1], color = 'red', marker = 'x')
        # plt.plot(ss_sim, bsim)
        cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        ax1.plot(s_model, b_model, 'r')
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        #ax1.legend(loc = 'best',fontsize = 'large', prop={'size': 8})
        ax1.set_xlim(s[0],s[-1])
        ax1.set_ylim(b[0],b[-1])
        ax1.tick_params(labelsize=14)
        # ax1.set_xticks(np.arange(s_model_min,s_model_max,1000))
        #cbar = fig.colorbar(cs, ticks = cticks)
        plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot+'/BL_prob_output_w_points_2.png')
        plt.show()
        
        ######
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        #plt.scatter(X0[:,0], X0[:,1], color = 'blue', marker = 'o', label = 'stable')
        #plt.scatter(X1[:,0], X1[:,1], color = 'red', marker = 'x', label = 'unstable')
        # plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        # plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # plt.scatter(X0_test[:,0], X0_test[:,1], color = 'blue', marker = 'o')
        # plt.scatter(X1_test[:,0], X1_test[:,1], color = 'red', marker = 'x')
        # plt.plot(ss_sim, bsim)
        cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        ax1.plot(s_model, b_model, 'r')
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        #ax1.legend(loc = 'best',fontsize = 'large', prop={'size': 8})
        ax1.set_xlim(s[0],s[-1])
        ax1.set_ylim(b[0],b[-1])
        ax1.tick_params(labelsize=14)
        # ax1.set_xticks(np.arange(s_model_min,s_model_max,1000))
        #cbar = fig.colorbar(cs, ticks = cticks)
        plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot+'/BL_prob_output_without_points_2.png')
        plt.show()
        
        ######
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        #plt.scatter(X0[:,0], X0[:,1], color = 'blue', marker = 'o', label = 'stable')
        #plt.scatter(X1[:,0], X1[:,1], color = 'red', marker = 'x', label = 'unstable')
        # plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        # plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # plt.scatter(X0_test[:,0], X0_test[:,1], color = 'blue', marker = 'o')
        # plt.scatter(X1_test[:,0], X1_test[:,1], color = 'red', marker = 'x')
        # plt.plot(ss_sim, bsim)
        cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        #ax1.plot(s_model, b_model, 'r')
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        #ax1.legend(loc = 'best',fontsize = 'large', prop={'size': 8})
        ax1.set_xlim(s[0],s[-1])
        ax1.set_ylim(b[0],b[-1])
        ax1.tick_params(labelsize=14)
        # ax1.set_xticks(np.arange(s_model_min,s_model_max,1000))
        #cbar = fig.colorbar(cs, ticks = cticks)
        plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot+'/BL_prob_output_without_points_sld.png')
        plt.show()
        
        
        
        
