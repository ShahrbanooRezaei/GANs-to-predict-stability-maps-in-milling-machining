import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import os

from numpy import genfromtxt
from scipy import stats
import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return [m, h]
  
tp = 5
transfer_learning_Tuning = 'Yes'



#### Select range of spindle speed and axial depth of cut 
tool = 'Setup 1'
tr = '/target_1/'
s_model_min = 2000
s_model_max = 20000
b_model_min = 0
b_model_max = 10
sss = 'S1-tg1'




for n in [0,1]: 
    print('nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
    res_mean = []
    res_mean_w2 = []
    for sample_num in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        #%%
        nm = 200
        s_n = 200
        b_n = 100
        aps_lim = b_model_max
        
        
        #### Input parameters
        rpms = np.linspace(s_model_min, s_model_max, s_n) 
        aps = np.linspace(b_model_min, b_model_max, b_n) 
        
        #### matrix of 20,000 points
        b = aps
        s = rpms
        XX, YY = np.meshgrid(b,s)
        grid_pred = np.c_[XX.ravel(), YY.ravel()]
        
        #### scale the 20,000 points
        b_scale = (b - b[0])/(b[-1]-b[0])
        s_scale = (s - s[0])/(s[-1]-s[0])
        xx, yy = np.meshgrid(b_scale, s_scale)
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        
        
        #### import True SLD
        True_sld = np.array(pd.read_csv('data/GAN_Map_'+tool+'_Test'+tr+'TrueSLD_0.csv', header = None))
        Y = True_sld.reshape((len(b)*len(s),1))
        
        
        # Find the boundary
        b_model = []
        for j in range(s_n):
            if np.sum(np.where(True_sld[j,:]==1)[0]) == 0:
                ind = -1
            else:
                ind = np.where(True_sld[j,:]==1)[0][0]
            b_model.append(b[ind])
        b_model = np.array(b_model)
        s_model = s
        
        
        trueSLD = 1-True_sld
        
        
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
        
        
        #%%
        if n !=0:
            rot_TLG = 'data/GAN_Map_'+tool+'_Test'+tr+str(tp)+'_test_points/S'+str(sample_num)+'/'+str(n)
            
        else:
            
            rot_TLG ='data/GAN_Map_'+tool+'_Test'+tr+str(tp)+'_test_points/S'+str(sample_num)
        
        
        
        
        if n !=0:
            ind_a = 10
            rot_final_fig = 'data_final/GAN_Map_'+tool+'_Test/'+sss+'/figs/'+str(sample_num)+'_'+str(n)
            if os.path.isdir(rot_final_fig) == False: 
                os.makedirs(rot_final_fig)
        else:
            ind_a = 5
            rot_final_fig = 'data_final/GAN_Map_'+tool+'_Test/'+sss+'/figs/'+str(sample_num)+'_'+str(n)
            if os.path.isdir(rot_final_fig) == False: 
                os.makedirs(rot_final_fig)
        
        if n !=0:
            ind_a = 10
            rot_final_f = 'data_final/GAN_Map_'+tool+'_Test/'+sss+'/res/'+str(n)
            if os.path.isdir(rot_final_f) == False: 
                os.makedirs(rot_final_f)
        else:
            ind_a = 5
            rot_final_f ='data_final/GAN_Map_'+tool+'_Test/'+sss+'/res/'+str(n)
            if os.path.isdir(rot_final_f) == False: 
                os.makedirs(rot_final_f)
        
        
        #%%#########
        
        
        test_before_aug = np.array(pd.read_csv(rot_TLG+'/test_before_aug_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=None))
        b_test_before_aug = test_before_aug[:,0]
        s_test_before_aug = test_before_aug[:,1]
        test_pred = test_before_aug[:,2]
        
        s_test = test_before_aug[:,1].reshape(-1,1)
        b_test = test_before_aug[:,0].reshape(-1,1)
        Y_test = test_before_aug[:,2].reshape(-1,1)
        
        Y = Y_test
        
        
        X_plot = np.concatenate((s_test, b_test), axis = 1)
        X_train_plot, Y_train_plot = X_plot[:,:], Y
        
        
        
        X0_train = X_train_plot[np.where(Y== 0)[0],:]
        X1_train = X_train_plot[np.where(Y == 1)[0],:]
        
        
        
        xn = []
        xb = []
        a = (s_model_max-s_model_min)/ind_a
        ab = (b_model_max-b_model_min)/ind_a
        for i in range(10):
            xn.append(s_model_min+a*(i+1))
        for i in range(10):
            xb.append(b_model_min+ab*(i+1))
        
        
        cticks = np.linspace(0, 1, 11, endpoint=True)  
        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_subplot()  
        
        plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
        plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')
        # cs = ax1.contourf(YY,XX,1-Z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
        aa = 'g:'
        for i in range(10):
            ax1.plot([xn[i],xn[i]], [b_model_min,b_model_max],   aa)
        
        for i in range(10):
            ax1.plot([s_model_min,s_model_max], [xb[i],xb[i]],   aa)
        
        
        
        # ax1.plot(s_model, b_model, 'r')
        
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
        
        
        plt.savefig(rot_final_fig+'_test_points.png')
        plt.show()
        
        
        #%%
        ####################### BL predic
        root = rot_TLG+'/GAN_Input_Bayes_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv' #!!!!!!!!!!!!!
        df = genfromtxt(root, delimiter=',')
        
        
        Z = 1-df
        Z_BL = Z
        
        
        b = aps
        s = rpms
        XX, YY = np.meshgrid(b,s)
        
        
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
        # plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot_final_fig+'_BL_prob_output_with_SLD.png')
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
        # plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        # plt.savefig(rot_final_fig+'_BL_prob_output_without_sld.png')
        plt.show()
        
        #%%
        ####################### TL predic
        root = rot_TLG+'/ETL_prob_pred_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv' #!!!!!!!!!!!!!
        df = genfromtxt(root, delimiter=',')
        
        
        Z = df
        Z_TL = Z
        
        
        b = aps
        s = rpms
        XX, YY = np.meshgrid(b,s)
        
        
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
        # plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot_final_fig+'_TL_prob_output_with_SLD.png')
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
        # plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        # plt.savefig(rot_final_fig+'_TL_prob_output_without_sld.png')
        plt.show()
        
        #%%
        ####################### TL predic
        root = rot_TLG+'/sld_EGAN_ave_Bayes_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'_vae_inside.csv' #!!!!!!!!!!!!!
        df = genfromtxt(root, delimiter=',')
        
        
        Z = 1-df
        Z_GAN = Z
        
        
        b = aps
        s = rpms
        XX, YY = np.meshgrid(b,s)
        
        
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
        # plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        plt.savefig(rot_final_fig+'_GAN_prob_output_with_SLD.png')
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
        # plt.legend(loc = 'best', fontsize = 'large')
        #cbar.ax.tick_params(labelsize=14) 
        # plt.savefig(rot_final_fig+'_GAN_prob_output_without_sld.png')
        plt.show()
        
        
        #%% Prediction boundary
        
        a_all = np.zeros((4,nm))
        SLD_label = []
        l = 0
        for z in [Z_BL[:nm,:], Z_TL[:nm,:], Z_GAN[:nm,:], trueSLD[:nm,:]]: #!!!!!!!!!!! 
            sld_TR = z
            
            
            # TR
            SLD =  sld_TR < 0.5 #0.5
            SLD_label.append(SLD)
            x_TR = []
            y_TR = []
            
            for i in range(nm):
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
            l = l+1
        
        # truncated of labels in each spindle speed
        
        fig, ax = plt.subplots()
        ax.plot(rpms[:nm], a_all[0], label = 'BL', color = 'brown')
        ax.plot(rpms[:nm], a_all[1], label = 'ETL', color = 'blue')
        ax.plot(rpms[:nm], a_all[2], label = 'EGAN', color = 'green')
        ax.plot(rpms[:nm], a_all[3], label = 'Target', color = 'red')
        
        
        plt.ylabel('b [mm]',fontsize = 14)
        plt.xlabel('n [rpm]',fontsize = 14)
        plt.legend()
        plt.savefig(rot_final_fig+'_bound_output.png')
        plt.show()
        
        #%% ######### 
        
        from sklearn.metrics import confusion_matrix
        ev_metrics = np.zeros((3,2))
        SLD = [Z_BL[:nm,:], Z_TL[:nm,:], Z_GAN[:nm,:], trueSLD[:nm,:]]
        for i in range(3):
            ev_metrics[i,0] = np.mean(np.abs(SLD[-1]-SLD[i]))
            ev_metrics[i,1] = np.mean((SLD[-1]-SLD[i])**2)
            
        ACC = []
        Method = ['BL', 'TL', 'EGAN']
        CNN = [] 
        
        for i in range(3):
            pred_label = SLD_label[i] 
            pred_label = pred_label.astype(int)
            pred_label = pred_label.reshape(nm*100)
            
            true_label = SLD_label[-1] 
            true_label = true_label.astype(int)
            true_label = true_label.reshape(nm*100)
            
            
            
            # for tao in thres:
            # pred = []
            # for jj in range(score.shape[0]):
            #     if score[jj] <= tao:
            #         pred.append(0)
            #     else:
            #         pred.append(1)
            
            tn, fp, fn, tpp= confusion_matrix(true_label, pred_label).ravel()
            
            acc = 100*((tpp+tn)/(tpp+tn+fp+fn))
            sensetivity = 100*( tpp/(tpp + fn))
            specificity = 100*( tn/(tn + fp))
            F1 = 100*((2*tpp) / (2*tpp + fp + fn))
            ppv = 100*( tpp/(tpp+fp+0.000005))
            Gmean = np.sqrt(sensetivity*specificity)
                    
                
            CNN.append([Method[i],acc, sensetivity, specificity, F1, ppv,Gmean, ev_metrics[i,0], ev_metrics[i,1]])
        res_mean.append(CNN)
        df3=pd.DataFrame(CNN)
        df3.columns=['Method', 'acc', 'sensetivity', 'specificity', 'F1', 'PPV','Gmean', 'L1', 'MSE']
        # df3.to_csv(rot_final_fig+'_Eval_metrics.csv', index=False) 
        
        #%% ######### 
        from sklearn.metrics import confusion_matrix
        ev_metrics = np.zeros((3,2))
        SLD = [Z_BL[:nm,:], Z_TL[:nm,:], Z_GAN[:nm,:], trueSLD[:nm,:]]
        for i in range(3):
            ev_metrics[i,0] = np.mean(np.abs(SLD[-1]-SLD[i]))
            ev_metrics[i,1] = np.mean((SLD[-1]-SLD[i])**2)
            
        ACC = []
        Method = ['BL', 'TL', 'EGAN']
        CNN = [] 
        
        
        for i in range(3):
            pred_label = np.zeros((nm,100))
            for jj in range(nm):
                pred_label[jj,int(a_all[i][jj]):] = 1
                
            
            pred_label = pred_label.reshape(nm*100)
            pred_label = pred_label.reshape(-1,1)
            
            true_label = np.zeros((nm,100))
            for jj in range(nm):
                true_label[jj,int(a_all[-1][jj]):] = 1
            
            true_label = true_label.reshape(nm*100)
            true_label = true_label.reshape(-1,1)
            
            
            
            # for tao in thres:
            # pred = []
            # for jj in range(score.shape[0]):
            #     if score[jj] <= tao:
            #         pred.append(0)
            #     else:
            #         pred.append(1)
            
            tn, fp, fn, tpp= confusion_matrix(true_label, pred_label).ravel()
            
            acc = 100*((tpp+tn)/(tpp+tn+fp+fn))
            sensetivity = 100*( tpp/(tpp + fn))
            specificity = 100*( tn/(tn + fp))
            F1 = 100*((2*tpp) / (2*tpp + fp + fn))
            ppv = 100*( tpp/(tpp+fp+0.000005))
            Gmean = np.sqrt(sensetivity*specificity)
                    
                
            CNN.append([Method[i],acc, sensetivity, specificity, F1, ppv,Gmean, ev_metrics[i,0], ev_metrics[i,1]])
        res_mean_w2.append(CNN)
        df3=pd.DataFrame(CNN)
        df3.columns=['Method', 'acc', 'sensetivity', 'specificity', 'F1', 'PPV','Gmean', 'L1', 'MSE']
        # df3.to_csv(rot_final_fig+'_w2_Eval_metrics.csv', index=False)
    
    #%%
    #BL
    n_res = np.array(res_mean)[:,0,1:]
    n_res = n_res.astype('float')
    m_ci = np.zeros((6, n_res.shape[1]))
    for lp in range(n_res.shape[1]):
        m_ci[0,lp] = mean_confidence_interval(n_res[:,lp])[0]
        m_ci[1,lp] = mean_confidence_interval(n_res[:,lp])[1]
    
    # df3=pd.DataFrame(m_ci)
    # df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'L1', 'MSE']
    # df3.to_csv(rot_final_f+str(n)+'_BL_Eval_metrics_mc.csv', index=False)
    
    df3=pd.DataFrame(n_res)
    df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV','Gmean', 'L1', 'MSE']
    df3.to_csv(rot_final_f+str(n)+'_BL_Eval_metrics.csv', index=False)
    
    #TL
    n_res = np.array(res_mean)[:,1,1:]
    n_res = n_res.astype('float')
    # m_ci = np.zeros((2, n_res.shape[1]))
    for lp in range(n_res.shape[1]):
        m_ci[2,lp] = mean_confidence_interval(n_res[:,lp])[0]
        m_ci[3,lp] = mean_confidence_interval(n_res[:,lp])[1]
    
    # df3=pd.DataFrame(m_ci)
    # df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'L1', 'MSE']
    # df3.to_csv(rot_final_f+str(n)+'_TL_Eval_metrics_mc.csv', index=False)
    
    df3=pd.DataFrame(n_res)
    df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV','Gmean', 'L1', 'MSE']
    df3.to_csv(rot_final_f+str(n)+'_TL_Eval_metrics.csv', index=False)
    
    stat_tl = n_res
    
    #EGAN
    n_res = np.array(res_mean)[:,2,1:]
    n_res = n_res.astype('float')
    # m_ci = np.zeros((2, n_res.shape[1]))
    for lp in range(n_res.shape[1]):
        m_ci[4,lp] = mean_confidence_interval(n_res[:,lp])[0]
        m_ci[5,lp] = mean_confidence_interval(n_res[:,lp])[1]
    
    df3=pd.DataFrame(m_ci)
    df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'Gmean', 'L1', 'MSE']
    df3.to_csv(rot_final_f+str(n)+'_Eval_metrics_mc.csv', index=False)
    
    df3=pd.DataFrame(n_res)
    df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'Gmean', 'L1', 'MSE']
    df3.to_csv(rot_final_f+str(n)+'_EGAN_Eval_metrics.csv', index=False)
    
    stat_egan = n_res
    print('')
    print('acc',stats.ttest_ind(stat_tl[:,0], stat_egan[:,0]))
    print('sen',stats.ttest_ind(stat_tl[:,1], stat_egan[:,1]))
    print('spc',stats.ttest_ind(stat_tl[:,2], stat_egan[:,2]))
    print('F1',stats.ttest_ind(stat_tl[:,3], stat_egan[:,3]))
    print('PPV',stats.ttest_ind(stat_tl[:,4], stat_egan[:,4]))
    print('GMean',stats.ttest_ind(stat_tl[:,5], stat_egan[:,5]))
    print('L1', stats.ttest_ind(stat_tl[:,-2], stat_egan[:,-2]))
    print('MSE', stats.ttest_ind(stat_tl[:,-1], stat_egan[:,-1]))
    
    
    #%%
    #BL
    n_res = np.array(res_mean_w2)[:,0,1:]
    n_res = n_res.astype('float')
    m_ci = np.zeros((6, n_res.shape[1]))
    for lp in range(n_res.shape[1]):
        m_ci[0,lp] = mean_confidence_interval(n_res[:,lp])[0]
        m_ci[1,lp] = mean_confidence_interval(n_res[:,lp])[1]
    
    # df3=pd.DataFrame(m_ci)
    # df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'L1', 'MSE']
    # df3.to_csv(rot_final_f+str(n)+'_BL_Eval_metrics_mc.csv', index=False)
    
    df3=pd.DataFrame(n_res)
    df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'Gmean', 'L1', 'MSE']
    df3.to_csv(rot_final_f+str(n)+'_w2_BL_Eval_metrics.csv', index=False)
    
    #TL
    n_res = np.array(res_mean_w2)[:,1,1:]
    n_res = n_res.astype('float')
    # m_ci = np.zeros((2, n_res.shape[1]))
    for lp in range(n_res.shape[1]):
        m_ci[2,lp] = mean_confidence_interval(n_res[:,lp])[0]
        m_ci[3,lp] = mean_confidence_interval(n_res[:,lp])[1]
    
    # df3=pd.DataFrame(m_ci)
    # df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'L1', 'MSE']
    # df3.to_csv(rot_final_f+str(n)+'_TL_Eval_metrics_mc.csv', index=False)
    
    df3=pd.DataFrame(n_res)
    df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'Gmean', 'L1', 'MSE']
    df3.to_csv(rot_final_f+str(n)+'_w2_TL_Eval_metrics.csv', index=False)
    
    stat_tl = n_res
    
    #EGAN
    n_res = np.array(res_mean_w2)[:,2,1:]
    n_res = n_res.astype('float')
    # m_ci = np.zeros((2, n_res.shape[1]))
    for lp in range(n_res.shape[1]):
        m_ci[4,lp] = mean_confidence_interval(n_res[:,lp])[0]
        m_ci[5,lp] = mean_confidence_interval(n_res[:,lp])[1]
    
    df3=pd.DataFrame(m_ci)
    df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'Gmean', 'L1', 'MSE']
    df3.to_csv(rot_final_f+str(n)+'_w2_Eval_metrics_mc.csv', index=False)
    
    df3=pd.DataFrame(n_res)
    df3.columns=['acc', 'sensetivity', 'specificity', 'F1', 'PPV', 'Gmean', 'L1', 'MSE']
    df3.to_csv(rot_final_f+str(n)+'_w2_EGAN_Eval_metrics.csv', index=False)
    
    stat_egan = n_res
    print('w2')
    print('acc',stats.ttest_ind(stat_tl[:,0], stat_egan[:,0]))
    print('sen',stats.ttest_ind(stat_tl[:,1], stat_egan[:,1]))
    print('spc',stats.ttest_ind(stat_tl[:,2], stat_egan[:,2]))
    print('F1',stats.ttest_ind(stat_tl[:,3], stat_egan[:,3]))
    print('PPV',stats.ttest_ind(stat_tl[:,4], stat_egan[:,4]))
    print('GMean',stats.ttest_ind(stat_tl[:,5], stat_egan[:,5]))
    print('L1', stats.ttest_ind(stat_tl[:,-2], stat_egan[:,-2]))
    print('MSE', stats.ttest_ind(stat_tl[:,-1], stat_egan[:,-1]))


#%%
