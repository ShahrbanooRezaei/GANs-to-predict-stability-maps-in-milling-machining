'Fine tunning'
'Transfer learning code for paper titled: '
'Ensemble transfer learning for refining stability predictions in milling using experimental stability states' 

#### Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from pyDOE import lhs
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
import tensorflow as tf
import os
import scipy.ndimage as ndimage
from joblib import Parallel, delayed
import multiprocessing as mp

####  
tp = 5 #Number of test cuts
sample_num = 1 #sample number
n = 1 # 0 if 5 tests selected, 1 if 10 test selected
transfer_learning_Tuning = 'Yes'

ep = 50 # Traning episodes
N = 200 #number of neural networks


#### Select range of spindle speed and axial depth of cut for Setup 1

tool = 'normal_Setup_1'
tr = '/target_1/'
s_model_min = 2000
s_model_max = 20000
b_model_min = 0
b_model_max = 10




#%%
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
    rot = 'data/GAN_Map_'+tool+'_Test'+tr+str(tp)+'_test_points/S'+str(sample_num)+'/'+str(n)
    
else:
    
    rot ='data/GAN_Map_'+tool+'_Test'+tr+str(tp)+'_test_points/S'+str(sample_num)


#%%######################################################## Import the generated tests to fine tune

test_final_aft_aug = np.array(pd.read_csv(rot+'/test_aft_aug_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=None))

s_test = test_final_aft_aug[:,1].reshape(-1,1)
b_test = test_final_aft_aug[:,0].reshape(-1,1)
Y_test = test_final_aft_aug[:,2].reshape(-1,1)


s_test_scale = (s_test - s[0])/(s[-1] - s[0]) #!!!!!!!!!!!!!! scale it based on the training scaler
b_test_scale = (b_test - b[0])/(b[-1] - b[0])

X = np.concatenate((b_test_scale, s_test_scale), axis = 1)
Y = Y_test

X_train, Y_train = X, Y



y_train = tf.keras.utils.to_categorical(Y_train)

X_plot = np.concatenate((s_test, b_test), axis = 1)
X_train_plot, Y_train_plot = X_plot, Y

#### Plot

X0_train = X_train_plot[np.where(Y_train== 0)[0],:]
X1_train = X_train_plot[np.where(Y_train == 1)[0],:]


fig = plt.subplots(figsize = (7,5))   
plt.plot(s_model,b_model, 'g')
plt.ylabel('b [mm]',fontsize = 14)
plt.xlabel('n [rpm]',fontsize = 14)
plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5)
plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5)
plt.xlim(s[0],s[-1])
plt.ylim(b[0],b[-1])
plt.tick_params(labelsize=14)
plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Retrain N neural networks on test points
es_val_loss = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=False)
es_loss = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100, restore_best_weights=False)

def model_NN(aa):
    
    print(aa)
    model = load_model('model/'+tool+'/'+'SLD_'+str(aa)+'/model.h5')
    history = model.fit(X_train, y_train, epochs=ep, verbose=0, shuffle = True)
    
    if n == 0:
        os.makedirs('model/'+tool+'_RE/'+tr+str(tp)+'_test_points/S'+str(sample_num)+'/SLD_'+str(aa))
    model.save('model/'+tool+'_RE/'+tr+str(tp)+'_test_points/S'+str(sample_num)+'/SLD_'+str(aa)+'/model.h5')
    
    labels = model.predict(grid)[:,0]
    
    Z = labels.reshape(xx.shape)
    
    return Z


if transfer_learning_Tuning == 'Yes': 
    
    Z_all = Parallel(n_jobs=-1, backend = 'threading')(delayed(model_NN)(i) for i in range(N))
    
def model_NN_load(aa):
    print(aa)
    model = load_model('model/'+tool+'_RE/'+tr+str(tp)+'_test_points/S'+str(sample_num)+'/SLD_'+str(aa)+'/model.h5') 
    # Make predictions across region of interest
    labels = model.predict(grid)[:,0]
    
    # Plot decision boundary in region of interest
    Z = labels.reshape(xx.shape)
    # Z_all.append(Z)
    
    return Z    

if transfer_learning_Tuning == 'No':
    
    
    Z_all = Parallel(n_jobs=-1, backend = 'threading')(delayed(model_NN_load)(i) for i in range(N))

## find boundary on prediction of each NN
a_all = np.zeros((N,200))
for l in range(N): #!!!!!!!!!!! 
    sld_TR = Z_all[l]
    
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

# truncated of labels in each spindle speed
a_all_sort = np.sort(a_all, axis=0)
k_exc = 0.25
k_exc = int(0.25*N)
a_TR = np.sum(a_all_sort[k_exc:-k_exc,:], axis=0)/(N-2*k_exc)
fig, ax = plt.subplots()
ax.plot(x_TR, a_TR, label = 'TR', color = 'green')
ax.plot(x_G_t, y_G_t, label = 'GT', color = 'red')
plt.ylabel('b [mm]',fontsize = 14)
plt.xlabel('n [rpm]',fontsize = 14)
plt.savefig(rot+'/ETL_bound_output.png')
plt.show()



##### ETL Prob prediction
prob_pred = np.sum(np.array(Z_all), axis = 0)/N
df = pd.DataFrame(prob_pred)
df.to_csv(rot+'/ETL_prob_pred_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=False,index = False)


chatterProbabilities = prob_pred
fig, ax = plt.subplots()
angle = 90 # in degrees
new_data = ndimage.rotate(chatterProbabilities, angle, reshape=True)
im = ax.imshow(new_data,cmap='gray')


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

plt.savefig(rot+'/ETL_prob_output.png')
plt.show()



###################################################################################
'Plot results'
###################################################################################

Z = prob_pred


cticks = np.linspace(0, 1, 11, endpoint=True)  
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot()  

plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')

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
plt.savefig(rot+'/ETL_prob_output_W_points.png')
plt.show()

################

cticks = np.linspace(0, 1, 11, endpoint=True)  
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot()  

plt.scatter(X0_train[:,0], X0_train[:,1], color = 'blue', marker = 'o', alpha = 0.5,  label = 'stable')
plt.scatter(X1_train[:,0], X1_train[:,1], color = 'red', marker = 'x', alpha = 0.5,  label = 'unstable')

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
plt.savefig(rot+'/ETL_prob_output_W_points_2.png')
plt.show()

################

cticks = np.linspace(0, 1, 11, endpoint=True)  
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot()  

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
plt.savefig(rot+'/ETL_prob_output_Without_points_2.png')
plt.show()

############
cticks = np.linspace(0, 1, 11, endpoint=True)  
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot()  

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
plt.savefig(rot+'/ETL_prob_output_Without_points_sld.png')
plt.show()


