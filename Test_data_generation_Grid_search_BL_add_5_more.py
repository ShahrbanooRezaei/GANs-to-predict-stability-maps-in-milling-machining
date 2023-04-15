
#### Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import scipy.ndimage as ndimage

' Data Augmentation and Bayesian learning for 10 test cuts'

####
sample_num = 1

data_augment = 'yes' 
'Add additional 5 test cuts to the previous 5 cuts selected by the grid search ' 
tpb =  [3, 3, 1.5, 3, 1.5]
tps =  [11105.527638190957, 10500, 12613.065326633165, 8241.206030150754, 13500]
st =  [1, 0, 1, 1, 0] #0 stable, 1 unstable


#### Select range of spindle speed and axial depth of cut for Setup 1

tool = 'normal_Setup_1'
tr = '/target_1/'
s_model_min = 2000
s_model_max = 20000
b_model_min = 0
b_model_max = 10
sig_nbT = 50
sig_b = 0.1

#%% fixed parameters
n = 1
tp = 5
tpt = 5+5
tpt_p = tp

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
True_sld = np.array(pd.read_csv('data/GAN_Map_'+tool+'_Test/'+tr+'TrueSLD_0.csv', header = None))
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
#%%
new_test_points= np.zeros((tpt-tp, 3))
for i in range(new_test_points.shape[0]):
    x = tps[i]
    difference_array = np.absolute(s-x)
    index = difference_array.argmin()
    new_test_points[i,1] = s[index]
    
    x = tpb[i]
    difference_array = np.absolute(b-x)
    index = difference_array.argmin()
    new_test_points[i,0] = b[index]
    
    new_test_points[i,2] = st[i]

if n != 0:
    rot = 'data/GAN_Map_'+tool+'_Test/'+tr+str(tp)+'_test_points/S'+str(sample_num)+'/'+str(n) #???????????
    if not os.path.isdir(rot):
        os.makedirs(rot)
    np.savetxt(rot+'/'+'new_test_points.txt', new_test_points)
else:
    rot = 'data/GAN_Map_'+tool+'_Test/'+tr+str(tp)+'_test_points/S'+str(sample_num)
# %%%
################################################## Test ##################################################
# Select random test points <<< structured search

## 

test_before_aug = np.array(pd.read_csv('data/GAN_Map_'+tool+'_Test/'+tr+str(tp)+'_test_points/S'+str(sample_num)+'/test_before_aug_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=None))
b_test_before_aug = test_before_aug[:,0]
s_test_before_aug = test_before_aug[:,1]
test_pred = test_before_aug[:,2]

test_final = np.zeros((tpt, 3))
test_final[:tpt_p,:] = test_before_aug
test_final[tpt_p:,:] = new_test_points

df = pd.DataFrame(test_final)
df.to_csv(rot+'/test_before_aug_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=False,index = False)

b_test_before_aug = test_final[:,0]
s_test_before_aug = test_final[:,1]

s_test = test_final[:,1]
b_test = test_final[:,0]
test_pred = test_final[:,2]

data = np.zeros((1,s_n))
for i in range(1):
    for j in range(s_n):
        if np.sum(np.where(True_sld[j,:]==1)[0]) == 0:
            ind = -1
        else:
            ind = np.where(True_sld[j,:]==1)[0][0]
        data[i,j] = b[ind]



bsim = data[0, :] 
        
cticks = np.linspace(0, 1, 11, endpoint=True)   
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot()  
# cs = ax1.contourf(XX.T,YY.T,1-z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
ax1.plot(s, bsim, color = 'red')
plt.ylabel('b [mm]',fontsize = 14)
plt.xlabel('n [rpm]',fontsize = 14)
ax1.set_xlim(s[0],s[-1])
ax1.set_ylim(b[0],b[-1])
ax1.tick_params(labelsize=14)
ax1.set_xticks(np.arange(s_model_min,s_model_max+0.01,2000))

for i in range(tpt):
    if test_pred[i] == 0: 
        ax1.plot(s_test[i], b_test[i], 'bo')
    else:
        ax1.plot(s_test[i], b_test[i], 'rx')

plt.savefig(rot+'/test_points.png')
plt.show()  
Y_test = test_pred  

#%%###################################################################################################

'Add synthetic data'
#### Add additional stable and unstbale points at smaller and larger axial depths of cut

if data_augment == 'yes':
    
    for cnt in range(tpt):
    
        if Y_test[cnt] == 0:
    #        
            b_in = np.where(b < b_test[cnt])[0]
            b_test = np.append(b_test, b[b_in])
            Y_test = np.append(Y_test, np.zeros(shape = (len(b_in),1)))
            s_test = np.append(s_test, s_test[cnt]*np.ones(shape = (len(b_in),1)))
        
        if Y_test[cnt] == 1:
            
            b_in = np.where(b > b_test[cnt])[0]
            b_test = np.append(b_test, b[b_in])
            Y_test = np.append(Y_test, np.ones(shape = (len(b_in),1)))
            s_test = np.append(s_test, s_test[cnt]*np.ones(shape = (len(b_in),1)))
    
    
    #Add stable results at 0 axial depth of cut to prevent boundary from going below 0
    s_test = np.append(s_test, s)
    b_test = np.append(b_test, np.zeros(shape = (len(s))))
    Y_test = np.append(Y_test, np.zeros(shape = (len(s))))
    

test_final_aft_aug = np.zeros((s_test.shape[0],3))
test_final_aft_aug [:,0] = b_test
test_final_aft_aug [:,1] = s_test
test_final_aft_aug [:,2] = Y_test

df = pd.DataFrame(test_final_aft_aug)
df.to_csv(rot+'/test_aft_aug_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=False,index = False)

s_test = s_test.reshape(-1,1)
b_test = b_test.reshape(-1,1)

bsim = data[0, :] 
        
cticks = np.linspace(0, 1, 11, endpoint=True)   
fig = plt.figure(figsize=(7, 5))
ax1 = fig.add_subplot()  
# cs = ax1.contourf(XX.T,YY.T,1-z, cmap = cm.gray_r, levels = cticks, extend = 'min', alpha = 0.5)
ax1.plot(s, bsim, color = 'red')
plt.ylabel('b [mm]',fontsize = 14)
plt.xlabel('n [rpm]',fontsize = 14)
ax1.set_xlim(s[0],s[-1])
ax1.set_ylim(b[0],b[-1])
ax1.tick_params(labelsize=14)
ax1.set_xticks(np.arange(s_model_min,s_model_max+0.01,2000))

for i in range(s_test.shape[0]):
    if Y_test[i] == 0: 
        ax1.plot(s_test[i], b_test[i], 'bo')
    else:
        ax1.plot(s_test[i], b_test[i], 'rx')
plt.savefig(rot+'/test_points_aug.png')
plt.show() 

## Test points index in x and y dirextion
#s_test_before_aug
b_test_index = []
s_test_index = []

for i in range(len(b_test_before_aug)):
    b_test_index.append(np.where(b_test_before_aug[i] == b)[0][0])
    s_test_index.append(np.where(np.round(s_test_before_aug[i],4) == np.round(s,4))[0][0])

test_points = []

for i in range(tp):
    test_points.append([b_test_index[i], s_test_index[i]])

test_points = np.array(test_points) #[65,79] #21

Y_test_GAN = 1-Y_test
state = Y_test_GAN

trueSLD = 1-True_sld

#%%########################## Bayesian learning and Input for GAN #####################################

Ib = 100
IN = 200

b = b
N = s

# Prior stability
prior2 = np.linspace(0, 1, Ib) #np.zeros((100,200))
prior = [prior2[:] for item in range(IN)]
prior = np.array(prior).T

n = -1
prior2 = np.zeros((Ib,IN))
for i in range(Ib):
    prior2[i,:] = prior[n,:]
    n = n-1

prior =   prior2  


state = Y_test_GAN
n_t = test_points.shape[0]


# Likelihood probablities
#stable test
def p_sT_giv_sG(Nj,Nt,sig):
    p = 0.5+np.exp(-0.5*((Nj-Nt)/sig)**2)/2
    return p


def p_sT_giv_uG(Nj,Nt,sig,bi,bt):
    p = 0.5+np.exp(-0.5*((Nj-Nt)/sig)**2)/(1/(p_sT_giv_uG_NT(bi,bt)-0.5))
    return p

def p_sT_giv_uG_NT(bi,bt):
    if bi<= bt+3*sig_b:
        p = np.exp(-0.5*((bi-(bt+3*sig_b))/sig_b)**2)
    else:
        p = 1
    return p


# unstable test
def p_sT_giv_sG_n(Nj,Nt,sig):
    p = 0.5-np.exp(-0.5*((Nj-Nt)/sig)**2)/2
    return p

def p_sT_giv_uG_n(Nj,Nt,sig,bi,bt):
    p = 0.5-np.exp(-0.5*((Nj-Nt)/sig)**2)/(1/-(p_sT_giv_uG_NT_n(bi,bt)+0.5))
    return p

def p_sT_giv_uG_NT_n(bi,bt):
    if bi>= bt-3*sig_b:
        p = np.exp(-0.5*(((bt-3*sig_b)-bi)/sig_b)**2)
    else:
        p = 1
    return p

# Posterior probabilities update

for k in range(n_t):
    tb = test_points[k][0]
    tn = test_points[k][1]
    
    if state[k] == 1:
        prior[:tb,tn] = 1 #white stable
        for i in range(prior.shape[0]):
            if b[i] <= b[tb]+3*sig_b:
                sig_Nbi = (-sig_nbT/(3*sig_b))*b[i] + (sig_nbT*(b[tb]+3*sig_b))/(3*sig_b)
            else:
                sig_Nbi = 0
            if b[i] <= b[tb]:
                for j in range(prior.shape[1]):
                    if np.abs(N[tn]-N[j]) < 3*sig_Nbi:
                        likelihood = p_sT_giv_sG(N[j],N[tn],sig_Nbi)
                        prior[i,j] = (likelihood*prior[i,j])/(likelihood*prior[i,j]+(1-likelihood)*(1-prior[i,j]))
            else:
                for j in range(prior.shape[1]):
                    if N [j] == N[tn]:
                        likelihood = 1
                        likelihood_uG = p_sT_giv_uG_NT(b[i],b[tb])
                        prior[i,j] = (likelihood*prior[i,j])/(likelihood*prior[i,j]+likelihood_uG*(1-prior[i,j]))
                    else: 
                        if np.abs(N[tn]-N[j]) < 3*sig_Nbi:
                            likelihood = p_sT_giv_sG(N[j],N[tn],sig_Nbi)
                            likelihood_uG = p_sT_giv_uG(N[j],N[tn],sig_Nbi, b[i],b[tb])
                            
                            prior[i,j] = (likelihood*prior[i,j])/(likelihood*prior[i,j]+likelihood_uG*(1-prior[i,j]))
    else:
        prior[tb:,tn] = 0
        
        for i in range(prior.shape[0]):
            if b[i] >= b[tb]-3*sig_b:
                sig_Nbi = (sig_nbT/(3*sig_b))*b[i] - (sig_nbT*(b[tb]-3*sig_b))/(3*sig_b)
            else:
                sig_Nbi = 0
            if b[i] >= b[tb]:
                for j in range(prior.shape[1]):
                    if np.abs(N[tn]-N[j]) < 3*sig_Nbi:
                        likelihood = p_sT_giv_sG_n(N[j],N[tn],sig_Nbi)
                        prior[i,j] = (likelihood*prior[i,j])/(likelihood*prior[i,j]+(1-likelihood)*(1-prior[i,j]))
            else:
                for j in range(prior.shape[1]):
                    if N [j] == N[tn]:
                        likelihood = p_sT_giv_uG_NT_n(b[i],b[tb])
                        likelihood_uG = 1
                        prior[i,j] = (likelihood*prior[i,j])/(likelihood*prior[i,j]+likelihood_uG*(1-prior[i,j]))
                    else:
                        
                        if np.abs(N[tn]-N[j]) < 3*sig_Nbi:
                            likelihood = p_sT_giv_sG_n(N[j],N[tn],sig_Nbi)
                            likelihood_uG = p_sT_giv_uG_n(N[j],N[tn],sig_Nbi, b[i],b[tb])
                            
                            prior[i,j] = (likelihood*prior[i,j])/(likelihood*prior[i,j]+likelihood_uG*(1-prior[i,j]))
   
        


dataa = np.zeros((1, 1, 200, 100))
prior = 1-prior.T

### Save 
df = prior
df = pd.DataFrame(prior)

df.to_csv(rot+'/GAN_Input_Bayes_'+str(tool)+'_'+str(tp)+'_samples_'+'sample_'+str(sample_num)+'.csv', header=False,index = False)

###Plot
m=[]
for k in range(n_t):
    tb = test_points[k][0]
    tn = test_points[k][1]
    m.append([round(N[tn],4),round(b[tb],4),state[k],0])



chatterProbabilities = 1-prior
    
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

ax.plot(x_G_t, 100-np.array(y_G_t), color = 'red')
plt.savefig(rot+'/GAN_input_Bayes_w_sld.png')
plt.show()

##### without sld
fig, ax = plt.subplots()
angle = 90 # in degrees
new_data = ndimage.rotate(chatterProbabilities, angle, reshape=True)
im = ax.imshow(new_data,cmap='gray')
bar = plt.colorbar(im)
bar.set_label('Expected probability of stability')


ax.set_xlabel('n [rpm]')
ax.set_xticks(x)
ax.set_xticklabels(xlab)

ax.set_ylabel('b [mm]')
ax.set_yticks(y)
ax.set_yticklabels(ylab)
plt.savefig(rot+'/GAN_input_Bayes.png')
plt.show()


###############################3 New figure
Z = chatterProbabilities
b = aps

s_test = test_final_aft_aug[:tpt,1].reshape(-1,1)
b_test = test_final_aft_aug[:tpt,0].reshape(-1,1)
Y_test = test_final_aft_aug[:tpt,2].reshape(-1,1)


Y_train = Y_test

X_plot = np.concatenate((s_test, b_test), axis = 1)
X_train_plot, Y_train_plot = X_plot, Y

#### Plot

X0_train = X_train_plot[np.where(Y_train== 0)[0],:]
X1_train = X_train_plot[np.where(Y_train == 1)[0],:]


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
# ax1.plot(s_model, b_model, 'g')
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
plt.savefig(rot+'/GAN_input_Bayes_new.png')
plt.show()
