'Pre Training'
'Transfer learning code for paper titled: '
'Ensemble transfer learning for refining stability predictions in milling using experimental stability states' 

#### Import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import tensorflow as tf
import os
import scipy.ndimage as ndimage
from joblib import Parallel, delayed
from keras.models import load_model
import scipy.io

#### Input parameters
ep = 100 #Training episodes
N = 200 # Number of Neural networks 
 
#### Select range of spindle speed and axial depth of cut for Setup
tool = 'normal_Setup_1'
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


b = aps
s = rpms
XX, YY = np.meshgrid(b,s)
grid_pred = np.c_[XX.ravel(), YY.ravel()]

b_scale = (b - b[0])/(b[-1]-b[0])
s_scale = (s - s[0])/(s[-1]-s[0])
xx, yy = np.meshgrid(b_scale, s_scale)
grid = np.c_[xx.ravel(), yy.ravel()]


#### Load analytical SLDs from file 

data_train = np.zeros((N, s_n, b_n))

root = 'data/GAN_Map_'+tool+'/SLDMap.mat'


train_data = scipy.io.loadmat(root)
maps = train_data['maps']


Y_par = []
for i in range(N):
    data_train[i,:,:] = maps[:,:,i]#np.loadtxt(root+str(i+1)+'.txt')
    
    Y_panes = data_train[i,:,:]
    
    Y = Y_panes.reshape((len(b)*len(s),1))
    Y = Y.reshape(-1)
    Y_par.append(Y)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Train N Neural Networks on each SLD
def model_NN(y, aa):
    
    y = tf.keras.utils.to_categorical(y)
    
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(20, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(2, activation='softmax'))

    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    # fit model
    history = model.fit(X, y, epochs=ep, callbacks = [es_loss], verbose=0,  shuffle = True)
    
    os.makedirs('model/'+tool+'/'+'SLD_'+str(aa))
    model.save('model/'+tool+'/'+'SLD_'+str(aa)+'/model.h5')
    
    return model

es_val_loss = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, restore_best_weights=False)
es_loss = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=100, restore_best_weights=False)

X = np.c_[xx.ravel(), yy.ravel()]  

model = Parallel(n_jobs=-1, backend = 'threading')(delayed(model_NN)(Y_par[i], i) for i in range(N))



#### Plot prediction for last SLD
# Make predictions across region of interest
i = 2
model2 = load_model('model/'+tool+'/'+'SLD_'+str(i)+'/model.h5')
pred = model2.predict(grid)
labels = pred[:,0] #
labels1 = pred[:,1] #

# Plot decision boundary in region of interest
z = labels.reshape(xx.shape) 
   
############################ Plot 1 #########################

trueSLD = 1-z < 0.5
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig, ax = plt.subplots()

angle = 90 # in degrees
new_data = ndimage.rotate(z, angle, reshape=True) 
im = ax.imshow(new_data,cmap='gray')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
   

bar = plt.colorbar(im, cax=cax)
bar.set_label('Expected probability of stability')


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

