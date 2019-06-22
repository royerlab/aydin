import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
#import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.pitl.features.multiscale_convolutions import MultiscaleConvolutionalFeatures
from matplotlib import pyplot as plt
import pandas
import collections


def fc_bn(x, unit=1, act='relu', lyrname = None):
    x = Dense(unit, name = lyrname + 'fc')(x)
    x = BatchNormalization(name = lyrname + 'bn')(x)
    return Activation(act, name = lyrname + 'act')(x)

def conv1d_bn(x, unit, k_size, act='relu', lyrname = None):
    x = Conv1D(unit, k_size, padding='same', name = lyrname + 'fc')(x)
    x = BatchNormalization(name = lyrname + 'bn')(x)
    return Activation(act, name = lyrname + 'act')(x)

def extracthist(history, score):
    # Save training history
    hist = history.history
    numepo = history.epoch[-1] + 1
    hist.update({'epoch':history.epoch, 'test_loss': score})
    hist = collections.OrderedDict(hist)
    hist.move_to_end('epoch', last=False)
    return hist, numepo

# load images
data = np.load('sample_data/ChineseTrain.npy')
true_data = data[:,0,...]
noisy_data = data[:,1,...]
noisy_data0 = noisy_data[0:300]
# feature generation
msf = MultiscaleConvolutionalFeatures(exclude_center=True,
                                          kernel_widths=[3, 3],
                                          kernel_scales=[1, 3],
                                          )

features = []
for i in range(noisy_data0.shape[0]):
    features.append(msf.compute(noisy_data[i]))
features = np.stack(features)
features = features.reshape(-1,1,9)

test_features = []
for i in range(10):
    test_features.append(msf.compute(noisy_data[1000+i]))
test_features = np.stack(test_features)
test_features = test_features.reshape(-1,1,9)

#features = np.reshape(msf.compute(noisy_data0), (-1, 1, 9))

#test_features = np.reshape(msf.compute(noisy_data[1]), (-1, 1, 9))
target_val = noisy_data0.reshape(-1,1,1)
#test_target = noisy_data[-1].reshape(-1,1)

test_name = 'conv0fc4'
savepath = os.path.join('output_data/Chinese_test', test_name)
if not os.path.exists(savepath):
    os.makedirs(savepath)


feature_dim = features.shape[-1]
n_estimators=128
max_epoch = 10
batch_size = 128

#data = np.tile(np.arange(64), (20, 1)).reshape((20, 1, 64))
input_feature = Input(shape=(1, feature_dim), name='input')
#x = Conv1D(n_estimators, 1, padding='same', trainable = False, 
#           name='randomize_lyr')(input_feature)
#x = conv1d_bn(x, n_estimators, 16, lyrname='cv1')
#x = conv1d_bn(x, n_estimators, 16, lyrname='cv2')
#x = conv1d_bn(x, n_estimators, 16, lyrname='cv3')
x = fc_bn(input_feature, unit=128, lyrname = 'fc1')
x = fc_bn(x, unit=128*2, lyrname = 'fc2')
x = fc_bn(x, unit=128*2, lyrname = 'fc3')
x = fc_bn(x, act='linear', lyrname='fc_last')
model = Model(input_feature, x)
model.compile(optimizer = 'Adam', loss='mse')

EStop = EarlyStopping(monitor='val_loss', min_delta=0,
                      patience=6, verbose=1, mode='auto')
ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                             verbose=1, patience=3, mode='auto', min_lr=1e-8)
Chkpnt1 = ModelCheckpoint(savepath + '/weights.{epoch:02d}-vdl{val_loss:.4f}.h5',
                         monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True)

history = model.fit(features, target_val,  validation_split=0.25,
                    epochs=400, batch_size=batch_size,
                    callbacks=[EStop, ReduceLR, Chkpnt1])

score = model.evaluate(test_features, true_data[1000:1010].reshape(-1,1,1), 
                       batch_size=batch_size, verbose=1)
hist, numepo = extracthist(history, score)

# plot training history
fig, ax1 = plt.subplots(2, sharex=True)
#fig, ax1 = plt.subplots()
ax1[0].set_xlabel('Epoch')
ax1[0] = plt.subplot2grid((1,2),(0,0))
l1 = ax1[0].plot(hist['epoch'], hist['loss'], label='Training loss')
l2 = ax1[0].plot(hist['epoch'], hist['val_loss'],label='Validation loss')
l3 = ax1[0].plot(hist['epoch'], hist['test_loss'] * np.ones(numepo),label='Testing loss')
ax1[0].set_ylabel('Loss')
ax1[0].set_yscale("log")
ax2=ax1[0].twinx()
ax2.set_ylabel('Learn rate')
l22 = ax2.plot(hist['epoch'], hist['lr'], '.', label='Learn rate')
ax2.set_yscale("log")
ax1[0].set_title('Decoder learning history   ('+ test_name + ')', loc='left')
ax1[0].set_xlabel('Epoch')
# combining legends
lns = l1+l2+l3+l22
labs = [l.get_label() for l in lns]
#ax1[0].legend(lns, labs, loc=0)
ax1[0].legend(lns, labs, loc=2, borderaxespad=0, bbox_to_anchor=(1.3, 1))
plt.subplots_adjust(left=0.14, right=1, top=0.94, bottom=0.1, hspace=0.29, wspace=0)
plt.show()
plt.savefig(os.path.join(savepath, 'history.png'))

# output test images
output = model.predict(test_features)

plt.figure(figsize=(4,10))
plt.subplot(131)
plt.imshow(noisy_data[1000:1010].reshape(320,32), cmap='gray')
plt.title('Noisy')
plt.axis('off')
plt.subplot(132)
plt.imshow(output.reshape(320,32), cmap='gray')
plt.title('Denoised')
plt.axis('off')
plt.subplot(133)
plt.imshow(true_data[1000:1010].reshape(320,32), cmap='gray')
plt.title('Clean')
plt.axis('off')
plt.savefig(os.path.join(savepath, 'trainedimg.png'), dpi=300)