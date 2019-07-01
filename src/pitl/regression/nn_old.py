# from __future__ import absolute_import, print_function
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from keras import optimizers

class NNRegressor:
    """
    Regressor that uses the CNN.

    """

#    nnreg: NNRegressor

    def __init__(self,
#                 feature_dim=9,
#                 max_depth=-1,
#                 num_layers,
                 n_estimators=128,
                 kernel_size = 16,
                 learning_rate=0.001,
#                 eval_metric='l1',
                 early_stopping_rounds=5
                 ):
        """
        Constructs a LightGBM regressor.

        :param num_leaves:
        :type num_leaves:
        :param n_estimators:
        :type n_estimators:
        :param learning_rate:
        :type learning_rate:
        :param eval_metric:
        :type eval_metric:
        :param early_stopping_rounds:
        :type early_stopping_rounds:
        """

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

        self.EStop = EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=early_stopping_rounds, verbose=1, mode='auto')
        self.ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                     verbose=1, patience=3, mode='auto', min_lr=1e-8)
        

    def fit(self, x_train, y_train, x_test, y_test):
        """
        Fits function y=f(x) given training pairs (x_train, y_train).
        Stops when performance stops improving on the test dataset: (x_test, y_test).

        :param x_train:
        :type x_train:
        :param y_train:
        :type y_train:
        :param x_test:
        :type x_test:
        :param y_test:
        :type y_test:
        """
        
        def fc_bn(x, unit=1, act='relu', lyrname = None):
            x = Dense(unit, name = lyrname + 'fc')(x)
            x = BatchNormalization(name = lyrname + 'bn')(x)
            return Activation(act, name = lyrname + 'act')(x)
        
        def conv1d_bn(x, unit, k_size, act='relu', lyrname = None):
            x = Conv1D(unit, k_size, padding='same', name = lyrname + 'fc')(x)
            x = BatchNormalization(name = lyrname + 'bn')(x)
            return Activation(act, name = lyrname + 'act')(x)

        feature_dim = x_train.shape[-1]
        self.feature_dim = feature_dim
        input_feature = Input(shape=(1, feature_dim), name='input')
        x = fc_bn(input_feature, unit=self.n_estimators, lyrname = 'fc1')
        x = fc_bn(x, unit=self.n_estimators*2, lyrname = 'fc2')
        x = fc_bn(x, unit=self.n_estimators*2, lyrname = 'fc3')
        x = fc_bn(x, act='linear', lyrname='fc_last')
        model = Model(input_feature, x)
        opt = optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer = opt, loss='mse')
        self.nnreg = model
        
        x_train = x_train.reshape(-1,1,feature_dim)
        y_train = y_train.reshape(-1,1,1)
        x_test = x_test.reshape(-1,1,feature_dim)
        y_test = y_test.reshape(-1,1,1)
        
        self.nnreg.fit(x_train, y_train,
                       validation_data=(x_test, y_test),
                       epochs=100, batch_size=512,
                       callbacks=[self.EStop, self.ReduceLR])

    def predict(self, x):
        """
        Predicts y given x by applying the learned function f: y=f(x)
        :param x:
        :type x:
        :return:
        :rtype:
        """
        x = x.reshape(-1,1,x.shape[-1])
        return self.nnreg.predict(x)
