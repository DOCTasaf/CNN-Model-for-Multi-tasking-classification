from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten,Input
from keras.models import Sequential,Model
from keras.utils import plot_model,to_categorical
from keras.regularizers import l2
from keras.optimizers import Adam
import pre_data
from keras.callbacks import TensorBoard
import numpy as np
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras import regularizers
from utils import Tsne_plot,plot_train,confusion_plot,save_reprot,save_history,kappa,result_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,\
f1_score,roc_auc_score,roc_curve,cohen_kappa_score
from sklearn.metrics import precision_score,classification_report
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold,train_test_split
from operator import truediv
import warnings
warnings.filterwarnings('ignore')

import time
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from pre_data import pre_data

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def build_model(input_shape,num_classes):

    input_sig = Input(input_shape)
    
    x=Conv1D(16,64,strides=8,padding="same",name='cov1',kernel_regularizer=regularizers.l2(0.01))(input_sig)
    
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling1D(2,name='maxp1')(x)
#    x=Dropout(0.2)(x)
    
    x=Conv1D(32,3,strides=1,padding="same",name='cov2',kernel_regularizer=regularizers.l2(0.01))(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling1D(2,name='maxp2')(x)
    
    # x=Conv1D(64,3,strides=1,padding="same",name='cov3',kernel_regularizer=regularizers.l2(0.01))(x)
    # x=BatchNormalization()(x)
    # x=Activation('relu')(x)
    # x=MaxPooling1D(2)(x)
 
    x=Conv1D(64,3,strides=1,padding="same",name='cov4',kernel_regularizer=regularizers.l2(0.01))(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)
    x=MaxPooling1D(2,name='maxp3')(x)
    x=Flatten()(x)

    x=Dense(64,activation="relu",kernel_regularizer=regularizers.l2(0.01),name='fc1')(x)
    x=Dropout(0.4)(x)

    output=Dense(8,activation="softmax",name='fc2')(x)    
    #output=Dense(6,activation="softmax",name='fc2')(x)

    model = Model(input=input_sig, output=output)

    opt = Adam(lr=1e-3)#, decay=1e-3/10
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['accuracy'])
  
    model.summary()
    return model

if not os.path.exists('Model'):
    os.mkdir('Model')
