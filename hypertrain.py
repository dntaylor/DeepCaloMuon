import os
import sys
import argparse
import glob
import json
import pickle
from random import shuffle
import uuid
import datetime

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('convertDir', type=str, 
                    help='Directory of input numpy arrays')
parser.add_argument('trainDir', type=str,
                    help='Output directory')
parser.add_argument('numX', type=int,
                    help='The number of X arrays')


args = parser.parse_args()

# TODO, continue training?
inDir = args.convertDir
outDir = args.trainDir
optimize = True
trials_file = '{}/trials'.format(outDir)
if os.path.exists(outDir):
    print(outDir,'already exists')
    if optimize and os.path.exists(trials_file):
        print('continuing optimization')
    else:
        print('exiting')
        sys.exit(0)


# now import heavier stuff
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Concatenate, LSTM, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint, CSVLogger
from keras.utils import Sequence

from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import backend as k

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
import joblib

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from utilities import python_mkdir

python_mkdir(outDir)
truth_classes = ['pion','muon']

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
k.tensorflow_backend.set_session(tf.Session(config=config))

if optimize:
    max_evals = 10

    try:
        trials = joblib.load(trials_file)
        evals_loaded_trials = len(trials.statuses())
        max_evals += evals_loaded_trials
    except FileNotFoundError:
        trials = Trials()

# load all at once

nx = args.numX
def load_data():
    fnames = {truth: sorted([f for f in glob.glob('{}/output_{}*.x0.npy'.format(inDir,truth)) if 'validation' not in f]) for truth in truth_classes}
    X = {}
    Y = {}
    W = {}
    for truth in truth_classes:
        Xs = [[np.load(fname.replace('.x0.npy','.x{}.npy'.format(i))) for i in range(nx)] for fname in fnames[truth]]
        Ys = [np.load(fname.replace('.x0.npy','.y.npy')) for fname in fnames[truth]]
        Ws = [np.load(fname.replace('.x0.npy','.w.npy')) for fname in fnames[truth]]

        Ws = [np.reshape(w,(w.shape[0],1)) for w in Ws]

        X[truth] = [np.vstack([Xs[j][i] for j in range(len(Xs))]) for i in range(nx)]
        Y[truth] = np.vstack(Ys) 
        W[truth] = np.vstack(Ws) 

        n = W[truth].shape[0]

        # try dropping pions rather than weighting them
        rdrop = np.random.rand(n)
        keep = W[truth].reshape(n)>rdrop
        X[truth] = [X[truth][j][keep] for j in range(len(X[truth]))]
        Y[truth] = Y[truth][keep]
        W[truth] = W[truth][keep]
        nn = W[truth].shape[0]
        W[truth][W[truth].reshape(nn)<1] = 1

    class_counts = [Y[truth].shape[0] for truth in truth_classes]
    min_c = min(class_counts)

    #class_weights = [c/sum(class_counts) for c in class_counts]
    #for i,truth in enumerate(truth_classes):
    #    W[truth] = W[truth] * class_weights[i]
    
    X = {truth: [X[truth][i][:min_c] for i in range(nx)] for truth in truth_classes}
    Y = {truth: Y[truth][:min_c] for truth in truth_classes}
    W = {truth: W[truth][:min_c] for truth in truth_classes}

    #for truth in truth_classes:
    #    print(X[truth].shape)
    #    print(Y[truth].shape)
    #    print(W[truth].shape)
    X = [np.vstack([X[truth][i] for truth in truth_classes]) for i in range(nx)]
    Y = np.vstack([Y[truth] for truth in truth_classes])
    W = np.vstack([W[truth] for truth in truth_classes])
    W = np.reshape(W,(W.shape[0],))

    res = train_test_split(
        *X + [Y, W],
        shuffle = True,
        test_size = 0.1,
        random_state = 123456,
    )
    X_train = [res[2*i] for i in range(nx)]
    X_test  = [res[2*i+1] for i in range(nx)]
    Y_train = res[2*nx]
    Y_test  = res[2*nx+1]
    W_train = res[2*nx+2]
    W_test  = res[2*nx+3]

    return X_train, X_test, Y_train, Y_test, W_train, W_test




#############
### Model ###
#############

def build_model(input_shapes, num_classes, hyperspace):
    doLSTM = hyperspace.get('doLSTM',False)
    lstmWidth = int(hyperspace.get('lstmWidth',128))
    depth = int(hyperspace.get('depth',4))
    width = int(hyperspace.get('width',128))
    batchnorm = hyperspace.get('batchnorm',True)
    momentum = hyperspace.get('momentum',0.6)
    dropoutRate = hyperspace.get('dropoutRate',0.2)
    lr = hyperspace.get('lr',1e-4)
    pattern = []
    kernel = []
    if len(kernel) != len(pattern): kernel = [1]*len(pattern)

    inputs = [Input(shape=s) for s in input_shapes]

    concat = [inputs[0]]

    for i in range(1,len(input_shapes)):
        x = inputs[i]
        for j,p in enumerate(pattern):
            x  = Convolution1D(p, kernel[j], kernel_initializer='lecun_uniform',  activation='relu', name='{}_conv{}'.format(i,j))(x)
            if j<len(pattern)-1:
                if batchnorm:
                    x = BatchNormalization(momentum=momentum ,name='{}_conv_batchnorm{}'.format(i,j))(x)
                x = Dropout(dropoutRate,name='{}_conv_dropout{}'.format(i,j))(x)

        # LSTM
        if doLSTM:
            x = LSTM(lstmWidth,go_backwards=True,implementation=2, name='{}_lstm'.format(1))(x)
            if batchnorm:
                x = BatchNormalization(momentum=momentum,name='{}_lstm_batchnorm'.format(i))(x)
            x = Dropout(dropoutRate,name='{}_lstm_dropout'.format(i))(x)
        # flatten
        else:
            x = Flatten()(x)
        concat += [x]



    if len(concat)>1:
        layer = Concatenate()(concat)
    else:
        layer = concat[0]

    for i in range(depth):
        layer = Dense(width, activation='relu', kernel_initializer='lecun_uniform', name='dense{}'.format(i))(layer)
        if batchnorm:
            layer = BatchNormalization(momentum=momentum, name='dense_batchnorm{}'.format(i))(layer)
        layer = Dropout(dropoutRate, name='dense_dropout{}'.format(i))(layer)

    prediction = Dense(num_classes, activation='softmax', kernel_initializer='lecun_uniform', name='ID_pred')(layer)

    outputs = [prediction]

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = Adam(lr=lr)
    model.compile(
        optimizer=optimizer,
        loss=['categorical_crossentropy'],
        metrics=['accuracy'],
    )
    return model


def train_model(model, X_train, X_test, Y_train, Y_test, W_train, W_test, hyperspace):

    # save the trials before fit (to save the previous)
    joblib.dump(trials, trials_file, compress=('gzip', 3))

    model_uuid = str(uuid.uuid4())
    model_time = datetime.datetime.now()
    model_name = 'model_{}_{}'.format(model_time.strftime('%Y%m%d-%H%M%S'),model_uuid)

    best_name = '{}/KERAS_check_best_{}.h5'.format(outDir,model_name)

    callbacks = [
        ModelCheckpoint(best_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False),
        EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min'),
    ]
    history = model.fit(X_train, Y_train,
                        batch_size = 20000,
                        epochs = 200,
                        verbose = 0,
                        validation_split = 0.1,
                        shuffle = True,
                        sample_weight = W_train,
                        callbacks = callbacks,
                        )

    score = model.evaluate(X_test,Y_test,verbose=0)

    result = {
        'loss': score[0],
        'acc': score[1],
        'space': hyperspace,
        'history': history.history,
        'status': STATUS_OK,
        'model_name': model_name,
        'model_uuid': model_uuid,
    }
    result_path = '{}/result_{}.json'.format(outDir,model_name)
    with open(result_path,'w') as f:
        json.dump(result,f)

    return result



def prepare_optimize_model():
    X_train, X_test, Y_train, Y_test, W_train, W_test = load_data()
    input_shapes = [X_test[i].shape[1:] for i in range(nx)]
    num_classes = Y_test.shape[1]

    build_model_hyperopt = lambda hyperspace: build_model(input_shapes,num_classes,hyperspace)

    train_model_hyperopt = lambda hyperspace: train_model(build_model_hyperopt(hyperspace),X_train,X_test,Y_train,Y_test,W_train,W_test,hyperspace)

    return train_model_hyperopt


callbacks = [
    ModelCheckpoint('{}/KERAS_check_best_model.h5'.format(outDir), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False),
    EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='min'),
    CSVLogger('{}/training.csv'.format(outDir)),
]

modelArgs = {
    'doLSTM': False,
    'lstmWidth': 128,
    'depth': 4,
    'width': 128,
    'pattern': [],
    'kernel': [],
    'batchnorm': True,
    'momentum': 0.6, # 0.6-0.85 for large batches (5k+), larger (0.9-0.99) for smaller batches
    'dropoutRate': 0.2,
    'lr': 1e-4,
}

hyperspace = {
    'doLSTM': hp.choice('doLSTM',[
        (True, hp.quniform('lstmWidth',32,256,1)),
        (False, 0)
    ]),
    'depth': hp.quniform('depth',1,8,1),
    'width': hp.quniform('width',32,256,1),
    'batchnorm': hp.choice('batchnorm',[
        #(True, hp.loguniform('momentum',-0.6,-0.01)),
        (True, hp.choice('momentum',[0.6])),
        (False, 0)
    ]),
    #'dropoutRate': hp.uniform('dropoutRate',0.0,0.5),
    'dropoutRate': hp.choice('dropoutRate',[0.2]),
    'lr': hp.loguniform('lr',-12,-5),
}

if optimize:
    optimize_model = prepare_optimize_model()

    best = fmin(
        optimize_model,
        hyperspace,
        algo = tpe.suggest,
        max_evals = max_evals,
        trials=trials,
    )

    print(best)

    joblib.dump(trials, trials_file, compress=('gzip', 3))

    
    
else:
    X_train, X_test, Y_train, Y_test, W_train, W_test = load_data()
    print([xt.shape for xt in X_train])

    model = build_model([X_test[i].shape[1:] for i in range(nx)],Y_test.shape[1],**modelArgs)
    model.summary()
    
    history = model.fit(X_train, Y_train,
                        batch_size = 20000, 
                        epochs = 1000, 
                        verbose = 1,
                        validation_split = 0.1,
                        shuffle = True,
                        sample_weight = W_train,
                        callbacks = callbacks,
                        )

    

    hname = '{}/history.json'.format(outDir)
    with open(hname,'w') as f:
        json.dump(history.history,f)
    
    # plot loss and accurancy
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    
    epoch_count = range(1, len(loss) + 1)
    
    plt.figure()
    plt.plot(epoch_count, loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('{}/loss.png'.format(outDir))
    
    plt.figure()
    plt.plot(epoch_count, acc, 'r--')
    plt.plot(epoch_count, val_acc, 'b-')
    plt.legend(['Train', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('{}/accuracy.png'.format(outDir))
