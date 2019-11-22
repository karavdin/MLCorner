# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.")
# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import numpy as np
from numpy import inf
import keras
import matplotlib
import math
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
#from tensorflow_probability.python.layers.dense_variational import DenseFlipout

import h5py
from tensorflow_probability import distributions as tfd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization

from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard #for keras+theano
#from tensorflow.keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard #for tensorflow, e.g tfp #DOES NOT WORK
from keras.optimizers import Adam
from keras import metrics, regularizers
import pickle
from copy import deepcopy
import os
from functions import *
from PredictExternal import *
import time

import pandas as pd



def TrainAutoencoderNetwork(parameters, inputfolder, outputfolder):

    # Get parameters
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    regmethod=parameters['regmethod']
    regrate=parameters['regrate']
    batchnorm=parameters['batchnorm']
    epochs=parameters['epochs']
    learningrate = parameters['learningrate']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight=parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    train_new_model = True
    try:
        #model = keras.models.load_model('output/'+tag+'/model.h5')
        model = keras.models.load_model(outputfolder+'/model.h5')
        train_new_model = False
    except:
        pass
    if train_new_model: print('Couldn\'t find the model "%s", a new one will be trained!' % (tag))
    else:
        print('Found the model "%s", not training a new one, go on to next function.' % (tag))
        return
    if not os.path.isdir(outputfolder): os.makedirs(outputfolder)

    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix='')


    # Define the network
    model = Sequential()
    kernel_regularizer = None
    if regmethod == 'L1':
        kernel_regularizer=regularizers.l1(regrate)
    elif regmethod == 'L2':
        kernel_regularizer=regularizers.l2(regrate)


    print('Number of input variables: %i' % (input_train.shape[1]))
    model.add(Dense(layers[0], activation='relu', input_shape=(input_train.shape[1],), kernel_regularizer=kernel_regularizer))
    if regmethod == 'dropout': model.add(Dropout(regrate))
    if batchnorm: model.add(BatchNormalization())

    for i in layers[1:len(layers)+1]:
        model.add(Dense(i, activation='relu', kernel_regularizer=kernel_regularizer))
        if batchnorm: model.add(BatchNormalization())
        if regmethod == 'dropout': model.add(Dropout(regrate))

    model.add(Dense(input_train.shape[1], activation='linear', kernel_regularizer=kernel_regularizer)) #FixME: which activation is the best for Autoencoder 

    # Train the network
    opt = keras.optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-6, decay=0.0, amsgrad=False)
    mymetrics = [metrics.categorical_accuracy]
    # mymetrics = [metrics.categorical_accuracy, metrics.mean_squared_error, metrics.categorical_crossentropy, metrics.kullback_leibler_divergence, metrics.cosine_proximity]
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=mymetrics)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=mymetrics)

    model.summary()

    period = epochs / 5
    checkpointer = ModelCheckpoint(filepath=outputfolder+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=period)
    checkpointer_everymodel = ModelCheckpoint(filepath=outputfolder+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, mode='auto', period=1)
    checkpoint_bestmodel = ModelCheckpoint(filepath=outputfolder+'/model_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=100, verbose=0, mode='min', baseline=None, restore_best_weights=True)
    LRreducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_delta=0.001, mode='min')
    weights_train, weights_test = sample_weights_train, sample_weights_test
    if not eqweight:
        weights_train, weights_test = eventweights_train, eventweights_test
    model.fit(input_train, input_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, input_test, weights_test), callbacks=[checkpointer_everymodel, checkpoint_bestmodel], verbose=2)

    model.save(outputfolder+'/model.h5')
    with open(outputfolder+'/model_history.pkl', 'wb') as f:
        pickle.dump(model.history.history, f)


def TrainForMoreEpochs(parameters, nepochs):
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    regmethod=parameters['regmethod']
    regrate=parameters['regrate']
    batchnorm=parameters['batchnorm']
    epochs=parameters['epochs']
    learningrate = parameters['learningrate']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight=parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    new_parameters = parameters
    new_parameters['epochs'] = new_parameters['epochs'] + nepochs
    new_tag = dict_to_str(new_parameters)

    new_model_already_exists = False
    # Find out if the new model would already exist
    try:
        model = keras.models.load_model('output/'+new_tag+'/model.h5')
        new_model_already_exists = True
    except:
        pass
    if new_model_already_exists:
        print('The new model would already exists')
        return new_parameters
    else:
        print('Going to train for %i more epochs.' % nepochs)

    if not os.path.isdir('output/' + new_tag): os.makedirs('output/'+new_tag)



    # Get inputs
    input_train = np.load('input/'+classtag+'/input_'+fraction+'_train.npy')
    input_test = np.load('input/'+classtag+'/input_'+fraction+'_test.npy')
    input_val = np.load('input/'+classtag+'/input_'+fraction+'_val.npy')
    labels_train = np.load('input/'+classtag+'/labels_'+fraction+'_train.npy')
    labels_test = np.load('input/'+classtag+'/labels_'+fraction+'_test.npy')
    labels_val = np.load('input/'+classtag+'/labels_'+fraction+'_val.npy')
    with open('input/'+classtag+'/sample_weights_'+fraction+'_train.pkl', 'r') as f:
        sample_weights_train = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_train.pkl', 'r') as f:
        eventweights_train = pickle.load(f)
    with open('input/'+classtag+'/sample_weights_'+fraction+'_test.pkl', 'r') as f:
        sample_weights_test = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_test.pkl', 'r') as f:
        eventweights_test = pickle.load(f)
    with open('input/'+classtag+'/sample_weights_'+fraction+'_val.pkl', 'r') as f:
        sample_weights_val = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_val.pkl', 'r') as f:
        eventweights_val = pickle.load(f)


    model = keras.models.load_model(outputfolder+'/model.h5')
    with open(outputfolder+'/model_history.pkl', 'r') as f:
        model_history = pickle.load(f)

    # Train the network
    opt = keras.optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mymetrics = [metrics.categorical_accuracy]
    # mymetrics = [metrics.categorical_accuracy, metrics.mean_squared_error, metrics.categorical_crossentropy, metrics.kullback_leibler_divergence]
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=mymetrics)
    model.summary()

    period = epochs / 5
    checkpointer = ModelCheckpoint(filepath=outputfolder+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=period)
    weights_train, weights_test = sample_weights_train, sample_weights_test
    if not eqweight:
        weights_train, weights_test = eventweights_train, eventweights_test
    model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=nepochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer], verbose=1)


    model.save('output/'+new_tag+'/model.h5')
    with open('output/'+new_tag+'/model_history.pkl', 'wb') as f:
        pickle.dump(model.history.history, f)



    # Do the predictions
    print('Now that the model is trained, we\'re going to predict the labels of all 3 sets. ')
    print('predicting for training set')
    pred_train = model.predict(input_train)
    np.save('output/'+new_tag+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print('predicting for training set, class ' + str(cl))
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save('output/'+new_tag+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print('predicting for test set')
    pred_test = model.predict(input_test)
    np.save('output/'+new_tag+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print('predicting for test set, class ' + str(cl))
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save('output/'+new_tag+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print('predicting for val set')
    pred_val = model.predict(input_val)
    np.save('output/'+new_tag+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print('predicting for val set, class ' + str(cl))
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save('output/'+new_tag+'/prediction_val_class'+str(cl)+'.npy'  , tmp)

    return new_parameters


def _prior_normal_fn(sigma, dtype, shape, name, trainable, add_variable_fn):
    """Normal prior with mu=0 and sigma=sigma. Can be passed as an argument to
    the tpf.layers
    """
    del name, trainable, add_variable_fn

    dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(sigma))
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


def TrainBayesianNetwork(parameters, inputfolder, outputfolder):
    print("***** TrainBayesianNetwork *****")
    print("Read inputs from ",inputfolder)
    tfd = tfp.distributions
    load_model = False #FixMe: add option to load model

    # Get parameters
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    regmethod=parameters['regmethod']
    regrate=parameters['regrate']
    batchnorm=parameters['batchnorm']
    epochs=parameters['epochs']
    learningrate = parameters['learningrate']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight=parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    train_new_model = True
    try:
        model = keras.models.load_model(outputfolder+'/model.h5')
        file = h5py.File(outputfolder+'/model_weights.h5'.format(tag), 'r')
        print("File with weights:",outputfolder+'/model_weights.h5'.format(tag))
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)][:])
        train_new_model = False
    except:
        pass
    if train_new_model: print('Couldn\'t find the model "%s", a new one will be trained!' % (tag))
    else:
        print('Found the model "%s", not training a new one, go on to next function.' % (tag))
        return
    if not os.path.isdir(outputfolder): os.makedirs(outputfolder)


    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix='')

    print('---- Data is loaded! ----')
    # pandas summary.h5 file contains model information, trainings info, ...
    # file is created and filled with data later
    pandas_output = os.path.join(outputfolder, "summary.h5")

    #########################################################################
    
    # Begin defining calculation graph:
    # wrapping graph with graph.as_default() is better for the case of starting
    # main() several times in same python code -> each time new graph and new Session
    graph = tf.Graph()
    with graph.as_default():
        

        # Placeholders for later use; handle for feedable iterator
        lr_pl =  tf.placeholder(tf.float32, shape=(), name="learning_rate")
        loss_norm = tf.placeholder(tf.float32, shape=(), name="loss_norm")
        handle = tf.placeholder(tf.string, shape=[], name="handle")

        # just for dense deterministic dropout layer! Not used for BNNs
        is_training = tf.placeholder(tf.bool, shape=(), name="is_training")


        # # # iterator for train and validation dataset
        # Use placeholders to be more flexiable to handle big size of the input
        # See https://www.tensorflow.org/guide/datasets
        input_train_placeholder = tf.placeholder(input_train.dtype,input_train.shape, name="input_train")
        labels_train_placeholder = tf.placeholder(labels_train.dtype,labels_train.shape, name="labels_train")
        sample_weights_train_placeholder = tf.placeholder(sample_weights_train.dtype,sample_weights_train.shape, name = "sample_weights_train")
        dataset_train = tf.data.Dataset.from_tensor_slices((input_train_placeholder,labels_train_placeholder,sample_weights_train_placeholder)) #emit only one data at a time
        batched_dataset_train = dataset_train.batch(batch_size) # Combines consecutive elements of the Dataset into a single batch to train smaller batches of data to avoid out of memory errors.

        input_val_placeholder = tf.placeholder(input_val.dtype,input_val.shape, name="input_val")
        labels_val_placeholder = tf.placeholder(labels_val.dtype,labels_val.shape, name="labels_val")
        sample_weights_val_placeholder = tf.placeholder(sample_weights_val.dtype,sample_weights_val.shape, name="sample_weights_val")
        dataset_val = tf.data.Dataset.from_tensor_slices((input_val_placeholder,labels_val_placeholder,sample_weights_val_placeholder))
        batched_dataset_val = dataset_val.batch(batch_size) #same batch size as for training

        iterator_train = batched_dataset_train.make_initializable_iterator()
        iterator_val = batched_dataset_val.make_initializable_iterator()

        # feedable iterator, for switching between train/test/val
        iterator = tf.data.Iterator.from_string_handle(
            handle, batched_dataset_train.output_types, batched_dataset_train.output_shapes)
        next_element, next_labels, next_sample_weights = iterator.get_next()

        # Define the network
        # From https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout
#        sigma=1.0 #FixME, read https://arxiv.org/pdf/1904.10004.pdf to choose right value
        sigma=parameters['sigma']
        prior = lambda x, y, z, w, k: _prior_normal_fn(sigma, x, y, z, w, k)
#        method = lambda d: d.mean() # for test with deterministic NN
        method = lambda d: d.sample()
        layer_hd = []
        batchnorm_hd = []
        dropout_hd = []

        #### Without DropOut
        inputs = tf.keras.layers.Input(shape=(input_train.shape[1],),name="Input")
        layer_hd.append(tfp.layers.DenseFlipout(layers[0], activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method, name="DenseFlipout_1")(inputs))
        k=1
        for i in layers[1:len(layers)+1]:
            label=str(k+1)
            layer_hd.append(tfp.layers.DenseFlipout(i, activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method,name="DenseFlipout_"+label)(layer_hd[k-1]))
            k = k+1
        label=str(k)
        last_layer = tfp.layers.DenseFlipout(labels_train.shape[1], kernel_prior_fn=prior, kernel_posterior_tensor_fn=method,name="DenseFlipout_last")(layer_hd[k-1]) #not normilised output
        model = tf.keras.models.Model(inputs=inputs,outputs=last_layer)
        model.summary()         

        logits = model(next_element)
        label_distribution = tf.nn.softmax(logits, name="label_distribution") # probability distribution around k classes

        # Loss function:
        # For a BNN we minimize loss = KL(var. posterior | posterior)
        # this can be fruther decomposed into
        # loss = sum_i neg_log_likelihood_i + KL(var. posterior | prior)/train-size
        # the sum goes over N Monte Carlo samples. In practice we set N=1 for training.
        # the /train-size normalization comes from the fact that we use the mean over
        # on batch instead of a sum (tf.reduce_mean(cross_entropy)).
        # KL(var. posterior | posterior) is accessible as an attribute of the tfp.layers
        # and is computed analytically (at least for gaussian prior).
        # loss_reg stands for regularization:
        #     BNN: loss_reg = KL(var. posterior |prior)/train-size
        #     deterministic: loss_ref = L2 regularization

        labels = tf.cast(next_labels, tf.float32)
        weights = tf.cast(next_sample_weights, tf.float32)

        
        ## In mutually exclusive multilabel classification, we use softmax_cross_entropy_with_logits, which behaves differently: 
        ## each output channel corresponds to the score of a class candidate. The decision comes after, by comparing the respective outputs of each channel.
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name="cross_entropy")
        weighted_cross_entropy = cross_entropy * weights #apply the weights, relying on broadcasting of the multiplication
        neg_log_likelihood = tf.reduce_mean(weighted_cross_entropy)
        loss_reg = tf.reduce_sum(model.losses)

    
        if loss_reg == 0: # case for deterministic L2-scale=0
            loss_reg = tf.constant(0., dtype=neg_log_likelihood.dtype)
        
        # for BNN case need /train-size factor!
        is_deterministic = False
        if not is_deterministic:
            loss_reg /= batch_size
        loss = tf.add(neg_log_likelihood, loss_reg, name="loss")
       
        
        opt = tf.train.AdamOptimizer(learning_rate=learningrate, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False) #TEST
        #opt = keras.optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        train_op = opt.minimize(loss, name="train_op")
        ################################
        # Calculating mean predictive mean / stddev / accuracy and ROC for testing
        
        # Defintion of DenseFlipout layer in tensorflow probability:
        #   "This layer implements the Bayesian variational inference analogue to
        #   a dense layer by assuming the `kernel` and/or the `bias` are drawn
        #   from distributions. By default, the layer implements a stochastic
        #   forward pass via sampling from the kernel and bias posteriors"
        #
        # Calling label_distribution = neural_net(image) in the Session  will give us
        # one forward pass. If we want to sample several times, we need to call
        # label_distributions several times. This is implemented via one big loop
        # in tf.Session: Loop over all data-batches and MonteCarlo Samples
        #
        # Prediction of a BNN is given by the Mean over n MonteCarlo samples.
        # the mean is calculated as a running mean via tf.metrics.mean_tensor() here.
        # The std is calculated via std**2 = <x**2> - <x>**2 and tf.metrics.mean_tensor().
        
        # predicted label for single forward pass (n_MonteCarlo = 1)
        predicted_label = tf.round(label_distribution)
        
        # name_scope: be able to reset the internal variable independendly!
        # cheap estimate of accuracy via one forward pass (n_MonteCarlo = 1)

        with tf.name_scope('train'):
            train_acc, train_acc_op = tf.metrics.accuracy(labels=labels, predictions=predicted_label)
        with tf.name_scope('val'):
            val_acc, val_acc_op = tf.metrics.accuracy(labels=labels, predictions=predicted_label)

        ############################################################################
    
        # shitty set_shape! Problem: mean_tensor operation needs clearly defined
        # tensor shapes... (Maybe find a better solution). This will lead to
        # problems if one wants to load full meta graph with different batch_size
        label_distribution.set_shape([batch_size,labels_train.shape[1]])
        with tf.name_scope('mean'):
            mean, mean_op = tf.metrics.mean_tensor(label_distribution)
            
        with tf.name_scope("square_mean"):
                sq_mean, sq_mean_op = tf.metrics.mean_tensor(tf.square(label_distribution))
                
        # standard deviation, give zero if var<0 (happens when mean_op is close to 1,
        # so very small stddev)
        var = tf.subtract(sq_mean_op, tf.square(mean_op)) # variance
        stddev = tf.where(tf.greater(var, 0), tf.sqrt(var), tf.zeros_like(var))

        #############################################################################
        # same for unnormalized output (before sigmoid)
        logits.set_shape([batch_size,labels_train.shape[1]])
        with tf.name_scope('mean_logits'):
            mean_logits, mean_logits_op = tf.metrics.mean_tensor(logits)
        
        with tf.name_scope("square_mean_logits"):
            sq_mean_logits, sq_mean_logits_op = tf.metrics.mean_tensor(tf.square(logits))

        # standard deviation
        var_logits = tf.subtract(sq_mean_logits_op, tf.square(mean_logits_op)) # variance
        stddev_logits = tf.where(tf.greater(var_logits, 0), tf.sqrt(var_logits), tf.zeros_like(var_logits))

        ###################################################################################

        true_predicted_label = tf.cast(tf.round(mean_op), labels.dtype)
        acc_true, acc_true_op  = tf.metrics.mean(tf.cast(tf.equal(true_predicted_label, labels), tf.float32))

        num_thresholds = 5000 # default is 200
        auc, auc_op = tf.metrics.auc(labels, mean_op, num_thresholds=num_thresholds)
        roc_curve, roc_op = tf.contrib.metrics.streaming_curve_points(labels, mean_op, num_thresholds=num_thresholds)


        # copied from tensorflow probability example
        # Extract weight posterior statistics for layers with weight distributions
        names = []
        qmeans = []
        qstds = []
        for i, layer in enumerate(model.layers):
            try:
                q = layer.kernel_posterior
            except AttributeError:
                continue
            names.append("Layer {}".format(i))
            qmeans.append(q.mean())
            qstds.append(q.stddev())

        # Add ops to save and restore all the variables
        saver = tf.train.Saver()

    ############################################################################
    # End of computational graph and begin of session to run the graph and plot
    # the results. The predctions, weight-array, labels, ROC etc. are saved into
    # h5 file (summary.h5) with different keys

    #Allow growing memory for GPU, see https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
    #with tf.Session() as sess:

        # initialize all  variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # saving loss and accuracy values into lists for plotting
        loss_val_all, acc_all, acc_val_all = [], [], []
        loss_average_all, loss_likelihood_all = [], []

        # training methods
        lr_reduce = lr_reduce_on_plateau(
            lr=learningrate, patience=50,
            fraction=0.1, delta_improv=0.)
        check_early_stop = check_early_stopping(
            patience=100, delta_improv=0.)

        if not load_model:

            # for feeding handle and switching between iterators
            train_handle = sess.run(iterator_train.string_handle())
            validation_handle = sess.run(iterator_val.string_handle())

            print("-"*100)
            print("Start Training...")

            # ##Store prediction in each epoch for validation
            # traings loop, epoch starts with 0
            for epoch in range(epochs):
                sess.run(iterator_train.initializer, feed_dict={input_train_placeholder: input_train, labels_train_placeholder: labels_train, sample_weights_train_placeholder: sample_weights_train,
                                                                input_val_placeholder: input_val, labels_val_placeholder: labels_val, sample_weights_val_placeholder: sample_weights_val})
                start_time = time.time()

                # to save in-epoch-loss
                loss_in_epoch = []

                # loop over all batches (one epoch)
                while True:
                    try:
                        #sess.run(next_element)
                        _, _, loss_value= sess.run([train_op, train_acc_op, loss],
                                                   feed_dict={loss_norm: batch_size,
                                                              lr_pl: lr_reduce.lr,
                                                              handle: train_handle,
                                                              is_training: True,
                                                              
                                                          })
                        loss_in_epoch.append(loss_value)
                    except tf.errors.OutOfRangeError:
                        break           

                # mean over in-epoch loss
                loss_average = np.mean(loss_in_epoch)
                # Estimation of true accuracy!
                sess.run(iterator_train.initializer, feed_dict={input_train_placeholder: input_train, labels_train_placeholder: labels_train, 
                                                                sample_weights_train_placeholder: sample_weights_train,
                                                                input_val_placeholder: input_val, labels_val_placeholder: labels_val, 
                                                                sample_weights_val_placeholder: sample_weights_val}) # need for loss evaluation
                acc_true_value, train_acc_value, loss_reg_value = sess.run([acc_true, train_acc, loss_reg],
                            feed_dict={loss_norm: batch_size,
                                       handle: train_handle,
                                       is_training: False,
                                   })
                # reseting accuracy for val sample
                stream_vars_test = [v for v in tf.local_variables() if 'val/' in v.name]
                sess.run(tf.variables_initializer(stream_vars_test))

                # resetting trainings acc. as well (could actually be done in one steop)
                stream_vars_test = [v for v in tf.local_variables() if 'train/' in v.name]
                sess.run(tf.variables_initializer(stream_vars_test))

                # doing validation also in batches because of large val sample size
                sess.run(iterator_val.initializer, feed_dict={input_train_placeholder: input_train, labels_train_placeholder: labels_train, sample_weights_train_placeholder: sample_weights_train,
                                                              input_val_placeholder: input_val, labels_val_placeholder: labels_val, sample_weights_val_placeholder: sample_weights_val})
                val_loss_list = []
                while True:
                    try:
                        val_acc_value, val_loss = sess.run([val_acc_op, loss],
                            feed_dict={loss_norm: batch_size,
                                handle: validation_handle,
                                is_training: False,
                        }) # same train_size
                        val_loss_list.append(val_loss)
                    except tf.errors.OutOfRangeError:
                        break

                val_loss = np.mean(val_loss_list)
                end_time = time.time()
                time_duration = end_time - start_time

                # saving as [] for later ploting
                loss_val_all.append(val_loss)
                acc_all.append(train_acc_value)
                acc_val_all.append(val_acc_value)
                loss_average_all.append(loss_average)

                print("Epoch: {:d}, loss: {:4.2f}, train_acc: {:4.3f} ".format(
                    epoch+1, loss_average, train_acc_value) +\
                    "val_loss: {:4.2f}, val_acc: {:4.3f} (time: {:1.1f}s)".format(
                    val_loss, val_acc_value, time_duration))

                # Seperate loss-values: loss = loss_reg + log_likelihood
                loss_likelihood = loss_average - loss_reg_value
                loss_likelihood_all.append(loss_likelihood)
                print("loss_regularization: {:}, log_likelihood: {:}".format(loss_reg_value, loss_likelihood))

                # update_step: trainings methods
                lr_reduce.update_on_plateau(val_loss)
                if check_early_stop.update_and_check(val_loss):
                    break

            # mean and standard deviation of weights
            mean_vals, std_vals = sess.run([qmeans, qstds])

            # saving trainings parameter (loss, acc) to pandas h5 file
            columns = ["loss", "val_loss", "categorical_accuracy", "val_categorical_accuracy", "loss_likelihood"]
            data  = np.stack([loss_average_all, loss_val_all, acc_all, acc_val_all, loss_likelihood_all], axis=0).T
            df    = pd.DataFrame(data=data, columns=columns)
            df.to_hdf(pandas_output, key="training", format='table', complib = "blosc", complevel=5)

            # ######################################################################################



        # Not possible to store `model as for Keras
        # Work around https://github.com/tensorflow/probability/issues/325#issuecomment-477213850
        file = h5py.File(outputfolder+'/model_weights.h5'.format(tag), 'w')
        weight = model.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()
#        print("Stored model, layer 3:", model.layers[3].get_weights()[0][0])

        # safe model (weights) and entire meta graph
        save_path = saver.save(sess,outputfolder+'/model_weights_and_graph.ckpt'.format(tag))
        print("Finished training!")

      

# #The same as BayesianNetwork but with simple Dense layers from TensorFlow for debuging
# def TrainDeepNetwork(parameters):
#     tfd = tfp.distributions

#     # Get parameters
#     layers=parameters['layers']
#     batch_size=parameters['batchsize']
#     regmethod=parameters['regmethod']
#     regrate=parameters['regrate']
#     batchnorm=parameters['batchnorm']
#     epochs=parameters['epochs']
#     learningrate = parameters['learningrate']
#     runonfraction = parameters['runonfraction']
#     fraction = get_fraction(parameters)
#     eqweight=parameters['eqweight']
#     tag = dict_to_str(parameters)
#     classtag = get_classes_tag(parameters)
#     train_new_model = True
#     #FixME: can't simply re-load model with tfp, only weights are stored -> have to (re)build it 
#     if not os.path.isdir('output/DNN_' + tag): os.makedirs('output/DNN_'+tag)

#     input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder='input/'+classtag, filepostfix='')
#     kernel_regularizer = None
#     if regmethod == 'L1':
#         kernel_regularizer=regularizers.l1(regrate)
#     elif regmethod == 'L2':
#         kernel_regularizer=regularizers.l2(regrate)


#     # Define the network
#     # From https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout
#     sigma=parameters['sigma'] #FixME, read https://arxiv.org/pdf/1904.10004.pdf to choose right value
#     prior = lambda x, y, z, w, k: _prior_normal_fn(sigma, x, y, z, w, k)
#     method = lambda d: d.mean()
# #  method = lambda d: d.sample()
#     layer_hd = []
#     batchnorm_hd = []
#     dropout_hd = []
#     inputs = tf.keras.layers.Input(shape=(input_train.shape[1],))

# #     #With dropout
#     layer_hd.append(tf.layers.Dense(layers[0], activation=tf.nn.relu)(inputs)) 
#     dropout_hd.append(tf.layers.dropout(layer_hd[0], rate=regrate)) #FixME: regmethod might be different
#     #batchnorm_hd.append(tf.layers.batch_normalization(dropout_hd[0]))
#     k=1
#     for i in layers[1:len(layers)+1]:
#         print("current k:",k)
#         label=str(k+1)
#         layer_hd.append(tf.layers.Dense(i, activation=tf.nn.relu)(dropout_hd[k-1]))
#         dropout_hd.append(tf.layers.dropout(layer_hd[k], rate=regrate))
#         #batchnorm_hd.append(tf.layers.batch_normalization(dropout_hd[k]))
#         k = k+1
#     print("total number of hidden layers:",k)
#     last_layer = tf.layers.Dense(labels_train.shape[1], activation='softmax')(dropout_hd[k-1])

   

#     print('Number of output classes: %i' % (labels_train.shape[1]))
#     model = tf.keras.models.Model(inputs=inputs,outputs=last_layer)
#     opt  = tf.train.AdamOptimizer()
#     mymetrics = [metrics.categorical_accuracy]
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=mymetrics)
#     #model.compile(loss='categorical_hinge', optimizer=opt, metrics=mymetrics) #TEST
#     model.summary()

#     period = epochs / 5
#     #FixME: callbacks don't work for tfp, find way around
#     #For ModelCheckpoint store only weights due to issue with layers, see https://github.com/tensorflow/probability/issues/406#issuecomment-493958022
#     checkpointer = ModelCheckpoint(filepath='output/DNN_'+tag+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=period, save_weights_only=True)
#     checkpointer_everymodel = ModelCheckpoint(filepath='output/DNN_'+tag+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=1, save_weights_only=True)
#     #checkpoint_bestmodel = ModelCheckpoint(filepath='output/DNN_'+tag+'/model_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
#     checkpoint_bestmodel = ModelCheckpoint(filepath='output/DNN_'+tag+'/model_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', period=1)
#     earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=0, mode='min', baseline=None, restore_best_weights=True)
#     LRreducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_delta=0.001, mode='min')
#     tensorboard = TensorBoard(log_dir='logs/DNN_'+tag+'/',histogram_freq=50, write_graph=True, write_images=True)
#     weights_train, weights_test = sample_weights_train, sample_weights_test
#     if not eqweight:
#         weights_train, weights_test = eventweights_train, eventweights_test
#     #model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer_everymodel, checkpoint_bestmodel], verbose=2)
#     #model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[tensorboard], verbose=2)

#     model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), verbose=2)

#     # Not possible to store model as for Keras
#     # Work around https://github.com/tensorflow/probability/issues/325#issuecomment-477213850
#     file = h5py.File('output/DNN_'+tag+'/model.h5', 'w')
#     weight = model.get_weights()
#     for i in range(len(weight)):
#         file.create_dataset('weight' + str(i), data=weight[i])
#     file.close()

#     with open('output/DNN_'+tag+'/model_history.pkl', 'w') as f:
#         pickle.dump(model.history.history, f)
