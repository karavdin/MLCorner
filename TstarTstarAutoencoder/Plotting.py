import numpy as np
from numpy import inf
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import metrics, regularizers

from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, gStyle
from ROOT import gErrorIgnoreLevel, kInfo, kWarning, kError

import math
import pickle
import sys
import os
from functions import *
from constants import *
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import h5py
import pandas as pd


# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def _prior_normal_fn(sigma, dtype, shape, name, trainable, add_variable_fn):
    """Normal prior with mu=0 and sigma=sigma. Can be passed as an argument to                                                                                               
    the tpf.layers                                                                                                                                                           
    """
    del name, trainable, add_variable_fn
    dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(sigma))
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


def PlotAutoencoderPerformance(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print('Now plotting the performance')
    gErrorIgnoreLevel = kWarning

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Get model and its history
    model = keras.models.load_model(outputfolder+'/model.h5')
    with open(outputfolder+'/model_history.pkl', 'rb') as f:
        model_history = pickle.load(f, encoding='bytes')

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train, pred_test, pred_val, pred_signals, mse_train, mse_test, mse_val, mse_signals = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix) 

    #print("pred_train",pred_train)
    #print("mse_train",mse_train)

    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)


    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder)
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf')

    pred_trains, mse_trains, weights_trains, normweights_trains, lumiweights_trains,  pred_vals, mse_vals, weights_vals, normweights_vals, lumiweights_vals,  pred_tests, mse_tests , weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, mse_train=mse_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val, mse_val=mse_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test, mse_test=mse_test)

    plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, input_trains=input_train, pred_trains=pred_trains,  labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, isMSE=False)

    plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, input_trains=input_train, pred_trains=mse_trains,  labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=mse_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=mse_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, isMSE=True)

    # plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    
    #Store model as JSON file for usage in UHH2
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open(outputfolder+'/architecture.json', 'w') as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights(outputfolder+'/weights.h5')
    print("--- END of AutoencoderNN Plotting ---")


def PlotInputs(parameters, inputfolder, filepostfix, plotfolder):

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    classtag = get_classes_tag(parameters)
    tag = dict_to_str(parameters)

    if not os.path.isdir(plotfolder):
        os.makedirs(plotfolder)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    with open(inputfolder+'/variable_names.pkl', 'rb') as f:
        variable_names = pickle.load(f, encoding='bytes')

    # Divide into classes
    input_train_classes = {}
    input_test_classes = {}
    input_val_classes = {}
    weights_train_classes = {}
    weights_test_classes = {}
    weights_val_classes = {}
    for i in range(labels_train.shape[1]):
        input_train_classes[i] = input_train[labels_train[:,i] == 1]
        input_test_classes[i] = input_test[labels_test[:,i] == 1]
        input_val_classes[i] = input_val[labels_val[:,i] == 1]
        weights_train_classes[i] = sample_weights_train[labels_train[:,i] == 1]
        weights_test_classes[i] = sample_weights_test[labels_test[:,i] == 1]
        weights_val_classes[i] = sample_weights_val[labels_val[:,i] == 1]

    # Create class-title dictionary
    classes = parameters['classes']
    classtitles = {}
    for key in classes.keys():
        list = classes[key]
        title = ''
        for i in range(len(list)):
            title = title + list[i]
            if i < len(list)-1:
                title = title + '+'
        classtitles[key] = title

    matplotlib.style.use('default')
    nbins = 50
    idx = 0
    for varname in variable_names:
        #varname = varname.decoding('utf-8') #value from bytes
        varname = str(varname, 'utf-8', 'ignore')
        #print("varname",varname)
        xmax = max([max(input_train_classes[i][:,idx]) for i in range(len(input_train_classes))])
        xmin = min([min(input_train_classes[i][:,idx]) for i in range(len(input_train_classes))])
        if xmax == xmin: xmax = xmin + 1.
        xmin = min([0,xmin])
        #print("xmax, xmin",xmax, xmin)
        binwidth = (xmax - xmin) / float(nbins)
        bins = np.arange(xmin, xmax + binwidth, binwidth)

        plt.clf()
        fig = plt.figure()
        for i in range(len(input_train_classes)):
            mycolor = 'C'+str(i)
            plt.hist(input_train_classes[i][:,idx], weights=weights_train_classes[i], bins=bins, histtype='step', label='Training sample, '+classtitles[i], color=colorstr[i])
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(varname)
        plt.ylabel('Number of events / bin')
        fig.savefig(plotfolder + '/' + varname + '_'+fraction+'.pdf')
        idx += 1

        sys.stdout.write( '{0:d} of {1:d} plots done.\r'.format(idx, len(variable_names)))
        if not i == len(variable_names): sys.stdout.flush()
        plt.close()

def plot_loss(parameters, plotfolder, model_history):
    print('Starting to plot Loss')
    eqweight = parameters['eqweight']

    # def fitfunc(x, a, b, c, d, e):
    #     return a + b/x + c*x + d*x*x + e/x/x

    tag = dict_to_str(parameters)
    plt.clf()
    fig = plt.figure()
    plt.grid()
    x, fitx, fitfunc, postfitpars_train = fit_loss(model_history['loss'])
    x, fitx, fitfunc, postfitpars_val = fit_loss(model_history['val_loss'])

    plt.plot(x, model_history['loss'], label = 'Training set')
    plt.plot(x, model_history['val_loss'], label = 'Validation set')
    #plt.plot(fitx, fitfunc(fitx, *postfitpars_train), label="Fit (training set)") #FixME for BNN
    #plt.plot(fitx, fitfunc(fitx, *postfitpars_val), label="Fit (validation set)") #FixME for BNN

    plt.legend(loc='upper right')
    #plt.ylim([0.1, 0.25])
    plt.ylim([0.01,100])
    plt.yscale("log")
    if eqweight:
#        plt.ylim([0.01, 0.06])
        plt.ylim([0.1,100])
    plt.ylabel('Loss')
    plt.xlabel('Number of training epochs')
    fig.savefig(plotfolder+'/Loss.pdf')
    plt.close()

def plot_accuracy(parameters, plotfolder, model_history):
    print('Starting to plot accuracy')
    tag = dict_to_str(parameters)
    plt.clf()
    fig = plt.figure()
    plt.grid()
    x = range(len(model_history['categorical_accuracy'])+1)
    x = x[1:]
    plt.plot(x, model_history['categorical_accuracy'], label = 'Training set')
    plt.plot(x, model_history['val_categorical_accuracy'], label = 'Validation set')
    plt.legend(loc='lower right')
    plt.ylim([0., 1.05])
    plt.ylabel('Prediction accuracy')
    plt.xlabel('Number of training epochs')
    fig.savefig(plotfolder+'/Accuracy.pdf')

def plot_outputs_1d_nodes(parameters, plotfolder, input_trains, pred_trains, labels_train, weights_trains, lumiweights_trains, normweights_trains, pred_vals, labels_val, weights_vals, lumiweights_vals, normweights_vals, pred_signals=None, eventweight_signals=None, normweight_signals=None, usesignals=[0], isMSE=False):

    print('Starting to plot the classifier output distribution')
    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)
    do_sig = (pred_signals is not None) and (eventweight_signals is not None) and (normweight_signals is not None)
#    do_sig = False #TEST
 #   print("pred_trains=",pred_trains)
    for cl in range(input_trains.shape[1]):
        print("cl of plot:",cl)
#        print("pred_trains[",cl,"][0]=",pred_trains[cl][0])
        #print("pred_trains[0][",cl,"]=",pred_trains[0][cl])
        # 'cl' is the output node number

        nbins = 100
        binwidth = 1./float(nbins)
        y_trains = {}
        y_vals = {}
        y_trains_norm = {}
        y_vals_norm = {}
        bin_edges_trains = {}
        bin_edges_vals = {}
        bin_centers = {}
        yerrs = {}
        yerrs_norm = {}
        y_signals = {}
        yerrs_signals = {}

        for i in range(1):
          # 'i' is the true class (always the first index)
            #print("pred_trains[i][cl]",pred_trains[i][cl])
            #print("Now i is",i," ",pred_trains.shape) ##CHECK DIMENSION OF DICT!!!
            y_trains[i], dummy = np.histogram(pred_trains[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
            y_trains_norm[i], bin_edges_trains[i] = np.histogram(pred_trains[i][cl], bins=nbins, weights=weights_trains[i][cl])
            y_vals[i], dummy = np.histogram(pred_vals[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
            y_vals_norm[i], bin_edges_vals[i] = np.histogram(pred_vals[i][cl], bins=nbins, weights=weights_vals[i][cl])
            bin_centers[i] = 0.5*(bin_edges_trains[i][1:] + bin_edges_trains[i][:-1])
            yerrs_norm[i] = y_vals_norm[i]**0.5
            yerrs[i] = y_vals[i]**0.5
            y_vals_norm[i] = y_vals_norm[i] * normweights_vals[i][cl][0]
            yerrs_norm[i] = yerrs_norm[i] * normweights_vals[i][cl][0]

        if do_sig:
            for key in pred_signals.keys():
                y_signals[key], dummy = np.histogram(pred_signals[key][:,cl], bins=nbins, weights=eventweight_signals[key])
                yerrs_signals[key] = y_signals[key]**0.5

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(1):
            classtitle_to_use = classtitles[i]
            #print("pred_trains[i][cl]",pred_trains[i][cl])
            plt.hist(pred_trains[i][cl], weights=weights_trains[i][cl]*normweights_trains[i][cl], bins=nbins, histtype='step', label='Training sample, ' + classtitles[i], color=colorstr[i])
            plt.errorbar(bin_centers[i], y_vals_norm[i], yerr=yerrs_norm[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            # if i == cl:
            #     #classtitle_to_use = classtitles[i]
            #     classtitle_to_use = str(i) #FixME: read name of variable instead of var id
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.hist(pred_signals[usesignals[sigidx]][:,cl], weights=eventweight_signals[usesignals[sigidx]]*normweight_signals[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])
        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        xaxis_name = ''
        if isMSE: 
            xaxis_name +='MSE for ' 
        else:
            xaxis_name +='Value of ' 
    
        plt.xlabel(xaxis_name+'output for node '+classtitle_to_use)
        plt.ylabel('Normalized number of events / bin')
        title = 'Distribution_node'+str(cl)+'_norm'
        if isMSE: title += '_MSE'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(1):
            classtitle_to_use = classtitles[i]
            plt.errorbar(bin_centers[i], y_vals[i], yerr=yerrs[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            # print i, y_vals[i]
#             if i == cl:
# #                classtitle_to_use = classtitles[i]
#                 classtitle_to_use = str(i) #FixME: read name of variable instead of var id
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.hist(pred_signals[usesignals[sigidx]][:,cl], weights=eventweight_signals[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])
                # plt.errorbar(bin_centers[i], y_signals[usesignal], yerr=yerrs_signals[usesignal], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Signal sample', color='k')

        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        xaxis_name = ''
        if isMSE: 
            xaxis_name +='MSE for ' 
        else:
            xaxis_name +='Value of ' 
    
        plt.xlabel(xaxis_name+'output for node '+classtitle_to_use)
        plt.ylabel('Number of events / bin (weighted by luminosity)')
        title = 'Distribution_node'+str(cl)
        if isMSE: title += '_MSE'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)
        plt.close()


def plot_outputs_1d_classes(parameters, plotfolder, pred_trains, labels_train, weights_trains, lumiweights_trains, normweights_trains, pred_vals, labels_val, weights_vals, lumiweights_vals, normweights_vals, use_best_model=False):

    print('Starting to plot the classifier output distribution')
    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)

    for cl in range(labels_train.shape[1]):
        # 'cl' is the output node number
        nbins = 50
        binwidth = 1./float(nbins)
        y_trains = {}
        y_vals = {}
        y_trains_norm = {}
        y_vals_norm = {}
        bin_edges_trains = {}
        bin_edges_vals = {}
        bin_centers = {}
        yerrs = {}
        yerrs_norm = {}
        #
        # print 'class: ', cl
        # print 'integral lumiweighted: ', lumiweights_vals[cl][0].sum()
        # print 'integral lumi+classwe: ', weights_vals[cl][0].sum()
        # print 'integral normed lumi+classwe: ', (weights_vals[cl][0]*normweights_vals[cl][0]).sum()


        for i in range(labels_train.shape[1]):
            # 'i' is the true class (always the first index)
            y_trains[i], dummy = np.histogram(pred_trains[cl][i], bins=nbins, weights=lumiweights_trains[cl][i])
            y_trains_norm[i], bin_edges_trains[i] = np.histogram(pred_trains[cl][i], bins=nbins, weights=weights_trains[cl][i])
            y_vals[i], dummy = np.histogram(pred_vals[cl][i], bins=nbins, weights=lumiweights_vals[cl][i])
            y_vals_norm[i], bin_edges_vals[i] = np.histogram(pred_vals[cl][i], bins=nbins, weights=weights_vals[cl][i])
            bin_centers[i] = 0.5*(bin_edges_trains[i][1:] + bin_edges_trains[i][:-1])
            yerrs_norm[i] = y_vals_norm[i]**0.5
            yerrs[i] = y_vals[i]**0.5
            y_vals_norm[i] = y_vals_norm[i] * normweights_vals[cl][i][0]
            yerrs_norm[i] = yerrs_norm[i] * normweights_vals[cl][i][0]

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            plt.hist(pred_trains[cl][i], weights=weights_trains[cl][i]*normweights_trains[cl][i], bins=nbins, histtype='step', label='Training sample, ' + classtitles[i], color=colorstr[i])
            plt.errorbar(bin_centers[i], y_vals_norm[i], yerr=yerrs_norm[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            if i == cl:
                classtitle_to_use = classtitles[i]

        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('Classifier output for true class '+classtitle_to_use)
        plt.ylabel('Normalized number of events / bin')
        title = 'Distribution_class'+str(cl)+'_norm'
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            plt.errorbar(bin_centers[i], y_vals[i], yerr=yerrs[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            if i == cl:
                classtitle_to_use = classtitles[i]

        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('Classifier output for true class '+classtitle_to_use)
        plt.ylabel('Number of events / bin (weighted by luminosity)')
        title = 'Distribution_class'+str(cl)+''
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)
        plt.close()


def plot_prediction_samples(parameters, plotfolder, pred_train_all, labels_train, eventID):
    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)
    
    print('Starting to plot prediction in all samples for 1 event and 1 class')
    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)
#    y_trains[i], dummy = np.histogram(pred_trains_all[:,eventID,class_label], bins=nbins)

    nbins = 25
    binwidth = 1./float(nbins)

    plt.clf()
    fig = plt.figure()
    true_lable = 'true lable: '
    for i in range(labels_train.shape[1]):
        plt.hist(pred_train_all[:,eventID,i], bins=nbins, histtype='step', label='Training sample, prediction for ' + classtitles[i]+' node', color=colorstr[i])
        if(labels_train[eventID][i]>0):
            true_lable = true_lable + classtitles[i]
    plt.legend(loc='best', prop={'size': 8})
    plt.yscale('log')
    plt.xlim([-0.05, 1.05])
#    plt.xlabel('Classifier output for class #'+str(class_label)+', event '+str(eventID))
    plt.xlabel('Classifier output for event '+str(eventID)+', '+true_lable)
    plt.ylabel('Number of events / bin')

    #title = 'Distribution_event'+str(eventID)+'_class'+str(class_label)
    title = 'Distribution_train_event'+str(eventID)
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()
