import numpy as np
from numpy import inf
import itertools
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils import check_consistent_length, assert_all_finite, column_or_1d, check_array
import scipy.optimize as opt
from scipy.optimize import fsolve
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from copy import deepcopy

from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, kOrange, gStyle
from ROOT import gErrorIgnoreLevel
from ROOT import kInfo, kWarning, kError

from constants import *

import math
import pickle
import sys
import os

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

def dict_to_str(parameters):
    layers_str = [str(parameters['layers'][i]) for i in range(len(parameters['layers']))]
    tag = 'layers_'
    for i in range(len(layers_str)):
        tag = tag + layers_str[i]
        if i < len(layers_str)-1: tag = tag + '_'
    tag = tag + '__batchsize_' + str(parameters['batchsize'])
    tag = tag + '__classes_' + str(len(parameters['classes'])) + '_'
    for i in range(len(parameters['classes'])):
        for j in range(len(parameters['classes'][i])):
            tag = tag + parameters['classes'][i][j]
            if j < len(parameters['classes'][i]) - 1:
                tag = tag + '+'
        if i < len(parameters['classes']) - 1:
            tag = tag + '_'

    tag = tag + '__regmethod_' + parameters['regmethod']
    tag = tag + '__regrate_' + '{num:06d}'.format(num=int(parameters['regrate']*100000.))
    tag = tag + '__batchnorm_' + str(parameters['batchnorm'])
    tag = tag + '__epochs_' + str(parameters['epochs'])
    tag = tag + '__learningrate_' + '{num:06d}'.format(num=int(parameters['learningrate']*100000.))
    tag = tag + '__runonfraction_' + '{num:03d}'.format(num=int(parameters['runonfraction']*100.))
    tag = tag + '__eqweight_' + str(parameters['eqweight'])
    #tag = tag + '__preprocess_' + str(parameters['preprocess'])
    #tag = tag + '__priorSigma_' + '{num:03d}'.format(num=int(parameters['sigma']*100.))
    #print("------ Sigma in TAG: ",parameters['sigma'])
    #if len(tag.split('__')) != len(parameters): raise ValueError('in dict_to_str: Number of parameters given in the dictionary does no longer match the prescription how to build the tag out of it.')
    return tag

def get_classes_tag(parameters):
    tag = 'classes_' + str(len(parameters['classes'])) + '_'
    for i in range(len(parameters['classes'])):
        for j in range(len(parameters['classes'][i])):
            tag = tag + parameters['classes'][i][j]
            if j < len(parameters['classes'][i]) - 1:
                tag = tag + '+'
        if i < len(parameters['classes']) - 1:
            tag = tag + '_'
    return tag

def get_classtitles(parameters):
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
    return classtitles

def get_fraction(parameters):
    runonfraction = parameters['runonfraction']
    string = str('{num:03d}'.format(num=int(parameters['runonfraction']*100.)))
    return string


#load input data
def load_data(parameters, inputfolder, filepostfix):

    print('Loading data...')
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)


    input_train = np.load(inputfolder+'/input_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    input_test = np.load(inputfolder+'/input_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    input_val = np.load(inputfolder+'/input_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)

    labels_train = np.load(inputfolder+'/labels_'+fraction+'_train'+filepostfix+'.npy')
    labels_test = np.load(inputfolder+'/labels_'+fraction+'_test'+filepostfix+'.npy')
    labels_val = np.load(inputfolder+'/labels_'+fraction+'_val'+filepostfix+'.npy')
    sample_weights_train = np.load(inputfolder+'/sample_weights_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    eventweights_train = np.load(inputfolder+'/eventweights_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    sample_weights_test = np.load(inputfolder+'/sample_weights_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    eventweights_test = np.load(inputfolder+'/eventweights_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    sample_weights_val = np.load(inputfolder+'/sample_weights_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)
    eventweights_val = np.load(inputfolder+'/eventweights_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)

    signal_identifiers = ['TstarTstarToTgluonTgluon_All', 'TstarTstarToTgluonTgluon_M-700', 'TstarTstarToTgluonTgluon_M-1500'] #FixME: should be set in steer file

    signals = {}
    signal_eventweights = {}
    signal_normweights = {}
    for i in range(len(signal_identifiers)):
        signals[i] = np.load(inputfolder+'/' + signal_identifiers[i] + filepostfix+'.npy').astype(np.float32)
        signal_eventweights[i] = np.load(inputfolder+'/' + signal_identifiers[i] + '_eventweight'+filepostfix+'.npy').astype(np.float32)
        sum_signal_eventweights = signal_eventweights[i].sum()
        signal_normweights[i] = np.array([1./sum_signal_eventweights for j in range(signal_eventweights[i].shape[0])])
    return input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights
    


def load_predictions(outputfolder, filepostfix):

    print('Loading predictions... for filepostfix:',filepostfix)
#    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']
    signal_identifiers = ['TstarTstarToTgluonTgluon_All', 'TstarTstarToTgluonTgluon_M-700', 'TstarTstarToTgluonTgluon_M-1500'] #FixME: should be set in steer file
    # Load model prediction
    pred_signals = {}
    pred_train = np.load(outputfolder+'/prediction_train'+filepostfix+'.npy').astype(np.float32)
    pred_val = np.load(outputfolder+'/prediction_val'+filepostfix+'.npy').astype(np.float32)
    pred_test = np.load(outputfolder+'/prediction_test'+filepostfix+'.npy').astype(np.float32)
    print("LOAD PREIDCT: pred_train.shape",pred_train.shape)
    mse_signals = {}
    mse_train = np.load(outputfolder+'/mse_train'+filepostfix+'.npy').astype(np.float32)
    mse_val = np.load(outputfolder+'/mse_val'+filepostfix+'.npy').astype(np.float32)
    mse_test = np.load(outputfolder+'/mse_test'+filepostfix+'.npy').astype(np.float32)

    for i in range(len(signal_identifiers)):
        pred_signals[i] = np.load(outputfolder+'/prediction_'+signal_identifiers[i]+''+filepostfix+'.npy').astype(np.float32)
        mse_signals[i] = np.load(outputfolder+'/mse_'+signal_identifiers[i]+''+filepostfix+'.npy').astype(np.float32)

    return pred_train, pred_test, pred_val, pred_signals, mse_train, mse_test, mse_val, mse_signals




def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def roc_curve_own(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True):
    # Copied from https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/metrics/ranking.py#L535
    # Extended by purity-part

    fps, tps, thresholds = binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    # purity!
   # prt = 0
    #if((tps+fps).all()>0):
#    prt = tps/(tps+fps)
    prt = np.ones(tps.shape)
    np.divide(tps,tps+fps, out=prt, where=(tps+fps)!=0)
    return fpr, tpr, thresholds, prt


def get_fpr_tpr_thr_auc(parameters, pred_val, labels_val, weights_val):

    eqweight = parameters['eqweight']
    FalsePositiveRates = {}
    TruePositiveRates = {}
    Thresholds = {}
    SignalPuritys = {}
    aucs = {}

    for i in range(labels_val.shape[1]):
        FalsePositiveRates[i], TruePositiveRates[i], Thresholds[i], SignalPuritys[i] = roc_curve_own(labels_val[:,i], pred_val[:,i], sample_weight=weights_val)
        aucs[i] = np.trapz(TruePositiveRates[i], FalsePositiveRates[i])

    return (FalsePositiveRates, TruePositiveRates, Thresholds, aucs, SignalPuritys)

def get_cut_efficiencies(parameters, predictions, thresholds, weights):
    effs_list = []
    indices = []
    length = thresholds.shape[0]
    stepsize = length/10000

    for i in range(thresholds.shape[0]):
        if i%(stepsize)==0:
            effs_list.append(weights[predictions > thresholds[i]].sum())
            indices.append(i)
    sumweights = weights.sum()
    effs = np.array(effs_list)
    effs /= sumweights
    return np.array(effs), indices



def get_data_dictionaries(parameters, eventweights_train, sample_weights_train, pred_train, labels_train, mse_train, eventweights_val, sample_weights_val, pred_val, labels_val, mse_val, eventweights_test, sample_weights_test, pred_test, labels_test, mse_test):
    print("get_dict: pred_train.shape[1]",pred_train.shape[1])
    classes = parameters['classes']
    eqweight = parameters['eqweight']
    pred_trains = {}
    pred_vals = {}
    pred_tests = {}
    mse_trains = {}
    mse_vals = {}
    mse_tests = {}

    weights_trains = {}
    weights_vals = {}
    weights_tests = {}
    normweights_trains = {}
    normweights_vals = {}
    normweights_tests = {}
    lumiweights_trains = {}
    lumiweights_vals = {}
    lumiweights_tests = {}

    for cl in classes.keys():
        pred_trains_thistrueclass = {}
        pred_vals_thistrueclass = {}
        pred_tests_thistrueclass = {}
        mse_trains_thistrueclass = {}
        mse_vals_thistrueclass = {}
        mse_tests_thistrueclass = {}

        weights_trains_thistrueclass = {}
        weights_vals_thistrueclass = {}
        weights_tests_thistrueclass = {}
        normweights_trains_thistrueclass = {}
        normweights_vals_thistrueclass = {}
        normweights_tests_thistrueclass = {}
        lumiweights_trains_thistrueclass = {}
        lumiweights_vals_thistrueclass = {}
        lumiweights_tests_thistrueclass = {}
        #for node in classes.keys():
        for node in range(pred_train.shape[1]):#FixME: use here names of input variables
            if not eqweight:
                weights_trains_thistrueclass[node] = eventweights_train[labels_train[:,cl] == 1]
                weights_vals_thistrueclass[node] = eventweights_val[labels_val[:,cl] == 1]
                weights_tests_thistrueclass[node] = eventweights_test[labels_test[:,cl] == 1]
            else:
                weights_trains_thistrueclass[node] = sample_weights_train[labels_train[:,cl] == 1]
                weights_vals_thistrueclass[node]   = sample_weights_val[labels_val[:,cl] == 1]
                weights_tests_thistrueclass[node]   = sample_weights_test[labels_test[:,cl] == 1]
            pred_trains_thistrueclass[node] = pred_train[:,node][labels_train[:,cl] == 1]
            pred_vals_thistrueclass[node] = pred_val[:,node][labels_val[:,cl] == 1]
            pred_tests_thistrueclass[node] = pred_test[:,node][labels_test[:,cl] == 1]

            mse_trains_thistrueclass[node] = mse_train[:,node][labels_train[:,cl] == 1]
            mse_vals_thistrueclass[node] = mse_val[:,node][labels_val[:,cl] == 1]
            mse_tests_thistrueclass[node] = mse_test[:,node][labels_test[:,cl] == 1]

            lumiweights_trains_thistrueclass[node] = eventweights_train[labels_train[:,cl] == 1]
            lumiweights_vals_thistrueclass[node] = eventweights_val[labels_val[:,cl] == 1]
            lumiweights_tests_thistrueclass[node] = eventweights_test[labels_test[:,cl] == 1]
            sum_train = weights_trains_thistrueclass[node].sum()
            sum_val   = weights_vals_thistrueclass[node].sum()
            sum_test   = weights_tests_thistrueclass[node].sum()
            normweights_trains_thistrueclass[node] = np.array([1./sum_train for j in range(weights_trains_thistrueclass[node].shape[0])])
            normweights_vals_thistrueclass[node]   = np.array([1./sum_val for j in range(weights_vals_thistrueclass[node].shape[0])])
            normweights_tests_thistrueclass[node]   = np.array([1./sum_test for j in range(weights_tests_thistrueclass[node].shape[0])])

        pred_trains[cl] = pred_trains_thistrueclass
        pred_vals[cl] = pred_vals_thistrueclass
        pred_tests[cl] = pred_tests_thistrueclass

        mse_trains[cl] = mse_trains_thistrueclass
        mse_vals[cl] = mse_vals_thistrueclass
        mse_tests[cl] = mse_tests_thistrueclass

        weights_trains[cl] = weights_trains_thistrueclass
        weights_vals[cl] = weights_vals_thistrueclass
        weights_tests[cl] = weights_tests_thistrueclass
        normweights_trains[cl] = normweights_trains_thistrueclass
        normweights_vals[cl] = normweights_vals_thistrueclass
        normweights_tests[cl] = normweights_tests_thistrueclass
        lumiweights_trains[cl] = lumiweights_trains_thistrueclass
        lumiweights_vals[cl] = lumiweights_vals_thistrueclass
        lumiweights_tests[cl] = lumiweights_tests_thistrueclass

    return pred_trains, mse_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, mse_vals, weights_vals, normweights_vals, lumiweights_vals,  pred_tests, mse_tests, weights_tests, normweights_tests, lumiweights_tests


#create dictinaries for data (train, test or val) sample
def get_data_dictionaries_onesample(parameters, eventweights, sample_weights, pred, labels):
    classes = parameters['classes']
    eqweight = parameters['eqweight']
    pred_d = {}
    weights_d = {}   
    normweights_d = {}
    lumiweights_d = {}
   

    for cl in classes.keys():
#        print("cl = ",cl)
        pred_thistrueclass = {}
        weights_thistrueclass = {}
        normweights_thistrueclass = {}
        lumiweights_thistrueclass = {}
        for node in classes.keys():
            #print "node = ",node
            # if(node == 3): 
            #     print "labels[:,cl] = ",labels[:,cl]
            #     print "pred[:,node][labels[:,cl] == 1] = ",pred[:,node][labels[:,cl] == 1]
            if not eqweight:
                weights_thistrueclass[node] = eventweights[labels[:,cl] == 1]
            else:
                weights_thistrueclass[node] = sample_weights[labels[:,cl] == 1]

            pred_thistrueclass[node] = pred[:,node][labels[:,cl] == 1]
            lumiweights_thistrueclass[node] = eventweights[labels[:,cl] == 1]
            sum_train = weights_thistrueclass[node].sum()
            normweights_thistrueclass[node] = np.array([1./sum_train for j in range(weights_thistrueclass[node].shape[0])])

        pred_d[cl] = pred_thistrueclass
        weights_d[cl] = weights_thistrueclass
        normweights_d[cl] = normweights_thistrueclass
        lumiweights_d[cl] = lumiweights_thistrueclass

    #print("pred_d[0]",pred_d[0])
    return pred_d, weights_d, normweights_d, lumiweights_d


def get_indices_wrong_predictions(labels, preds):
    mask = []
    for i in range(labels.shape[0]):
        label = -1
        predclass = -1
        maxpred = -1
        for j in range(labels.shape[1]):
            if labels[i,j] == 1: label = j
            if maxpred < preds[i,j]:
                maxpred = preds[i,j]
                predclass = j
        if label != predclass:
            mask.append(i)

    return mask


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def conf_matrix(labels, predictions, weights):
    # will return a list of lists, shape=NxN
    # each list is for one true class
    if labels.shape != predictions.shape:
        raise ValueError('Labels and predictions do not have the same shape.')
    if labels.shape[0] != weights.shape[0]:
        raise ValueError('Labels and weights do not have the same length (.shape[0]).')

    # initialize confusion matrix
    matrix = np.zeros((labels.shape[1], labels.shape[1]))

    # format inputs
    for i in range(labels.shape[0]):
        label = -1
        predclass = -1
        maxpred = -1
        for j in range(labels.shape[1]):
            if labels[i,j] == 1: label = j
            if maxpred < predictions[i,j]:
                maxpred = predictions[i,j]
                predclass = j
        if label == -1: raise ValueError('For this event, the labels of all classes are 0, so the event doesn\'t have a class?')
        matrix[label,predclass] = matrix[label,predclass] + weights[i]

    return matrix

def log_model_performance(parameters, model_history, outputfolder):
    loss_train = model_history['loss']
    loss_val = model_history['val_loss']
    #Fit Losses
    x, fitx, fitfunc, pars_train = fit_loss(model_history['loss'])
    x, fitx, fitfunc, pars_val = fit_loss(model_history['val_loss'])

    #build difference
    pars = [pars_train, pars_val]
    # def func_diff(fitx, mypars):
    #     return fitfunc(fitx, *mypars[0]) - fitfunc(fitx, *mypars[1])
    def func_diff(x, fitfunc, pars):
        return fitfunc(x, *(pars[0])) - fitfunc(x, *(pars[1]))

    #Find intersection between losses - aka roots of (f_train - f_val)
    # sol = opt.root_scalar(func_diff, (pars))
    roots = fsolve(func_diff, [100.], (fitfunc, pars))
    #print roots

    # Get closest integer numbers
    for root in roots:
        x_low = int(math.floor(root))
        x_high = int(math.ceil(root))

    # compare losses at these values, pick the one where difference of losses is smallest
        diff_xlow = math.fabs(fitfunc(x_low, *pars_train) - fitfunc(x_low, *pars_val))
        diff_xhigh = math.fabs(fitfunc(x_high, *pars_train) - fitfunc(x_high, *pars_val))
        bestloss_val = fitfunc(x_low, *pars_val)
        bestx = x_low
        if diff_xhigh < diff_xlow:
            bestloss_val = fitfunc(x_high, *pars_val)
            bestx = x_high
        if root < 10:
            bestloss_val = 999999
            bestx = 999999
        print("Validation loss in point of closest approach: %f, reached after %i epochs" % (bestloss_val, bestx))


    acc_train = model_history['categorical_accuracy']
    acc_val = model_history['val_categorical_accuracy']
    tag = dict_to_str(parameters)
    with open(outputfolder+'/ModelPerformance.txt', 'w') as f:
        f.write('\n\n====================\n')
        f.write('Tag: %s\n\n' % (tag))
        #f.write('Minimum validation loss reached after %i epochs\n' % (loss_val.index(min(loss_val))))
        f.write('Validation loss in point of closest approach: %2.3f, reached after %i epochs\n' % (bestloss_val, bestx))
        f.write('Performance: training loss (min, final) -- validation loss (min, final) -- training acc (min, final) -- validation acc (min, final)\n')
        f.write('                         ({0:2.3f}, {1:2.3f}) --               ({2:2.3f}, {3:2.3f}) --            ({4:1.3f}, {5:1.3f}) --              ({6:1.3f}, {7:1.3f})\n'.format(min(loss_train), loss_train[len(loss_train)-1], min(loss_val), loss_val[len(loss_val)-1], min(acc_train), acc_train[len(acc_train)-1], min(acc_val), acc_val[len(acc_val)-1]))

def fit_loss(losslist, maxfev=50000):

    def fitfunc(x, a, b, c, d, e):
        return a + b/x + c*x + d*x*x + e/x/x + d*x*x*x
#    print("len(losslist)",len(losslist))
    x = range(len(losslist)+1)
    x = x[1:]
    x = np.array(x)
    #fitx = x[9:]
    #fity = losslist[9:]
    fitx = x[2:]
    fity = losslist[2:]

    postfitpars, cov = opt.curve_fit(fitfunc, fitx, fity, maxfev=maxfev,method='dogbox')
#    print("postfitpars: ",postfitpars)
#    postfitpars, cov = opt.curve_fit(fitfunc, fitx, fity) #TEST

    return x, fitx, fitfunc, postfitpars







class lr_reduce_on_plateau:
    """Class for reducing lr when given validation loss is not improving
    any longer.

    Args:
        lr:       inital learning rate (float)
        patience: number of times loss doesn't improve
            until reduce on plateau is triggered
        fraction: determines lr update after triggering reduce on plateau
            lr = lr*fraction (should be float between 0 and 1)
        delte_improv: loss value has to be smaller than best loss value
            minus delta_improv to count as an improvment

    Example for usage:
        lr = lr_reduce_on_plateau(
            lr=0.1, patience=3, fraction=0.1)
        for i in n_epochs:
            ...
            sess.run(train_op, feed_dict={lr: lr.lr})
            lr.update_on_plateau(val_loss)
    """

    def __init__(self, lr, patience, delta_improv, fraction):
        self.patience = patience
        self.fraction = fraction
        self.delta_improv = delta_improv
        self.lr = lr # initial lr
        self.best_loss = None
        self.counter = 0

    def _update(self, x):
        if self.best_loss == None:
            self.best_loss = x
        elif (self.best_loss - x) > self.delta_improv:
            self.counter = 0.
        else:
            self.counter +=1
        if x < self.best_loss:
            self.best_loss = x

    def _lr_reduce(self):
        self.lr = self.lr*self.fraction

    def update_on_plateau(self, x):
        self._update(x)
        if self.counter >= self.patience:
            self._lr_reduce()
            self.counter = 0. # reset counter
            print("Reduce on Plateau triggered! (patience: {:}, fraction: {:}, ".format(
                self.patience, self.fraction) +\
                "delta_improv: {:}, new lr: {:})".format(self.delta_improv, self.lr))



class check_early_stopping:
    """Class for Early Stopping. Early Stopping stopps training
    after validation loss hasen't improved for patience-time.
    Look into example for usgae (check mehtod just gives Boolen output).

    Args:
        patience: number of times loss doesn't improve
            until Early Stopping is triggered
        delte_improv: loss value has to be smaller than best loss value
            minus delta_improv to count as an improvment

    Example for usage:
        check = check_early_stopping(
            patience=3, fraction=0.1)
        for i in n_epochs:
            ...
            sess.run(train_op)
            if check.update_and_check(val_loss):
                break
    """

    def __init__(self, patience, delta_improv=0.):
        self.patience = patience
        self.delta_improv = delta_improv
        self.best_loss = None
        self.counter = 0

    def _update(self, x):
        if self.best_loss == None:
            self.best_loss = x
        elif (self.best_loss - x) > self.delta_improv:
            self.counter = 0.
        else:
            self.counter +=1
        if x < self.best_loss:
            self.best_loss = x

    def update_and_check(self, x):
        self._update(x)
        if self.counter >= self.patience:
            print("Early Stopping triggered! (patience: {:}, delta_improv: {:})".format(
                self.patience, self.delta_improv))
            return True
        else:
            return False
