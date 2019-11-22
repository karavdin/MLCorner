#!/usr/bin/env python

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
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

warnings.simplefilter(action='ignore', category=RuntimeWarning)

import numpy as np
from numpy import inf
import keras
import matplotlib
matplotlib.use('Agg')
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
import math
import pickle
from Training import *
from Plotting import *
from GetInputs import *
#from RankNetworks import *
from PredictExternal import *
from functions import *
#from TrainModelOnPredictions import *
#from TrainSecondNetwork import *
#from TrainThirdNetwork import *
#from ExportModel import *


#variations = ['NOMINAL','JEC_up','JEC_down','JER_up','JER_down']
variations = ['NOMINAL']
merged_str = 'Merged'

#parameters related to input data
parameters_input = {
    'runonfraction': 1.00, #100% of events
    #'runonfraction': 0.01, #1% of events
    'eqweight':False,
    'preprocess': 'MinMaxScaler',    
    'inputdir': 'input/TStarTstar_MLtests_numpy/',#general path to inputs with systematic variations
    'inputsubdir': 'MLInput/', #path to input files: inputdir + systvar + inputsubdir
    'prepreprocess': 'RAW' #for inputs with systematics don't do preprocessing before merging all inputs on one,     #FixME: add prepreprocessing in case one does not need to merge inputs
}



#parameters related to NN
parameters_network = {
    'layers':[128, 32, 8, 32, 128],
    'batchsize': 512,
    #'classes':{0: ['TstarTstarToTgluonTgluon_M-700'], 1: ['TTbar','WJets','ST','DYJets']},
    #'classes':{0: ['TstarTstarToTgluonTgluon_M-700'], 1:['TstarTstarToTgluonTgluon_M-800']},
    'classes':{0: ['TTbar','ST']},
    #'regmethod': 'dropout',
    'regmethod': '',
    'regrate':0.10000,
    'batchnorm': False,
    'epochs':4000, #TEST
    'learningrate': 0.00050,
    #'sigma': 1.0, #sigma for Gaussian prior (BNN only)
}

#at the moment all the following scripts expect parameters in one dictionary
parameters = parameters_input
parameters.update(parameters_network)


tag = dict_to_str(parameters)
classtag = get_classes_tag(parameters)

#prepare inputs for each variation
for ivars in range(len(variations)):
    merged_str = merged_str+'__'+variations[ivars]
    parameters['systvar'] = variations[ivars]
    inputfolder = parameters['inputdir']+parameters['inputsubdir']+parameters['systvar']+'/'+parameters['prepreprocess']+'/'+ classtag
    GetInputs(parameters)
    PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder='Plots/'+parameters['prepreprocess']+'/InputDistributions/'+parameters['systvar']+'/' + classtag)
    
MixInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, variations=variations, filepostfix='')
SplitInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
FitPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')

ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='train')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='test')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='val')

ApplySignalPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='') 

inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag
outputfolder='output/'+parameters['preprocess']+'/'+merged_str+'/' + classtag+'/DNN_'+tag
plotfolder = 'Plots/'+parameters['preprocess']
PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder=plotfolder+'/InputDistributions/'+merged_str+'/' + classtag)


#Deterministic Autoencoder
TrainAutoencoderNetwork(parameters, inputfolder=inputfolder, outputfolder=outputfolder)
PredictExternalAutoencoder(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='')
#Plot validation results and store model for usage in UHH2, etc
PlotAutoencoderPerformance(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='', plotfolder=plotfolder+'/Output/'+merged_str+'/'+'/DNN_'+tag, use_best_model=False, usesignals=[0,1,2])

#TrainBayesianNetwork(parameters, inputfolder=inputfolder, outputfolder=outputfolder)
# PredictExternalBayesianNetwork(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='',nsamples=100)
# #Plot validation results and store model for usage in UHH2, etc
# PlotBayesianPerformance(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='', plotfolder=plotfolder+'/Output/'+merged_str+'/'+'/BNN_'+tag, use_best_model=False, usesignals=[2,4])

