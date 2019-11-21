#!/usr/bin/env python

import subprocess
import os
import time

from python.ModuleRunner import *
from python.constants import *


"""This is macro to steer Root to Numpy conversion. """


#paths are set in constants.py
ModuleRunner = ModuleRunner(path_MLDIR, outpath)

# ---- Macro for ML inputs preparation ----
#names of the process, e.g part after uhh2.AnalysisModuleRunner. in the input file name
procnames = ['TTbar', 'ST', 'DYJets', 'WJets', 'TstarTstarToTgluonTgluon', 'TstarTstarToTgammaTgluon_M-1500', 'TstarTstarToTgammaTgluon_M-1600', 'TstarTstarToTgluonTgluon_M-700',
             'TstarTstarToTgluonTgluon_M-800','TstarTstarToTgluonTgluon_M-900','TstarTstarToTgluonTgluon_M-1100','TstarTstarToTgluonTgluon_M-1200','TstarTstarToTgluonTgluon_M-1300',
             'TstarTstarToTgluonTgluon_M-1600']
#name of branches to be skipped in conversion
## exact names
unwanted_exact_tags = []
## partial names to exclude common set of variables at once
unwanted_tags = []

syst_vars = ['NOMINAL']
for syst_var in syst_vars:
    print'--- Convert inputs for ',syst_var,' variation ---'
    ModuleRunner.ReadoutMLVariables(procnames=procnames,unwanted_tags=unwanted_tags, unwanted_exact_tags=unwanted_exact_tags,syst_var=syst_var)

