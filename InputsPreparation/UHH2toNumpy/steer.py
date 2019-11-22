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
#procnames = ['TstarTstarToTgluonTgluon_M-700','TstarTstarToTgluonTgluon_M-800', 'WJets', 'DYJets']
procnames = ['TTbar', 'ST', 'TstarTstarToTgluonTgluon_All', 'TstarTstarToTgammaTgluon_M-1500', 'TstarTstarToTgammaTgluon_M-1600', 'TstarTstarToTgluonTgluon_M-700','TstarTstarToTgluonTgluon_M-800','TstarTstarToTgluonTgluon_M-900','TstarTstarToTgluonTgluon_M-1100','TstarTstarToTgluonTgluon_M-1200','TstarTstarToTgluonTgluon_M-1300', 'TstarTstarToTgluonTgluon_M-1500']
#name of branches to be skipped in conversion
## exact names
unwanted_exact_tags = ['eventweight']
## partial names to exclude common set of variables at once
unwanted_tags = ['TTbarReconstruction','TTbarReconstruction_','m_','TstarTstar_tgtg_best']

syst_vars = ['NOMINAL']
for syst_var in syst_vars:
    print'--- Convert inputs for ',syst_var,' variation ---'
    ModuleRunner.ReadoutMLVariables(procnames=procnames,unwanted_tags=unwanted_tags, unwanted_exact_tags=unwanted_exact_tags,syst_var=syst_var)

