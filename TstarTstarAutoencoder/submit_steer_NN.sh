#!/bin/bash
#SBATCH --partition=cms-uhh,cms,allgpu
#SBATCH --time=1-00:00:00                           # Maximum time requested
#SBATCH --constraint=GPU
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --workdir   /beegfs/desy/user/karavdia/MLCorner/TstarTstarAutoencoder/workdir/output    # directory must already exist!
#SBATCH --job-name  steer
#SBATCH --output    steer-%N-%j.out            # File to which STDOUT will be written
#SBATCH --error     steer-%N-%j.err            # File to which STDERR will be written
#SBATCH --mail-type ALL                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user anastasia.karavdina@desy.de          # Email to which notifications will be sent. It defaults to <userid@mail.desy.de> if none is set.

source ~/.setenv_py3_7_tensoflow2_0
echo $PYTHONPATH
cd ~/TstarTstarAutoencoder/
./steer_NN.py
