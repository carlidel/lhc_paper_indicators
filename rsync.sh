#!/bin/bash

rsync -av --exclude='data' /home/camontan/Insync/carlo.montanari3@studio.unibo.it/OneDrive_Biz/projects/lhc_paper_indicators abp:/home/HPC/camontan

rsync -av --exclude='data' /home/camontan/Insync/carlo.montanari3@studio.unibo.it/OneDrive_Biz/projects/lhc_paper_indicators lxplus:/afs/cern.ch/work/c/camontan/public

# opposite

rsync -av --exclude='data' abp:/home/HPC/camontan/lhc_paper_indicators /home/camontan/Insync/carlo.montanari3@studio.unibo.it/OneDrive_Biz/projects/lhc_paper_indicators

rsync -av --exclude='data' lxplus:/afs/cern.ch/work/c/camontan/public/lhc_paper_indicators /home/camontan/Insync/carlo.montanari3@studio.unibo.it/OneDrive_Biz/projects/lhc_paper_indicators