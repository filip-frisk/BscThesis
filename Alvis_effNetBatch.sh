#!/bin/bash
#SBATCH -A SNIC2020-33-67 -p alvis
#SBATCH -n 1
#SBATCH --time=75:00:00
#SBATCH --gpus-per-node=V100:1
#SBATCH -J filip_albert_4th_iteration_allfilters_with_loss_and_acc_curves_saved

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
deactivate

# Load the ALVIS module environment suitable for the job
# module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 PyTorch/1.7.1-Python-3.7.4

# Loads the correct virtual environment
source ../clusterEnv/bin/activate

# Prints some statistics from the run
start = `date`
echo "Starting at `date`"
echo "Job Name - EffNet"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Current working directory is `pwd`"

# cat $0

export PYTHONPATH=$PYTHONPATH:~/clusterEnv/lib/python3.6/site-packages/

# runs 4 model trainings, 1 for each filter
python3 main.py --epochs 100 --pretrained --batch-size 32 --wd 5e-5 --lr 0.01 --image_size 224 --momentum 0.9 --advprop -val -e --filter '' --outdest 'Iteration_5/Output_No_filter' -p 100
echo "--------------------------- NEXT TRAINING SESSION ---------------------------"
python3 main.py --epochs 100 --pretrained --batch-size 32 --wd 5e-5 --lr 0.01 --image_size 224 --momentum 0.9 --advprop -val -e --filter '_SCD' --outdest 'Iteration_5/Output_SCD' -p 100
echo "--------------------------- NEXT TRAINING SESSION ---------------------------"
python3 main.py --epochs 100 --pretrained --batch-size 32 --wd 5e-5 --lr 0.01 --image_size 224 --momentum 0.9 --advprop -val -e --filter '_Macenko' --outdest 'Iteration_5/Output_Macenko' -p 100
echo "--------------------------- NEXT TRAINING SESSION ---------------------------"
python3 main.py --epochs 100 --pretrained --batch-size 32 --wd 5e-5 --lr 0.01 --image_size 224 --momentum 0.9 --advprop -val -e --filter '_Reinhard' --outdest 'Iteration_5/Output_Reinhard' -p 100
echo "--------------------------- TRAINING SESSION DONE ---------------------------"

#python3 main.py --epochs 25 --pretrained --batch-size 32 --wd 5e-5 --lr 0.01 --image_size 32 --momentum 0.9 --advprop -val -e --outdest 'Output_Test' -p 110

echo "Program finished with exit code $? at: `date`"

echo "done"
echo "*** på dagen"
