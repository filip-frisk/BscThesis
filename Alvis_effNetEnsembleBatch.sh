#!/bin/bash
#SBATCH -A SNIC2020-33-67 -p alvis
#SBATCH -n 1
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=T4:1
#SBATCH -J filip_albert_Ensembling_All_iterations

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

python3 ensemble.py --pretrained --wd 5e-5 --lr 0.01 --image_size 224 --batch-size 32 --outdest "." --momentum 0.9 -p 100 --model_paths "Iteration_10/Output_No_filter/model_best_4.pth.tar
Iteration_8/Output_Macenko/model_best_3.pth.tar
Iteration_6/Output_Reinhard/model_best_3.pth.tar
Iteration_9/Output_SCD/model_best_3.pth.tar
Iteration_5/Output_No_filter/model_best_0.pth.tar
Iteration_5/Output_No_filter/model_best_1.pth.tar
Iteration_5/Output_No_filter/model_best_2.pth.tar
Iteration_5/Output_No_filter/model_best_3.pth.tar
Iteration_5/Output_No_filter/model_best_4.pth.tar
Iteration_5/Output_Macenko/model_best_0.pth.tar
Iteration_5/Output_Macenko/model_best_1.pth.tar
Iteration_5/Output_Macenko/model_best_2.pth.tar
Iteration_5/Output_Macenko/model_best_3.pth.tar
Iteration_5/Output_Macenko/model_best_4.pth.tar
Iteration_5/Output_Reinhard/model_best_0.pth.tar
Iteration_5/Output_Reinhard/model_best_1.pth.tar
Iteration_5/Output_Reinhard/model_best_2.pth.tar
Iteration_5/Output_Reinhard/model_best_3.pth.tar
Iteration_5/Output_Reinhard/model_best_4.pth.tar
Iteration_5/Output_SCD/model_best_0.pth.tar
Iteration_5/Output_SCD/model_best_1.pth.tar
Iteration_5/Output_SCD/model_best_2.pth.tar
Iteration_5/Output_SCD/model_best_3.pth.tar
Iteration_5/Output_SCD/model_best_4.pth.tar
Iteration_6/Output_No_filter/model_best_0.pth.tar
Iteration_6/Output_No_filter/model_best_1.pth.tar
Iteration_6/Output_No_filter/model_best_2.pth.tar
Iteration_6/Output_No_filter/model_best_3.pth.tar
Iteration_6/Output_No_filter/model_best_4.pth.tar
Iteration_6/Output_Macenko/model_best_0.pth.tar
Iteration_6/Output_Macenko/model_best_1.pth.tar
Iteration_6/Output_Macenko/model_best_2.pth.tar
Iteration_6/Output_Macenko/model_best_3.pth.tar
Iteration_6/Output_Macenko/model_best_4.pth.tar
Iteration_6/Output_Reinhard/model_best_0.pth.tar
Iteration_6/Output_Reinhard/model_best_1.pth.tar
Iteration_6/Output_Reinhard/model_best_2.pth.tar
Iteration_6/Output_Reinhard/model_best_3.pth.tar
Iteration_6/Output_Reinhard/model_best_4.pth.tar
Iteration_6/Output_SCD/model_best_0.pth.tar
Iteration_6/Output_SCD/model_best_1.pth.tar
Iteration_6/Output_SCD/model_best_2.pth.tar
Iteration_6/Output_SCD/model_best_3.pth.tar
Iteration_6/Output_SCD/model_best_4.pth.tar
Iteration_7/Output_No_filter/model_best_0.pth.tar
Iteration_7/Output_No_filter/model_best_1.pth.tar
Iteration_7/Output_No_filter/model_best_2.pth.tar
Iteration_7/Output_No_filter/model_best_3.pth.tar
Iteration_7/Output_No_filter/model_best_4.pth.tar
Iteration_7/Output_Macenko/model_best_0.pth.tar
Iteration_7/Output_Macenko/model_best_1.pth.tar
Iteration_7/Output_Macenko/model_best_2.pth.tar
Iteration_7/Output_Macenko/model_best_3.pth.tar
Iteration_7/Output_Macenko/model_best_4.pth.tar
Iteration_7/Output_Reinhard/model_best_0.pth.tar
Iteration_7/Output_Reinhard/model_best_1.pth.tar
Iteration_7/Output_Reinhard/model_best_2.pth.tar
Iteration_7/Output_Reinhard/model_best_3.pth.tar
Iteration_7/Output_Reinhard/model_best_4.pth.tar
Iteration_7/Output_SCD/model_best_0.pth.tar
Iteration_7/Output_SCD/model_best_1.pth.tar
Iteration_7/Output_SCD/model_best_2.pth.tar
Iteration_7/Output_SCD/model_best_3.pth.tar
Iteration_7/Output_SCD/model_best_4.pth.tar
Iteration_8/Output_No_filter/model_best_0.pth.tar
Iteration_8/Output_No_filter/model_best_1.pth.tar
Iteration_8/Output_No_filter/model_best_2.pth.tar
Iteration_8/Output_No_filter/model_best_3.pth.tar
Iteration_8/Output_No_filter/model_best_4.pth.tar
Iteration_8/Output_Macenko/model_best_0.pth.tar
Iteration_8/Output_Macenko/model_best_1.pth.tar
Iteration_8/Output_Macenko/model_best_2.pth.tar
Iteration_8/Output_Macenko/model_best_3.pth.tar
Iteration_8/Output_Macenko/model_best_4.pth.tar
Iteration_8/Output_Reinhard/model_best_0.pth.tar
Iteration_8/Output_Reinhard/model_best_1.pth.tar
Iteration_8/Output_Reinhard/model_best_2.pth.tar
Iteration_8/Output_Reinhard/model_best_3.pth.tar
Iteration_8/Output_Reinhard/model_best_4.pth.tar
Iteration_8/Output_SCD/model_best_0.pth.tar
Iteration_8/Output_SCD/model_best_1.pth.tar
Iteration_8/Output_SCD/model_best_2.pth.tar
Iteration_8/Output_SCD/model_best_3.pth.tar
Iteration_8/Output_SCD/model_best_4.pth.tar
Iteration_9/Output_No_filter/model_best_0.pth.tar
Iteration_9/Output_No_filter/model_best_1.pth.tar
Iteration_9/Output_No_filter/model_best_2.pth.tar
Iteration_9/Output_No_filter/model_best_3.pth.tar
Iteration_9/Output_No_filter/model_best_4.pth.tar
Iteration_9/Output_Macenko/model_best_0.pth.tar
Iteration_9/Output_Macenko/model_best_1.pth.tar
Iteration_9/Output_Macenko/model_best_2.pth.tar
Iteration_9/Output_Macenko/model_best_3.pth.tar
Iteration_9/Output_Macenko/model_best_4.pth.tar
Iteration_9/Output_Reinhard/model_best_0.pth.tar
Iteration_9/Output_Reinhard/model_best_1.pth.tar
Iteration_9/Output_Reinhard/model_best_2.pth.tar
Iteration_9/Output_Reinhard/model_best_3.pth.tar
Iteration_9/Output_Reinhard/model_best_4.pth.tar
Iteration_9/Output_SCD/model_best_0.pth.tar
Iteration_9/Output_SCD/model_best_1.pth.tar
Iteration_9/Output_SCD/model_best_2.pth.tar
Iteration_9/Output_SCD/model_best_3.pth.tar
Iteration_9/Output_SCD/model_best_4.pth.tar
Iteration_10/Output_No_filter/model_best_0.pth.tar
Iteration_10/Output_No_filter/model_best_1.pth.tar
Iteration_10/Output_No_filter/model_best_2.pth.tar
Iteration_10/Output_No_filter/model_best_3.pth.tar
Iteration_10/Output_No_filter/model_best_4.pth.tar
Iteration_10/Output_Macenko/model_best_0.pth.tar
Iteration_10/Output_Macenko/model_best_1.pth.tar
Iteration_10/Output_Macenko/model_best_2.pth.tar
Iteration_10/Output_Macenko/model_best_3.pth.tar
Iteration_10/Output_Macenko/model_best_4.pth.tar
Iteration_10/Output_Reinhard/model_best_0.pth.tar
Iteration_10/Output_Reinhard/model_best_1.pth.tar
Iteration_10/Output_Reinhard/model_best_2.pth.tar
Iteration_10/Output_Reinhard/model_best_3.pth.tar
Iteration_10/Output_Reinhard/model_best_4.pth.tar
Iteration_10/Output_SCD/model_best_0.pth.tar
Iteration_10/Output_SCD/model_best_1.pth.tar
Iteration_10/Output_SCD/model_best_2.pth.tar
Iteration_10/Output_SCD/model_best_3.pth.tar
Iteration_10/Output_SCD/model_best_4.pth.tar"


echo "Program finished with exit code $? at: `date`"

echo "Marco scare"
