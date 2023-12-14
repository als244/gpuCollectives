#!/bin/bash
#SBATCH --job-name=test         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default) 
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jbyoon@princeton.edu
#SBATCH --gres=gpu:2            # Number of Requested GPUs # Change the number of GPUs to equal the argument

# Arguments: Program being executed number of GPUs
./SingProcMultDevAllReduce 2