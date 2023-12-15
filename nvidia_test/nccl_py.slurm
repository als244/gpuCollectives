#!/bin/bash
#SBATCH --job-name=test         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=2G         # memory per cpu-core (4G is default) 
#SBATCH --time=00:03:00          # total run time limit (HH:MM:SS)
#### CHANGE THE MAIL ADDRESS
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=jbyoon@princeton.edu
#SBATCH --gres=gpu:4

# Arguments: Program being executed number of GPUs Buffer Size
#./SingProcMultDevAllReduce 2 33554432 (4.194MB) => Default
#./SingProcMultDevAllReduce 2 8589934592 (1.073GB)
# ./SingProcMultDevAllReduce 2 4294967296 (536.8MB)
# ./SingProcMultDevAllReduce 2 2147483648 (268.4MB)
# ./SingProcMultDevAllReduce 2 1073741824 (134.2MB)
# ./SingProcMultDevAllReduce 2 536870912 (67.1MB)
#./SingProcMultDevAllReduce 2 268435456 (33.55MB)
#./SingProcMultDevAllReduce 2 134217728 (16.775MB)
#./SingProcMultDevAllReduce 2 67108864 (8.3875 MB)
#./SingProcMultDevAllReduce 2 33554432 (4.194MB)
#./SingProcMultDevAllReduce 2 16777216 (2.097MB)
#./SingProcMultDevAllReduce 2 8388608 (1.048MB)
#./SingProcMultDevAllReduce 2 4194304 (524.288KB)
#./SingProcMultDevAllReduce 2 2097152 (262.144KB)
#./SingProcMultDevAllReduce 2 1048576 (131.072KB)
#./SingProcMultDevAllReduce 2 524288 (65.536KB)
#./SingProcMultDevAllReduce 2 262144 (32.768KB)
#./SingProcMultDevAllReduce 2 131072 (16.384KB)
#./SingProcMultDevAllReduce 2 65536 (8.192KB)
#./SingProcMultDevAllReduce 2 32768 (4.096KB)
#./SingProcMultDevAllReduce 2 16384 (2.048KB)
#./SingProcMultDevAllReduce 2 8192 (1.024KB)
#./SingProcMultDevAllReduce 2 4096 (512B)
#./SingProcMultDevAllReduce 2 2048 (256B)
#./SingProcMultDevAllReduce 2 1024 (128B)
# source ~/.nccl.conf
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
./SingProcMultDevReduce 4 8589934592
