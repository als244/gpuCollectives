import subprocess
import os

def edit_slurm_script(file_path, num_gpus=None, memory_input=None, program_input=None):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if num_gpus and line.startswith('#SBATCH --gres=gpu:'):
            new_lines.append(f'#SBATCH --gres=gpu:{num_gpus}\n')
        elif memory_input and line.startswith('./'):
            new_lines.append(f'./{program_input} {num_gpus} {memory_input}\n')
        else:
            new_lines.append(line)

    with open(file_path, 'w') as file:
        file.writelines(new_lines)

def execute_slurm_script(file_path):
    subprocess.run(['sbatch', file_path])

def change_environment_vars(file_path, proto=None, threads=None):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if proto and line.startswith('NCCL_PROTO'):
            new_lines.append(f'NCCL_PROTO={proto}')
        elif threads and line.startswith('NCCL_THREADS'):
            new_lines.append(f'NCCL_THREADS={threads}')
        else:
            new_lines.append(line)

# Example usage
slurm_script_path = 'nccl_py.slurm'
environment_vars_path = '/home/jbyoon/.nccl.conf'
gpu_inputs = [2] # , 4, 6]
memory_inputs = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 
2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 
536870912, 1073741824, 2147483648, 4294967296, 8589934592]
#  [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 
# 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 
# 536870912, 1073741824, 2147483648, 4294967296, 8589934592]

program_inputs = ['spmd_ar', 'spmd_b', 'spmd_r']
proto_inputs = ['Simple', 'LL', 'LL128']
thread_inputs = [64, 128, 256, 512]


for gpu in gpu_inputs:
    for mem in memory_inputs:
        for prog in program_inputs:
            # edit_slurm_script(slurm_script_path, num_gpus=gpu, memory_input=mem, program_input=prog)
            # change_environment_vars(environment_vars_path, proto='Simple', threads=512)
            # execute_slurm_script(slurm_script_path)
            cd = f'./{prog} {gpu} {mem}\n'
            os.system(cd)



