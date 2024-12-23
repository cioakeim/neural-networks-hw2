#!/bin/bash 

#SBATCH --job-name=mult_lin
#SBATCH --partition=rome
#SBATCH --output=test_mult_lin.stdout
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=7:00:00
#SBATCH --array=0


module load gcc/13.2.0-iqpfkya cmake/3.27.9-nmh6tto eigen/3.4.0-titj7ys 

MY_HOME="/home/c/cioakeim"

source $MY_HOME/intel/oneapi/mkl/latest/env/vars.sh lp64


store_path="/home/c/cioakeim/nns/SVMs/MULT/LINEAR/test_$SLURM_ARRAY_TASK_ID"
dataset_path="/home/c/cioakeim/nns/cifar-10-batches-bin"
training_size="50000"
test_size="10000"
C_list="1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1e3,1e4"
kernel_type="LINEAR"


project_dir="/home/c/cioakeim/nns/neural-networks-hw2"

cd "$project_dir"
mkdir -p build
cd build
cmake -DMY_HOME_DIR="$MY_HOME" -DMKL_INTERFACE_FULL=intel_lp64 ..
make


./testMultiSVM "$store_path" "$dataset_path" "$training_size" "$test_size" "$C_list" "$kernel_type"

