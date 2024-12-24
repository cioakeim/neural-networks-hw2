#!/bin/bash 

#SBATCH --job-name=1v1_linear
#SBATCH --partition=rome
#SBATCH --output=test_linear.stdout
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=0:30:00

module load gcc/13.2.0-iqpfkya cmake/3.27.9-nmh6tto eigen/3.4.0-titj7ys 

MY_HOME="/home/c/cioakeim"
MY_HOME="/home/chris"

project_dir="/home/c/cioakeim/nns/neural-networks-hw2"
project_dir="/home/chris/Documents/programms/nns/neural-networks-hw2"

source $MY_HOME/aocl/5.0.0/aocc/amd-libs.cfg
source $MY_HOME/intel/oneapi/mkl/latest/env/vars.sh lp64


store_path="$project_dir/data/SVMs/LINEAR"
dataset_path="$project_dir/data/cifar-10-batches-bin"
class_1_id="0"
class_2_id="1"
training_size="50000"
test_size="10000"
C_list="1e-3,5e-3,1e-2,1e-1,1,5,10,50,100,1000"
kernel_type="LINEAR"



cd "$project_dir"
mkdir -p build
cd build
#cmake -DMY_HOME_DIR="$MY_HOME" -DMKL_INTERFACE_FULL=intel_lp64 ..
cmake -DMKL_INTERFACE_FULL=intel_lp64 ..
make


./test2ClassSVM "$store_path" "$dataset_path" "$training_size" "$test_size" "$class_1_id" "$class_2_id" "$C_list" "$kernel_type"
