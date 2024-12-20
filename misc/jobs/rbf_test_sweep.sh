#!/bin/bash 

#SBATCH --job-name=1v1_rbf
#SBATCH --partition=rome
#SBATCH --output=test_rbf.stdout
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=4:00:00
#SBATCH --array=0-9


module load gcc/13.2.0-iqpfkya cmake/3.27.9-nmh6tto eigen/3.4.0-titj7ys 

source /home/c/cioakeim/aocl/5.0.0/aocc/amd-libs.cfg

sigma_list=("1e-3" "5e-3" "1e-2" "2.5e-2" "5e-2" "7.5e-2" "1e-1" "5e-1" "1" "5")

store_path="/home/c/cioakeim/nns/SVMs/RBF/test_$SLURM_ARRAY_TASK_ID"
dataset_path="/home/c/cioakeim/nns/cifar-10-batches-bin"
class_1_id="0"
class_2_id="1"
training_size="50000"
test_size="10000"
C_list="1e-3,5e-3,1e-2,1e-1,1,5,10,50,100,1000"
kernel_type="RBF"
sigma="${sigma_list[$SLURM_ARRAY_TASK_ID]}"


project_dir="/home/c/cioakeim/nns/neural-networks-hw2"

cd "$project_dir"
mkdir -p build
cd build
#cmake -DOSQP_MANUAL_LINK=ON -DBLAS_LIBRARIES="/home/c/cioakeim/aocl/5.0.0/aocc/lib/libblis.so" ..
#make


./test2ClassSVM "$store_path" "$dataset_path" "$training_size" "$test_size" "$class_1_id" "$class_2_id" "$C_list" "$kernel_type" "$sigma"


