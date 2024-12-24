#!/bin/bash 



for i in {0..3}
do
  export SLURM_ARRAY_TASK_ID=$i
  ./poly3_sweep.sh 


done
