#Login TACC cluster at UT Austin
ssh -l usesrID stampede.tacc.xsede.org

## using CUDA
module load cuda

nvcc -arch=compute_35 -code=sm_35

## Run jobs on Stampede

sbatch job-stampede-gpu 


## SIMD is not efficient for branches
![](gpubranch.png)

## hiding stall with mutiple contexts

e.g. We have multiple group of registers/buffers for store different context.

![](hiding_stall.png)
