#Login TACC cluster at UT Austin
ssh -l usesrID stampede.tacc.xsede.org

## using CUDA
module load cuda

nvcc -arch=compute_35 -code=sm_35

## Run jobs on Stampede

sbatch job-stampede-gpu 


## SIMD is not efficient for branches
![]
