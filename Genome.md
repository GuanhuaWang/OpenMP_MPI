# homework 3 Genome assemble

## make file

source init.sh   // or module load bupc

make

## run jobs

### on local machine

bash job-serial

### only running upc code
upcrun -n 4 hello_world


### on the server/remote machine

sbatch job-scale-multi-node

or

sbatch job-scale-single-node


### change UPC share heap size

upcc -shared-heap=144MB

### Check correctness of the UPC Genome assemble

cat pgen\*.out > pgen.out

sort pgen.out > pgen.sorted 

sort serial.out > serial.sorted

diff -q serial.sorted pgen.sorted
