# homework 3 Genome assemble

## make file

source init.sh

make

## run jobs

// on local machine

bash job-serial

### only running upc code
upcrun -n 4 hello_world


//on the server/remote machine

sbatch job-scale-multi-node

or

sbatch job-scale-single-node


### change UPC share heap size

upcc -shared-heap=144MB
