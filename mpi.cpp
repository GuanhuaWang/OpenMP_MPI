#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <set>
#include "util.h"

int num = 0;
double dim = 0.0;
double size = 0.0;
int NSTEPS = 10000;


int main( int argc, char **argv )
{    
    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    //printf("the n is %d\n", n);
    //-----------------------------------------------------------------------------------------


    //int n = 100;
    //int tag=1, i;
    //
    //  set up MPI
    //
    int n_proc, rank;
    // init the MPI nodes by passing the input parameters
    // to all the nodes in the system
    //MPI_Status stat;
    MPI_Init( &argc, &argv );
    //printf("get the init\n");
    // get the size of the node in the system
    // the results are saved at n_proc
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if (n_proc > 1) {
        // get the rank of the MPI network, saved at rank
        // rank is like the ID of a node for this local machine
        
        //printf("the rank is %d, the num of proc is %d\n", rank, n_proc);
        FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
        FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

        
        // this is the buffer for send infor
        // same here, we need to allo the space for n particles
        particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
        // alloc the space for buffer
        int sizeSendLower = 0;
        int sizeSendUpper = 0;
        int sizeReceiveLower = 0;
        int sizeReceiveUpper = 0;

        particle_t *sendBufferParticleLower = (particle_t*) malloc(n * sizeof(particle_t));
        particle_t *receiveBufferParticleLower = (particle_t*) malloc (n * sizeof(particle_t));

        particle_t *sendBufferParticleUpper = (particle_t*) malloc(n * sizeof(particle_t));
        particle_t *receiveBufferParticleUpper = (particle_t*) malloc (n * sizeof(particle_t));
        
        // here we need to set up the env, i.e. the grid 
        set_size( n ); // calculate the size
        
        int *boardcast_sizes = (int*) malloc(n_proc * sizeof(int));
        int *boardcast_offset = (int*) malloc(n_proc * sizeof(int));
        // added for the mpigatherv
        int *gather_sizes = (int*) malloc(n_proc * sizeof(int));
        int *gather_offset = (int*) malloc(n_proc * sizeof(int));
        int *gather_count_sizes = (int*) malloc(n_proc * sizeof(int));
        int *gather_count_offset = (int*) malloc(n_proc * sizeof(int));
        for (int i = 0; i < n_proc; i++) {
            gather_count_sizes[i] = 1;
            gather_count_offset[i] = i;
        }

        for (int i = 0; i < n_proc; i++) {
            boardcast_sizes[i] = n;
            boardcast_offset[i] = 0;
        }

        init_grid_dim(); // cal the dim and num of the grid system

        int row_per_proc = num / n_proc; // the row of grids per each proc
        int grid_per_proc = row_per_proc * num;
        if (rank == n_proc - 1) {
            // the last one should keep all the rest of grid
            row_per_proc = num - num / n_proc *(n_proc - 1);
            grid_per_proc = num * row_per_proc;
        }
        row_per_proc += 2;  // we allocate the grid margin with the main grid together
        grid_per_proc += 2*num;
        // then here to allocate the space for the grids on this local machine
        // printf("init info: rank: %d; row_per_proc: %d; grid_per_proc: %d; num: %d \n", rank, row_per_proc, grid_per_proc, num);
        subgrid *local_grid = new subgrid[grid_per_proc];
        init_grid_main(local_grid, rank, grid_per_proc, n_proc); // init the grid with the padding
        //test_print_gird(local_grid, grid_per_proc);
        // printf("after grid\n");
        // then we set one fake pnt to pnt to the core grid

        subgrid *core_grid = local_grid + num;

        
        // printf("finish the alloc space for buffers\n");
        // define the particle type on the openMPI, which has 6 doubles and one int id, extend the MPI_DOUBLE
        MPI_Datatype PARTICLE, oldtypes[2];     // there are two types
        int blockcounts[2]; // for the MPI TYPE DEFINE
        /* MPI_Aint type used to be consistent with syntax of */ 
        /* MPI_Type_extent routine */ 
        MPI_Aint    offsets[2], extent; 
        offsets[0] = 0; 
        oldtypes[0] = MPI_DOUBLE; 
        blockcounts[0] = 6;     // 6 double number 
        /* Setup description of the 2 MPI_INT fields n, type */ 
        /* Need to first figure offset by getting size of MPI_FLOAT */ 
        MPI_Type_extent(MPI_DOUBLE, &extent); 
        offsets[1] = 6 * extent; 
        oldtypes[1] = MPI_INT; 
        blockcounts[1] = 1; 

        /* Now define structured type and commit it */ 

        MPI_Type_create_struct(2, blockcounts, offsets, oldtypes, &PARTICLE); 
        MPI_Type_commit(&PARTICLE); 

        if( rank == 0 ) {
            // printf("before init the particles\n");
            init_particles( n, particles ); // we have alloc space for pnter particles
            // here we can just boradcast all the particles to the nodes
            //test_print_particle(particles, n);
            // printf("after init the particles\n");
            //for (i=0; i<n_proc; i++) 
            //  MPI_Send(particles, n, PARTICLE, i, tag, MPI_COMM_WORLD);
      
        }
        //int source = 0;
        //MPI_Recv(des, n, PARTICLE, source, tag, MPI_COMM_WORLD, &stat);
        MPI_Scatterv( particles, boardcast_sizes, boardcast_offset, PARTICLE, particles, n, PARTICLE, 0, MPI_COMM_WORLD );
        // printf("finish the MPI_Scatterv, rank is %d\n", rank);
        //printf("the rank is %d\n", rank);
        //test_print_particle(particles, n);
        fill_grid(particles, local_grid, n, rank, n_proc, grid_per_proc);
        //printf("finished fill, rank is %d\n", rank);
        
        //printf("begin to communicate the force\n");
        double simulation_time = read_timer( );
        
        for( int step = 0; step < NSTEPS; step++ )
        {
            navg = 0;
            dmin = 1.0;
            davg = 0.0;
            // 
            //  collect all global data locally (not good idea to do)
            //
            /**
            int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, const int *recvcounts, const int *displs,
                       MPI_Datatype recvtype, MPI_Comm comm)
            **/
            // MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
            
            // here send and receive the message, here maybe data racing
            // parepre the data for send
            
            //printf("begin to communicate the force\n");
            if (rank == 0) {
                // here we just need to prepare the lower buffer
                getLowerSend(sendBufferParticleLower, core_grid, row_per_proc, &sizeSendLower);
                // here we need to send the mess to say how many particle we want to send to the next one...
                MPI_Send(&sizeSendLower, 1, MPI_INT, 1, rank + n_proc, MPI_COMM_WORLD);
                MPI_Recv(&sizeReceiveLower, 1, MPI_INT, 1, rank + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // then we can begin to send the particles
                MPI_Send(sendBufferParticleLower, sizeSendLower, PARTICLE, rank + 1, rank + 3*n_proc, MPI_COMM_WORLD);
                MPI_Recv(receiveBufferParticleLower, sizeReceiveLower, PARTICLE, rank + 1, rank + 1 + 2*n_proc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


            } else if (rank == n_proc - 1) {
                getUpperSend(sendBufferParticleUpper, core_grid, row_per_proc, &sizeSendUpper);
                MPI_Send(&sizeSendUpper, 1, MPI_INT, rank - 1, rank, MPI_COMM_WORLD);
                MPI_Recv(&sizeReceiveUpper, 1, MPI_INT, rank-1, rank - 1 + n_proc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Send(sendBufferParticleUpper, sizeSendUpper, PARTICLE, rank - 1, rank + 2*n_proc, MPI_COMM_WORLD);
                MPI_Recv(receiveBufferParticleUpper, sizeReceiveUpper, PARTICLE, rank - 1, rank - 1 + 3*n_proc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


            } else {
                getLowerSend(sendBufferParticleLower, core_grid, row_per_proc, &sizeSendLower);
                getUpperSend(sendBufferParticleUpper, core_grid, row_per_proc, &sizeSendUpper);

                // for tag rule, go up is rank, go down is rank + n_proc
                MPI_Send(&sizeSendUpper, 1, MPI_INT, rank - 1, rank, MPI_COMM_WORLD);
                MPI_Send(&sizeSendLower, 1, MPI_INT, rank + 1, rank + n_proc, MPI_COMM_WORLD);

                MPI_Recv(&sizeReceiveUpper, 1, MPI_INT, rank-1, rank - 1 + n_proc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&sizeReceiveLower, 1, MPI_INT, rank+1, rank + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
                MPI_Send(sendBufferParticleLower, sizeSendLower, PARTICLE, rank + 1, rank + 3*n_proc, MPI_COMM_WORLD);
                MPI_Send(sendBufferParticleUpper, sizeSendUpper, PARTICLE, rank - 1, rank + 2*n_proc, MPI_COMM_WORLD);

                MPI_Recv(receiveBufferParticleLower, sizeReceiveLower, PARTICLE, rank + 1, rank + 1 + 2*n_proc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(receiveBufferParticleUpper, sizeReceiveUpper, PARTICLE, rank - 1, rank - 1 + 3*n_proc, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            }
            
            //printf("after receive and send mess\n");
            // here we need to update the margin grid
            if (rank == 0) {
                updateGridLower(receiveBufferParticleLower, sizeReceiveLower, local_grid, rank, n_proc, row_per_proc);
            } else if (rank == n_proc - 1) {
                updateGridUpper(receiveBufferParticleUpper, sizeReceiveUpper, local_grid, rank, n_proc);
            } else {
                updateGridLower(receiveBufferParticleLower, sizeReceiveLower, local_grid, rank, n_proc, row_per_proc);
                updateGridUpper(receiveBufferParticleUpper, sizeReceiveUpper, local_grid, rank, n_proc);            
            }
            //printf("after update the grid\n");
        
            if( find_option( argc, argv, "-no" ) == -1 )
              if( fsave && (step%SAVEFREQ) == 0 )
                save( fsave, n, particles );


            apply_forces(&dmin, &davg, &navg, grid_per_proc, core_grid, local_grid);
            //printf(" after apply force\n");


            // here I try the allreduce..
            move_particles(core_grid, rank, n_proc, grid_per_proc, sendBufferParticleUpper, 
                &sizeSendUpper);
            // first all gather to get the count of the particles
            MPI_Allgatherv(&sizeSendUpper, 1, MPI_INT, gather_sizes, gather_count_sizes, gather_count_offset, MPI_INT, MPI_COMM_WORLD);
            // all the particles are gatherd in the upper buffer, and we have the size of that for all nodes
            //test_print_gather(gather_sizes, n_proc); // it's the send the size for each rank
            // change the offset
            gather_offset[0] = 0;
            for (int i = 1; i < n_proc; i++) {
                gather_offset[i] = gather_offset[i-1] + gather_sizes[i - 1];
            }
            //test_print_particle(sendBufferParticleUpper, sizeSendUpper);
            // then do the gatherv
            // here root get the all data... and it will partition it into each node and sent to them
            MPI_Gatherv(sendBufferParticleUpper, sizeSendUpper, PARTICLE, receiveBufferParticleUpper,
                         gather_sizes, gather_offset, PARTICLE, 0, MPI_COMM_WORLD);
            // then we have all particles in the receiveBufferParticleUpper...
            int totalSize = gather_offset[n_proc - 1] +gather_sizes[n_proc - 1];
            if(rank == 0) {
                //printf("after gather\n");
                //test_print_particle(receiveBufferParticleUpper, gather_offset[n_proc - 1] +gather_sizes[n_proc - 1]);
                // do parition
                //partitionParticle(receiveBufferParticleUpper, sendBufferMove, gather_offset[n_proc - 1] + gather_sizes[n_proc - 1], n,
                //                   n_proc, gather_sizes, boardcast_sizes, boardcast_offset);
                
                // just scatterv the receive to all the ranks
                // boardcast to the others
                
                for (int i = 0; i < n_proc; i++) {
                    boardcast_sizes[i] = totalSize;
                    boardcast_offset[i] = 0;
                }
                MPI_Scatterv(receiveBufferParticleUpper, boardcast_sizes, boardcast_offset, PARTICLE,
                         receiveBufferParticleLower, totalSize, PARTICLE, 0, MPI_COMM_WORLD );
                /**
                for (int i = 0; i < n_proc; i++) {
                    boardcast_sizes[i] = 1;
                    boardcast_offset[i] = i;
                }
                MPI_Scatterv(gather_sizes, boardcast_sizes, boardcast_offset, MPI_INT, &sizeReceiveUpper, 1, MPI_INT, 0, MPI_COMM_WORLD );
                **/
            }
            MPI_Barrier(MPI_COMM_WORLD);
            // update the particle
            updateAddParticle2(particles, receiveBufferParticleLower, core_grid,
                       rank, n_proc, totalSize, grid_per_proc);
            MPI_Barrier(MPI_COMM_WORLD);
            /** 
            updateAddParticle(particles, receiveBufferParticleLower, core_grid, rank, n_proc, sizeReceiveLower, grid_per_proc);
            updateAddParticle(particles, receiveBufferParticleUpper, core_grid, rank, n_proc, sizeReceiveUpper, grid_per_proc);
            **/
            if( find_option( argc, argv, "-no" ) == -1 )
            {
              
              MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
              MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
              MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

            // all reduce the three values to the rank 0
              if (rank == 0){
                //
                // Computing statistical data
                //
                if (rnavg) {
                  absavg +=  rdavg/rnavg;
                  nabsavg++;
                }
                if (rdmin < absmin) absmin = rdmin;
              }
            }
        }

        simulation_time = read_timer( ) - simulation_time;
      
        if (rank == 0) {  
          printf( "n = %d, simulation time = %g seconds", n, simulation_time);

          if( find_option( argc, argv, "-no" ) == -1 )
          {
            if (nabsavg) absavg /= nabsavg;
          // 
          //  -The minimum distance absmin between 2 particles during the run of the simulation
          //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
          //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
          //
          //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
          //
          printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
          if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
          if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
          }
          printf("\n");     
            
          //  
          // Printing summary data
          //  
          if( fsum)
            fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
        }
      
        //
        //  release resources
        //
        if ( fsum )
            fclose( fsum );
        
        free(sendBufferParticleUpper);
        free(sendBufferParticleLower);
        free(receiveBufferParticleUpper);
        free(receiveBufferParticleLower);
        free(boardcast_sizes);
        free(boardcast_offset);
        free(gather_sizes);
        free(gather_offset);
        free(gather_count_offset);
        free(gather_count_sizes);
        delete[] local_grid;
        free( particles );
        if( fsave )
            fclose( fsave );


        MPI_Type_free(&PARTICLE);
        MPI_Finalize();
        
        return 0;
    } else {
        // get the rank of the MPI network, saved at rank
        // rank is like the ID of a node for this local machine
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        //printf("the rank is %d, the num of proc is %d\n", rank, n_proc);
        FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
        FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

        
        // this is the buffer for send infor
        // same here, we need to allo the space for n particles
        particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
        // alloc the space for buffer
        
        // here we need to set up the env, i.e. the grid 
        set_size( n ); // calculate the size
        init_particles( n, particles );

        init_grid_proc0();
        fill_grid_proc0(particles, n);
        double simulation_time = read_timer( );

        for( int step = 0; step < NSTEPS; step++ )
        {
        navg = 0;
            davg = 0.0;
        dmin = 1.0;

            apply_forces_proc0(&dmin, &davg, &navg, particles, n);

            move_particles_proc0(particles, n);

            if( find_option( argc, argv, "-no" ) == -1 )
            {
              //
              // Computing statistical data
              //
              if (navg) {
                absavg +=  davg/navg;
                nabsavg++;
              }
              if (dmin < absmin) absmin = dmin;

              //
              //  save if necessary
              //
              if( fsave && (step%SAVEFREQ) == 0 )
                  save( fsave, n, particles );
            }
        }
        simulation_time = read_timer( ) - simulation_time;

        printf( "n = %d, simulation time = %g seconds", n, simulation_time);

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          if (nabsavg) absavg /= nabsavg;
        //
        //  -The minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
        }
        printf("\n");

        //
        // Printing summary data
        //
        if( fsum)
            fprintf(fsum,"%d %g\n",n,simulation_time);

        //
        // Clearing space
        //
        if( fsum )
            fclose( fsum );
        free( particles );
        if( fsave )
            fclose( fsave );
        dele_grid_proc0();
        MPI_Finalize();
        return 0;

    }
    
    



}
