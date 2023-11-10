/*************************************************
 * Laplace Serial C Version
 *
 * Temperature is initially 0.0
 * Boundaries are as follows:
 *
 *      0         T         0
 *   0  +-------------------+  0
 *      |                   |
 *      |                   |
 *      |                   |
 *   T  |                   |  T
 *      |                   |
 *      |                   |
 *      |                   |
 *   0  +-------------------+ 100
 *      0         T        100
 *
 *  John Urbanic, PSC 2014
 *
 ************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

// size of plate
#define COLUMNS    1000
#define ROWS       1000

// largest permitted change in temp (This value takes about 3400 steps)
#define MAX_TEMP_ERROR 0.01

//   helper routines
void initialize(double Temperature_last[][COLUMNS+2], int rows_per_proc[], int rank, int nprocs);
void track_progress(int iteration, double Temperature[][COLUMNS+2], int rows_per_proc[], int rank);

int main(int argc, char *argv[]) {

    int ierr;
    int nprocs,rank;
    ierr = MPI_Init(&argc,&argv);
    ierr = MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int rows_per_proc[nprocs];

    // make these allocatable
    int rows_by_nprocs = floor(ROWS/nprocs);
    int extra_rows = ROWS - rows_by_nprocs*nprocs;
    for(int r=0; r<nprocs; r++){
    	rows_per_proc[r]=rows_by_nprocs+2;	
    }
    if((extra_rows % 4)==0){
    	rows_per_proc[nprocs-1]=rows_per_proc[nprocs-1]+extra_rows/4;
        rows_per_proc[nprocs-2]=rows_per_proc[nprocs-2]+extra_rows/4;
        rows_per_proc[nprocs-3]=rows_per_proc[nprocs-3]+extra_rows/4;
	rows_per_proc[nprocs-4]=rows_per_proc[nprocs-4]+extra_rows/4;
    }
    else if((extra_rows % 2)==0){
    	rows_per_proc[nprocs-1]=rows_per_proc[nprocs-1]+extra_rows/2;
	rows_per_proc[nprocs-2]=rows_per_proc[nprocs-2]+extra_rows/2;
    }
    else{
    	rows_per_proc[nprocs-1]=rows_per_proc[nprocs-1]+extra_rows;
    }
    

    double Temperature[rows_per_proc[rank]][COLUMNS+2];      // temperature grid
    double Temperature_last[rows_per_proc[rank]][COLUMNS+2]; // temperature grid from last iteration

    int i, j;                                            // grid indexes
    int max_iterations;                                  // number of iterations
    int iteration=1;                                     // current iteration
    double dt=100, global_dt=100;                                       // largest change in t
    struct timeval start_time, stop_time, elapsed_time;  // timers

    max_iterations = 4000;

    gettimeofday(&start_time,NULL); // Unix timer

    initialize(Temperature_last,rows_per_proc,rank,nprocs); // initialize Temp_last including boundary conditions

    // do until error is minimal or until max steps
    while ( global_dt > MAX_TEMP_ERROR && iteration <= max_iterations ) {
	// Put MPI Send here for both top and bottom edges; Send tlast
    	if(rank==0){
    	   ierr = MPI_Send(&Temperature_last[rows_per_proc[rank]-1],COLUMNS+2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);
    	}	 
    	else if(rank==nprocs-1){
	   ierr = MPI_Send(&Temperature_last[0],COLUMNS+2,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD);
    	}
    	else{
	   ierr = MPI_Send(&Temperature_last[rows_per_proc[rank]-1],COLUMNS+2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD); // bottom edge
           ierr = MPI_Send(&Temperature_last[0],COLUMNS+2,MPI_DOUBLE,rank-1,1,MPI_COMM_WORLD); // top edge
    	}	
	// Put MPI Recv here for both top and bottom edges; Recv tlast
   	if(rank==0){
    	   ierr = MPI_Recv(&Temperature_last[rows_per_proc[rank]-1],COLUMNS+2,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    	}
    	else if(rank==nprocs-1){
    	   ierr = MPI_Recv(&Temperature_last[0],COLUMNS+2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    	}
    	else{
    	   ierr = MPI_Recv(&Temperature_last[0],COLUMNS+2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    	   ierr = MPI_Recv(&Temperature_last[rows_per_proc[rank]-1],COLUMNS+2,MPI_DOUBLE,rank+1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    	}
	// Change loop indices
        // main calculation: average my four neighbors
        for(i = 1; i < rows_per_proc[rank]-1; i++) {
            for(j = 1; j <= COLUMNS; j++) {
                Temperature[i][j] = 0.25 * (Temperature_last[i+1][j] + Temperature_last[i-1][j] +
                                            Temperature_last[i][j+1] + Temperature_last[i][j-1]);
            }
        }
        
        dt = 0.0; // reset largest temperature change

	//Change loop indicies
        // copy grid to old grid for next iteration and find latest dt
        for(i = 1; i < rows_per_proc[rank]-1 ; i++){
            for(j = 1; j <= COLUMNS; j++){
	      dt = fmax( fabs(Temperature[i][j]-Temperature_last[i][j]), dt);
	      Temperature_last[i][j] = Temperature[i][j];
            }
        }
	for(j=1; j <= COLUMNS; j++){
	    if(rank==0){
	    	Temperature_last[rows_per_proc[rank]-1][j]=Temperature_last[rows_per_proc[rank]-2][j];
	    }
	    else if(rank==nprocs-1){
	    	Temperature_last[0][j]=Temperature_last[1][j];
	    }
	    else{
	    	Temperature_last[0][j]=Temperature_last[1][j];
	    	Temperature_last[rows_per_proc[rank]-1][j]=Temperature_last[rows_per_proc[rank]-2][j];
	    }
	}
	// MPI_AllReduce on dt here - synchronization point
	ierr = MPI_Allreduce(&dt,&global_dt,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
	
        // periodically print test values
	// Only rank==np-1 can print this(since the track_progress function prints the diagonal elements
	// near bottom right); note that if rank==np-1 has less than 5 rows this will break
        if((iteration % 100) == 0 && rank==nprocs-1 ) {
 	    track_progress(iteration,Temperature,rows_per_proc,rank);
        }

	iteration++;// no need to do anything with this since every process is sychronized here
    }

    gettimeofday(&stop_time,NULL);
    timersub(&stop_time, &start_time, &elapsed_time); // Unix time subtract routine

    // Only rank==0 does printing here
    if(rank==0){ 
    	printf("\nMax error at iteration %d was %f\n", iteration-1, global_dt);
    	printf("Total time was %f seconds.\n", elapsed_time.tv_sec+elapsed_time.tv_usec/1000000.0);
    }

    ierr = MPI_Finalize();
}


// initialize plate and boundary conditions
// Temp_last is used to to start first iteration
void initialize(double Temperature_last[][COLUMNS+2], int rows_per_proc[], int rank, int nprocs){

    int i,j;

    // Change to loop over local domain
    for(i = 0; i < rows_per_proc[rank]; i++){
        for (j = 0; j <= COLUMNS+1; j++){
            Temperature_last[i][j] = 0.0;
        }
    }

    // these boundary conditions never change throughout run

    // Change to loop over local domain
    // set left side to 0 and right to a linear increase
    int total_rows=0;
    for(int r=0; r<rank; r++){
    	total_rows = total_rows+rows_per_proc[r]-2;
    }
    for(i = 1; i < rows_per_proc[rank]-1; i++) {
       Temperature_last[i][0] = 0.0;
       Temperature_last[i][COLUMNS+1] = (100.0/ROWS)*(total_rows+i-1);
    }
    for(j=0; j <= COLUMNS+1; j++){
            if(rank==0){
                Temperature_last[rows_per_proc[rank]-1][j]=Temperature_last[rows_per_proc[rank]-2][j];
            }
            else if(rank==nprocs-1){
                Temperature_last[0][j]=Temperature_last[1][j];
            }
            else{
                Temperature_last[0][j]=Temperature_last[1][j];
                Temperature_last[rows_per_proc[rank]-1][j]=Temperature_last[rows_per_proc[rank]-2][j];
            }
        }
    
    // set top to 0 and bottom to linear increase
    if(rank==0){
    	for(j = 0; j <= COLUMNS+1; j++) {
           Temperature_last[0][j] = 0.0;
    	}
    }
    if(rank==nprocs-1){
    	for(j = 0; j <= COLUMNS+1; j++) {
           Temperature_last[rows_per_proc[rank]-1][j] = (100.0/COLUMNS)*j;
        }
    }
}


// print diagonal in bottom right corner where most action is
void track_progress(int iteration, double Temperature[][COLUMNS+2], int rows_per_proc[], int rank) {

    int i;
    int cnt=COLUMNS-4;
    int total_rows=0;
    for(int r=0; r<rank; r++){
        total_rows = total_rows+rows_per_proc[r]-2;
    }

    printf("---------- Iteration number: %d ------------\n", iteration);
    for(i = rows_per_proc[rank]-6; i <= rows_per_proc[rank]-2; i++) {
        printf("[%d,%d]: %5.2f  ", total_rows+i, total_rows+i, Temperature[i][cnt]);
	cnt++;
    }
    printf("\n");
}
