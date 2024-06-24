#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main (int argc, char* argv[])
{

  int thread_count = 6;

#pragma omp parallel num_threads(thread_count) 
    {

	    int myID = omp_get_thread_num(); 
	    int num_threads = omp_get_num_threads (); 




	    printf("Hello from thread %d of %d\n", myID, num_threads);
    }

  return 0;
}
