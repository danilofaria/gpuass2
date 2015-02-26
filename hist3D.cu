#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <stdio.h>     
#include <stdlib.h>     
#include <sys/time.h>
#include <cuda.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>      // std::invalid_argument
#include <assert.h>     /* assert */



#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

// error checking for CUDA calls: use this around ALL your calls!
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
 
// cannot return values, so must store
// result in global memory location ("count")
// also: must make sure this thread maps to useful data! (what if
// the # of threads is > than the number of data elements!)
__global__ void histogram_gpu (unsigned int max, unsigned int *A, unsigned int *histogram)
{
 
    int n = blockDim.x * blockIdx.x + threadIdx.x;
 
    // do nothing if we are not in the useable space of
    // threads (see kernel launch call: you may be creating
    // more threads than you need)
    if (n >= max) return;
 
    unsigned int a = A[n];
 
    atomicAdd(&histogram[(a-1)/100], 1);
 
}
 

__global__ void histogram_gpu2 (unsigned int max, unsigned int *A, unsigned int *histogram)
{
 
    __shared__ unsigned int s_histogram [10];
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int tId = threadIdx.x;
    if (tId < 10) 
        s_histogram[tId]=0;
    __syncthreads();
 
    // do nothing if we are not in the useable space of
    // threads (see kernel launch call: you may be creating
    // more threads than you need)
    if (n >= max) return;
 
    unsigned int a = A[n];
 
    atomicAdd(&s_histogram[(a-1)/100], 1);
    __syncthreads();

    if (tId < 10) 
        histogram[tId] = s_histogram[tId];
 
}

int main (int argc, char *argv[])
{
 
    if (argc <= 1) {
        printf("please pass in the name of a file\n");
        return 0;
    }

    std::ifstream infile;
    char * filename = argv[1];
    infile.open(filename);

    if (!infile) {
        printf("please indicate a valid filename\n");
        return 0;
    }

    int x, y, z;
    infile >> x >> y >> z;
    int maxTested = x * y *z;
    unsigned int n, i;


    // std::vector<unsigned int> intVec;
    // while(infile >> n) intVec.push_back(n);
    // how many test do we wish to make:
    // unsigned int maxTested = intVec.size(); 

    struct timeval t0, t1, t2;
 
 
    // allocate the array of integers to hold the data:
    unsigned int *h_intAArray;
    h_intAArray = (unsigned int *) malloc (maxTested * sizeof (unsigned int));
 

    std::string line;
    i =0;
    while(getline(infile,line)){
        std::stringstream   linestream(line);
        std::string         value;
        while(getline(linestream,value,','))
        {
            char * pEnd;
            int n = std::strtol (value.c_str(),&pEnd,10);
            if(n==0){
                continue;
            }else{
            h_intAArray[i] = n;
            i++;}
        }
        // Line Finished
    }
    assert (maxTested==i);

    // // fill it with numbers in file
    // for (unsigned int i = 0; i < maxTested; ++i) {
    //     infile >> n;
    //     h_intAArray[i] = n;
    // }

    // start basic timing:
    gettimeofday (&t0, 0);


    int histogram[10];
    for(int i=0;i<10;i++) histogram[i] = 0;
    // count how many are prime:
    for (int i = 0; i < maxTested; ++i) {
        n = h_intAArray[i];
        histogram[(n-1)/100]++; 
    }

    // how much time has elapsed?
    gettimeofday (&t1, 0);
 
    //
    // GPU version
    //
 
    // allocate the A array on the GPU, and copy the data over:
    unsigned int *d_intAArray;
    // allocate the histogram array on the GPU
    unsigned int *d_histogram;
 
    GPU_CHECKERROR(
    cudaMalloc ((void **) &d_intAArray, maxTested * sizeof (unsigned int))
    );
 
    GPU_CHECKERROR(
    cudaMemcpy ((void *) d_intAArray,
                (void *) h_intAArray,
                maxTested * sizeof (unsigned int),
                cudaMemcpyHostToDevice)
    );

    GPU_CHECKERROR(
    cudaMalloc ((void **) &d_histogram, 10 * sizeof (unsigned int))
    );
 
    GPU_CHECKERROR(
        cudaMemset ((void *) d_histogram, 0, 10 * sizeof (unsigned int))
    );
 
    // we want to run a grid of 512-thread blocks (for reasons you
    // will understand later. How many such blocks will we need?
    // NOTE: be SURE to prevent integer division if you use this
    // snippet: that "1.0*" is absolutely required to prevent
    // rounding before the ceil() call:
    unsigned int threads_per_block = 512;
    unsigned int num_blocks = ceil (maxTested / (1.0*threads_per_block) );
 
    // launch the kernel:
    histogram_gpu<<<num_blocks, threads_per_block>>>
                                        (maxTested,
                                        d_intAArray,
                                        d_histogram);
 
    // get back the histogram:
    unsigned int h_histogram[10];
 
    cudaMemcpy ((void *) h_histogram,
                (void *) d_histogram,
                10 * sizeof(unsigned int),
                cudaMemcpyDeviceToHost);
    
    // make sure the GPU is finished doing everything!
    cudaDeviceSynchronize();

    // finish timing:
    gettimeofday (&t2, 0);
 
    // free up the memory:
    cudaFree (d_intAArray);
    cudaFree (d_histogram);
    free (h_intAArray);
 
    // complete the timing:
    float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
    float timdiff2 = (1000000.0*(t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)) / 1000000.0;

    printf ("%.2f\n", timdiff1);
    for(int i=0;i<10;i++){
        printf ("%d ", histogram[i]);
    }
    printf ("\n");

    printf ("%.2f\n", timdiff2);
    for(int i=0;i<10;i++){
        printf ("%d ", h_histogram[i]);
    }
    printf ("\n");
 
    // printf ("%d %.2f %d %.2f\n", primeCount, timdiff1, h_numPrimes, timdiff2);
  
}