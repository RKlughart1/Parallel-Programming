
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
__global__ void initialize(double *a, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<n)
    a[i] =(double)i/n;
}

int main()
{
//Serial Code
    const int n = 10000000;	
    double *a;
    double start, end;
    a = (double*)malloc(n * sizeof(double));
    int i;          
    start = clock();
    for (i = 0; i < n; i++)
        a[i] = (double)i / n;
    end = clock();
    for (i = 0; i < 5; i++)
    printf("a[%d]: %.7f\n",i, a[i]);
 
        printf(" ...\n");
    for (i = n-5; i < n; i++)
        printf("a[%d]: %.7f\n", i, a[i]);
    double total = (end - start) / CLOCKS_PER_SEC;
    printf("time: %f\n\n",total);
    //Cuda
    double* ac;
    double* d_a;
    ac = (double*)malloc(n * sizeof(double));

    printf("Cuda\n");
    cudaMalloc(&d_a, sizeof(double)*n);
    double t = clock();
   
   
    initialize<<<10000,1000>>>(d_a, n);
    cudaDeviceSynchronize();
    t = (clock() - t) / CLOCKS_PER_SEC;

    cudaMemcpy(ac, d_a, n*sizeof(double), cudaMemcpyDeviceToHost);
    
    for (i = 0; i < 5; i++)
        printf("a[%d]: %.7f\n", i, ac[i]);

    printf(" ...\n");
    for (i = n - 5; i < n; i++)
        printf("a[%d]: %.7f\n", i, ac[i]);
    printf("time:%f\n", t);
    double timesfaster = total / t;
    printf("Using cuda, the code executed %f times faster\n", timesfaster);
}