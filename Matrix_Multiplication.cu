#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#define N 6000

//kernal function
__global__ void matrix_mult(int* A, int* B, int* C, int a_m, int b_n)
{
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int t = 0;
    if (y < b_n && x < a_m) {
        for (int i = 0; i < a_m; i++) {
            t = t + A[x * a_m + i] * B[i * b_n + y];
        }
        C[x * b_n + y] = t;
    }
}

// randomly initialize matrix A
void A_matrix(int* j, int m, int  n)
{
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            j[row * n + col] = rand() % 100;
        }
    }
}

// randomly initialize matrix B
void B_matrix(int* j, int n, int  k)
{
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < k; col++) {
            j[row * k + col] = rand() % 100;
        }
    }
}

int main(void)
{  // computing cuda runtime
    float runtime;
    cudaEvent_t start, stop;
    cudaStream_t stream0, stream1, stream2;
    int* A, * B, * C;   //host copies of a,b,c
    int* dev_A0, * dev_B0, * dev_C0; //device copies of a0,b0,c0
    int* dev_A1, * dev_B1, * dev_C1; //device copies of a1,b1,c1
    int* dev_A2, * dev_B2, * dev_C2; //device copies of a2,b2,c2
    int size = sizeof(int);
    int* a[N];
    int* b[N];
    int* c[N];
    //matrix dimensions A[50][20],B[20][50]
    int a_m = 50;
    int a_n = 20;
    int b_m = 20;
    int b_n = 50;
    //initalize cuda runtime
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //Initalize stream
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    //start cuda runtime
    cudaEventRecord(start, 0);

    //allocate space for device
    //stream0
    cudaMalloc((void**)&dev_A0, size * a_m * a_n);
    cudaMalloc((void**)&dev_B0, size * b_m * b_n);
    cudaMalloc((void**)&dev_C0, size * a_m * b_n);
    //stream1
    cudaMalloc((void**)&dev_A1, size * a_m * a_n);
    cudaMalloc((void**)&dev_B1, size * b_m * b_n);
    cudaMalloc((void**)&dev_C1, size * a_m * b_n);
    //stream2
    cudaMalloc((void**)&dev_A2, size * a_m * a_n);
    cudaMalloc((void**)&dev_B2, size * b_m * b_n);
    cudaMalloc((void**)&dev_C2, size * a_m * b_n);
    //allocate space for host
    cudaMallocHost((void**)&A, size * N);
    cudaMallocHost((void**)&B, size * N);
    cudaMallocHost((void**)&C, size * N);

    for (int i = 0; i < N; i++)
    {
        //initaializing A1.....AN matrix
        A_matrix(A, a_m, a_n);
        a[i] = A;
        //initaializing B matrix
        B_matrix(B, b_m, b_n);
        b[i] = B;
    }

    //multistreaming
    int split1 = N / 2;
    int split = N / 3;
    for (int i = 0;i < split; i++)
    {
        // copies results from host to device
        cudaMemcpyAsync(dev_A0, a[i], size * a_m * a_n, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_B0, b[i], size * b_m * b_n, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_A1, a[split1 + i], size * a_m * a_n, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_B1, b[split1 + i], size * b_m * b_n, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_A2, a[split + i], size * a_m * a_n, cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(dev_B2, b[split + i], size * b_m * b_n, cudaMemcpyHostToDevice, stream2);

        //calling kernal function
        int Threads = 32;
        dim3 grids((b_n + Threads - 1) / Threads, (a_m + Threads - 1) / Threads);
        dim3 blocks(Threads, Threads);
        matrix_mult << <grids, blocks, 0, stream0 >> > (dev_A0, dev_B0, dev_C0, a_m, b_n);
        matrix_mult << <grids, blocks, 0, stream1 >> > (dev_A1, dev_B1, dev_C1, a_m, b_n);
        matrix_mult << <grids, blocks, 0, stream2 >> > (dev_A2, dev_B2, dev_C2, a_m, b_n);

        // Transefr results from device to host 
        cudaMemcpyAsync(c[i], dev_C0, size * a_m * b_n, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(c[split + i], dev_C1, size * a_m * b_n, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(c[split + i], dev_C2, size * a_m * b_n, cudaMemcpyHostToDevice, stream2);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);


    // terminating cuda runtime
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runtime, start, stop);
    printf("Cuda Runtime for matrix with dimensions a=50x20, b=20x50 for N=6000 is : %f milliseconds\n\n", runtime);

    // free memory
    cudaFree(dev_A0);
    cudaFree(dev_B0);
    cudaFree(dev_C0);
    cudaFree(dev_A1);
    cudaFree(dev_B1);
    cudaFree(dev_C1);
    cudaFree(dev_A2);
    cudaFree(dev_B2);
    cudaFree(dev_C2);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    return(0);
}



