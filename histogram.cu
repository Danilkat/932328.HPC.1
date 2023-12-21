#include <iostream>
#include <cstdlib> // For rand() function
#include <ctime>   // For time() function
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cassert>

#define BIN_COUNT 256
#define SHARED_BIN_COUNT 32
#define THREADS 256
#define BLOCKS 1000


unsigned char* generateRandomString(int size) {
    std::srand(std::time(0));
    size_t dataSize = size * sizeof(unsigned char);

    unsigned char* randomString = (unsigned char*)malloc(dataSize);

    for (int i = 0; i < size; ++i) {
        unsigned char randomChar = (unsigned char)(std::rand() % BIN_COUNT);
        //unsigned char randomChar = (unsigned char)(32);
        randomString[i] = randomChar;
    }

    return randomString;
}

__global__ void histogramKernel(unsigned char* d_hist_data,
    long long size, unsigned long long* d_bin_data)
{
    __shared__ unsigned int d_bin_data_shared[BIN_COUNT];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int step = blockDim.x * gridDim.x;

    d_bin_data_shared[threadIdx.x] = 0;

    __syncthreads();

    
    while (i < size) {
        atomicAdd(&d_bin_data_shared[d_hist_data[i]], 1);
        i += step;
    }

    __syncthreads();

    atomicAdd(&d_bin_data[threadIdx.x], d_bin_data_shared[threadIdx.x]);
}

void myhistogramCPU(unsigned char* h_hist_data, unsigned long long* h_bin_data, unsigned int N) {
    clock_t start_time = clock();

    for (unsigned int i = 0; i < BIN_COUNT; i++) h_bin_data[i] = 0;
    for (unsigned int i = 0; i < N; i++) {
        unsigned char data = h_hist_data[i];
        h_bin_data[data]++;
    }

    clock_t end_time = clock();
    double search_time = (double)(end_time - start_time);
    printf("Время выполнения CPU: %f мс.\n", search_time);
}

cudaError_t histogramWithCuda(unsigned char* h_hist_data, unsigned long long* h_bin_data_GPU, unsigned long long size) {
    unsigned char* d_hist_data;
    unsigned long long* d_bin_data;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&d_hist_data, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&d_bin_data, BIN_COUNT * sizeof(long long));
    cudaMemset(d_bin_data, 0, BIN_COUNT * sizeof(long long));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Высчитываем с момента копирования
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaStatus = cudaMemcpy(d_hist_data, h_hist_data, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! %d");
        printf("%d", cudaStatus);
        goto Error;
    }

    histogramKernel << <size / 256,
        256 >> > (
            d_hist_data, size, d_bin_data);

    cudaStatus = cudaMemcpy(h_bin_data_GPU, d_bin_data, BIN_COUNT * sizeof(long long), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        printf("%d", cudaStatus);
        goto Error;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogramKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching histogramKernel!\n", cudaStatus);
        goto Error;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Время выполнения GPU: %f мс.", milliseconds);


Error:
    cudaFree(d_hist_data);
    cudaFree(d_bin_data);

    return cudaStatus;
}

int histogramWrapper(unsigned int size) {
    unsigned char* h_hist_data;
    unsigned long long* h_bin_data_CPU, * h_bin_data_GPU;

    h_hist_data = generateRandomString(size);
    /*for (int i = 0; i < size; i++)
    {
        if ((int)h_hist_data[i] > 31) {
            std::cout << h_hist_data[i];
        }
        std::cout << (int)h_hist_data[i];
        std::cout << std::endl;
    }*/
    h_bin_data_CPU = (unsigned long long*)malloc(BIN_COUNT * sizeof(unsigned long long));
    h_bin_data_GPU = (unsigned long long*)malloc(BIN_COUNT * sizeof(unsigned long long));

    cudaError_t cudaStatus = histogramWithCuda(h_hist_data, h_bin_data_GPU, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "histogramWithCuda failed!");
        return 1;
    }

    myhistogramCPU(h_hist_data, h_bin_data_CPU, size);

    //std::cout << "\n0 - " << h_bin_data_CPU[0] << std::endl;
    /*for (int i = 0; i < BIN_COUNT; i++)
    {
        if (h_bin_data_CPU[i] > 0) {
            std::cout << i << " - " << h_bin_data_CPU[i] << std::endl;
        }
    }
    std::cout << std::endl;*/
    std::cout << "\nСравнение результатов...\n";
    bool match = true;
    for (size_t i = 0; i < BIN_COUNT; i++)
    {
        if (h_bin_data_CPU[i] != h_bin_data_GPU[i]) {
            printf("Index %d. Asserted: %d. From kernel: %d ", i, h_bin_data_CPU[i], h_bin_data_GPU[i]);
            printf("FAILED \n");
            match = false;
        }
        /*else {
            printf("\n");
        }*/
    }
    std::cout << (match ? "Результаты сошлись" : "Результаты не сошлись");

    //std::cout << "\nGPU:\n";
    //std::cout << 0 << " - " << h_bin_data_GPU[0] << std::endl;
    /*for (int i = 0; i < BIN_COUNT; i++)
    {
        if (h_bin_data_GPU[i] > 0) {
            std::cout << i << " - " << h_bin_data_GPU[i] << std::endl;
        }
    }*/


    return 0;

}