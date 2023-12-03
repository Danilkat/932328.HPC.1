
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
int addWrapper();
int printIdsWrapper();

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void what_is_my_id(unsigned int* const block,
    unsigned int* const thread,
    unsigned int* const warp,
    unsigned int* const calc_thread)
{
    /* Thread id is block index * block size + thread offset into the block */
    const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    block[thread_idx] = blockIdx.x;
    thread[thread_idx] = threadIdx.x;
    /* Calculate warp using built in variable warpSize */
    warp[thread_idx] = threadIdx.x / warpSize;
    calc_thread[thread_idx] = thread_idx;
}

__global__ void what_is_my_id_2d_A(
    unsigned int* const block_x,
    unsigned int* const block_y,
    unsigned int* const thread,
    unsigned int* const calc_thread,
    unsigned int* const x_thread,
    unsigned int* const y_thread,
    unsigned int* const grid_dimx,
    unsigned int* const block_dimx,
    unsigned int* const grid_dimy,
    unsigned int* const block_dimy)
{
    const unsigned int idx =(blockIdx.x * blockDim.x) + threadIdx.x;
    const unsigned int idy =(blockIdx.y * blockDim.y) + threadIdx.y;
    const unsigned int thread_idx =((gridDim.x * blockDim.x) * idy) + idx;
    block_x[thread_idx] = blockIdx.x;
    block_y[thread_idx] = blockIdx.y;
    thread[thread_idx] = threadIdx.x;
    calc_thread[thread_idx] = thread_idx;
    x_thread[thread_idx] = idx;
    y_thread[thread_idx] = idy;
    grid_dimx[thread_idx] = gridDim.x;
    block_dimx[thread_idx] = blockDim.x;
    grid_dimy[thread_idx] = gridDim.y;
    block_dimy[thread_idx] = blockDim.y;
}

#define ARRAY_SIZE 128
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

int main()
{
    //return addWrapper();
    return printIdsWrapper();
}

int printIdsWrapper() {
    /* Total thread count = 2 * 64 = 128 */
    const unsigned int num_blocks = 2;
    const unsigned int num_threads = 64;
    dim3 threads_rect(32, 4);
    dim3 blocks_rect(1, 4);
    char ch;
    /* Declare pointers for GPU based params */
    unsigned int* gpu_block;
    unsigned int* gpu_thread;
    unsigned int* gpu_warp;
    unsigned int* gpu_calc_thread;
    /* Declare loop counter for use later */
    unsigned int i;
    /* Allocate four arrays on the GPU */
    cudaMalloc((void**)&gpu_block, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_warp, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
    /* Execute our kernel */
    what_is_my_id<<<blocks_rect, threads_rect>>>(gpu_block, gpu_thread, gpu_warp,
        gpu_calc_thread);
    /* Copy back the gpu results to the CPU */
    cudaMemcpy(cpu_block, gpu_block, ARRAY_SIZE_IN_BYTES,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_warp, gpu_warp, ARRAY_SIZE_IN_BYTES,
        cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES,
        cudaMemcpyDeviceToHost);
    /* Free the arrays on the GPU as now we’re done with them */
    cudaFree(gpu_block);
    cudaFree(gpu_thread);
    cudaFree(gpu_warp);
    cudaFree(gpu_calc_thread);
    /* Iterate through the arrays and print */
    for (i=0; i < ARRAY_SIZE; i++)
    {
        printf("Calculated Thread: %3u - Block: %2u - Warp %2u - Thread %3u\n",
            cpu_calc_thread[i], cpu_block[i], cpu_warp[i], cpu_thread[i]);
    }
    return 0;
}



int addWrapper() {
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
