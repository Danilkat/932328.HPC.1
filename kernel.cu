
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <algorithm> 
#include <conio.h>
#include <stdio.h>
#include "debug.h"
#include <ctime>
#include <Windows.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
int addWrapper(int arraySize);
int printIdsWrapper();
int printIds2DWrapper();

__global__ void addKernel(int *c, const int *a, const int *b, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        c[i] = a[i] + b[i];
    }
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
//#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))
#define ARRAY_SIZE_X 32
#define ARRAY_SIZE_Y 16
#define ARRAY_SIZE_IN_BYTES ((ARRAY_SIZE_X) * (ARRAY_SIZE_Y) * (sizeof(unsigned int)))

unsigned int cpu_block[ARRAY_SIZE];
unsigned int cpu_thread[ARRAY_SIZE];
unsigned int cpu_warp[ARRAY_SIZE];
unsigned int cpu_calc_thread[ARRAY_SIZE];

int main()
{
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    return addWrapper(1000000);
    //return printIdsWrapper();
    //getCudaDevice();
    //return printIds2DWrapper();
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

int printIds2DWrapper() {
    unsigned int cpu_block_x[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_block_y[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_warp[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_calc_thread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_xthread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_ythread[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_grid_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_block_dimx[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_grid_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    unsigned int cpu_block_dimy[ARRAY_SIZE_Y][ARRAY_SIZE_X];
    /* Total thread count = 32 * 4 = 128 */
    const dim3 threads_rect(32, 4); /* 32 * 4 */
    const dim3 blocks_rect(1, 4);
    /* Total thread count = 16 * 8 = 128 */
    const dim3 threads_square(16, 8); /* 16 * 8 */
    const dim3 blocks_square(2, 2);
    /* Needed to wait for a character at exit */
    char ch;
    /* Declare pointers for GPU based params */
    unsigned int* gpu_block_x;
    unsigned int* gpu_block_y;
    unsigned int* gpu_thread;
    unsigned int* gpu_warp;
    unsigned int* gpu_calc_thread;
    unsigned int* gpu_xthread;
    unsigned int* gpu_ythread;
    unsigned int* gpu_grid_dimx;
    unsigned int* gpu_block_dimx;
    unsigned int* gpu_grid_dimy;
    unsigned int* gpu_block_dimy;
    /* Allocate four arrays on the GPU */
    cudaMalloc((void**)&gpu_block_x, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_block_y, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_calc_thread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_xthread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_ythread, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_grid_dimx, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_block_dimx, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_grid_dimy, ARRAY_SIZE_IN_BYTES);
    cudaMalloc((void**)&gpu_block_dimy, ARRAY_SIZE_IN_BYTES);
    for (int kernel = 0; kernel < 2; kernel++)
    {
        switch (kernel)
        {
        case 0:
        {
            /* Execute our kernel */
            what_is_my_id_2d_A <<<blocks_rect, threads_rect>>> (gpu_block_x, gpu_block_y,
                gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
                gpu_grid_dimy, gpu_block_dimy);
        } break;
        case 1:
        {
            /* Execute our kernel */
            what_is_my_id_2d_A <<<blocks_square, threads_square>>> (gpu_block_x, gpu_block_y,
                gpu_thread, gpu_calc_thread, gpu_xthread, gpu_ythread, gpu_grid_dimx, gpu_block_dimx,
                gpu_grid_dimy, gpu_block_dimy);
        } break;
        default: exit(1); break;
        }
        /* Copy back the gpu results to the CPU */
        cudaMemcpy(cpu_block_x, gpu_block_x, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_y, gpu_block_y, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_thread, gpu_thread, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_calc_thread, gpu_calc_thread, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_xthread, gpu_xthread, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_ythread, gpu_ythread, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_grid_dimx, gpu_grid_dimx, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_dimx, gpu_block_dimx, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_grid_dimy, gpu_grid_dimy, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        cudaMemcpy(cpu_block_dimy, gpu_block_dimy, ARRAY_SIZE_IN_BYTES,
            cudaMemcpyDeviceToHost);
        printf("\nKernel %d\n", kernel);
        /* Iterate through the arrays and print */
        for (int y = 0; y < ARRAY_SIZE_Y; y++)
        {
            for (int x = 0; x < ARRAY_SIZE_X; x++)
            {
                printf("CT: %2u BKX: %1u BKY: %1u TID: %2u YTID: %2u XTID: %2u GDX: %1u BDX: %1u GDY % 1u BDY % 1u\n", cpu_calc_thread[y][x], cpu_block_x[y][x], cpu_block_y[y][x],
                    cpu_thread[y][x], cpu_ythread[y][x], cpu_xthread[y][x], cpu_grid_dimx[y][x],
                    cpu_block_dimx[y][x], cpu_grid_dimy[y][x], cpu_block_dimy[y][x]);
                /* Wait for any key so we can see the console window */
                ch = getch();
            }
        }
        /* Wait for any key so we can see the console window */
        printf("Press any key to continue\n");
        ch = getch();
    }
    /* Free the arrays on the GPU as now we’re done with them */
    cudaFree(gpu_block_x);
    cudaFree(gpu_block_y);
    cudaFree(gpu_thread);
    cudaFree(gpu_calc_thread);
    cudaFree(gpu_xthread);
    cudaFree(gpu_ythread);
    cudaFree(gpu_grid_dimx);
    cudaFree(gpu_block_dimx);
    cudaFree(gpu_grid_dimy);
    cudaFree(gpu_block_dimy);
}

int addWrapper(int arraySize) {
    size_t size = arraySize * sizeof(int);
    // Allocate the host input vector A
    int* h_A = (int*)malloc(size);

    // Allocate the host input vector B
    int* h_B = (int*)malloc(size);

    // Allocate the host output vector C
    int* h_C = (int*)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < arraySize; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(h_C, h_A, h_B, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

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

    // Высчитываем с момента копирования
    unsigned int start_time = clock();

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

    int threadsPerBlock = std::min((unsigned int)1024, size);
    int blocksPerGrid = std::max((unsigned int)1, (size + threadsPerBlock - 1) / threadsPerBlock);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(dev_c, dev_a, dev_b, size);

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

    unsigned int end_time = clock();
    unsigned int search_time = end_time - start_time;
    printf("Время выполнения: %d мс.", search_time);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

