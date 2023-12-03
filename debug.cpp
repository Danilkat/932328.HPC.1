#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
    case 2: // Fermi
        if (devProp.minor == 1) cores = mp * 48;
        else cores = mp * 32;
        break;
    case 3: // Kepler
        cores = mp * 192;
        break;
    case 5: // Maxwell
        cores = mp * 128;
        break;
    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
        else if (devProp.minor == 0) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
        else printf("Unknown device type\n");
        break;
    case 8: // Ampere
        if (devProp.minor == 0) cores = mp * 64;
        else if (devProp.minor == 6) cores = mp * 128;
        else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
        else printf("Unknown device type\n");
        break;
    case 9: // Hopper
        if (devProp.minor == 0) cores = mp * 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}

void printDevProp(cudaDeviceProp devProp)
{
	int CUDACores = getSPcores(devProp);
	printf("%s\n", devProp.name);
	printf("Major revision number:         %d\n", devProp.major);
	printf("Minor revision number:         %d\n", devProp.minor);
	printf("Total global memory:           %u", devProp.totalGlobalMem);
	printf(" bytes\n");
	printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
	printf("Total amount of shared memory per block: %u\n", devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", devProp.regsPerBlock);
	printf("Warp size:                     %d\n", devProp.warpSize);
	printf("Maximum memory pitch:          %u\n", devProp.memPitch);
	printf("Total amount of constant memory:         %u\n", devProp.totalConstMem);
	printf("CUDACores:         %d\n", CUDACores);
	return;
}

void getCudaDevice() {
	int deviceID;
	cudaDeviceProp props;

	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&props, deviceID);
	printDevProp(props);
};