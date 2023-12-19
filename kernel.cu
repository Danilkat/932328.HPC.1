
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <algorithm> 
#include <conio.h>
#include <stdio.h>
#include "debug.h"
#include <ctime>
#include <Windows.h>
#include <iostream>
#include "addVectors.cuh"
#include "histogram.cuh"
#include "debug.cuh"

int main()
{
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    getCudaDevice();

    int size = 0;
    std::cin >> size;
    /*addWrapper(size);*/
 

    //addConstWrapper();

    //return printIdsWrapper();
    //return printIds2DWrapper();
    histogramWrapper(size);
    
}

