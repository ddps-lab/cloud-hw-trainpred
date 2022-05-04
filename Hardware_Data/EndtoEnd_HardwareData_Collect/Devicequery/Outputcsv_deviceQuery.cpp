/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

// std::system includes

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <fstream> 
#include <iostream>
#include <memory>
#include <string>

int *pArgc = NULL;
char **pArgv = NULL;

#if CUDART_VERSION < 5000

// CUDA-C includes
#include <cuda.h>

// This function wraps the CUDA Driver API into a template function
template <class T>
inline void getCudaAttribute(T *attribute, CUdevice_attribute device_attribute,
                             int device) {
  CUresult error = cuDeviceGetAttribute(attribute, device_attribute, device);

  if (CUDA_SUCCESS != error) {
    fprintf(
        stderr,
        "cuSafeCallNoSync() Driver API error = %04d from file <%s>, line %i.\n",
        error, __FILE__, __LINE__);

    exit(EXIT_FAILURE);
  }
}

#endif /* CUDART_VERSION < 5000 */

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

  printf("%s Starting...\n\n", argv[0]);
  printf(
      " CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
  } else {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::ofstream myfile;
    myfile.open ("devicequery.csv");

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
     myfile << "Total amount of global memory(MBytes)," << static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f) << "\n";
#else
    snprintf(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
    myfile << "Total amount of global memory(MBytes)," << static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f) << "\n";
#endif
    printf("%s", msg);

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);
    myfile << "Multiprocessors," << deviceProp.multiProcessorCount << "\n";
    myfile << "CUDA Cores/MP," << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << "\n";
    myfile << "CUDA Cores," << _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount << "\n";

    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    myfile << "GPU Max Clock rate(MHz)," << deviceProp.clockRate * 1e-3f << "\n";


#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    myfile << "Memory Clock rate," << deviceProp.memoryClockRate * 1e-3f << "\n";
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);
    myfile << "Memory Bus Width," << deviceProp.memoryBusWidth << "\n";
    if (deviceProp.l2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             deviceProp.l2CacheSize);
      myfile << "L2 Cache Size," << deviceProp.l2CacheSize << "\n";

    }

#else
    // This only available in CUDA 4.0-4.2 (but these were only exposed in the
    // CUDA Driver API)
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
    myfile << "Memory Clock rate," << memoryClock * 1e-3f << "\n";

    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,
                          CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
           memBusWidth);
    myfile << "Memory Bus Width," << memBusWidth << "\n";

    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    if (L2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             L2CacheSize);
      myfile << "L2 Cache Size," << L2CacheSize << "\n";
    }

#endif

    printf(
        "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
        "%d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
        deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
        deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    myfile << "Maximum Texture Dimension Size (xyz) 1D," << deviceProp.maxTexture1D << "\n";
    myfile << "Maximum Texture Dimension Size (xyz) 2D[0]," << deviceProp.maxTexture2D[0] << "\n";
    myfile << "Maximum Texture Dimension Size (xyz) 2D[1]," << deviceProp.maxTexture2D[1] << "\n";
    myfile << "Maximum Texture Dimension Size (xyz) 3D[0]," << deviceProp.maxTexture3D[0] << "\n";
    myfile << "Maximum Texture Dimension Size (xyz) 3D[1]," << deviceProp.maxTexture3D[1] << "\n";
    myfile << "Maximum Texture Dimension Size (xyz) 3D[2]," << deviceProp.maxTexture3D[2] << "\n";

    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    myfile << "Maximum Layered 1D Texture Size num," << deviceProp.maxTexture1DLayered[0] << "\n";
    myfile << "Maximum Layered 1D Texture Size layer," << deviceProp.maxTexture1DLayered[1] << "\n";

    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);
    myfile << "Maximum Layered 2D Texture Size num 2D[0]," << deviceProp.maxTexture2DLayered[0] << "\n";
    myfile << "Maximum Layered 2D Texture Size num 2D[1]," << deviceProp.maxTexture2DLayered[1] << "\n";
    myfile << "Maximum Layered 2D Texture Size num layers," << deviceProp.maxTexture2DLayered[2] << "\n";

    printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
    myfile << "Total amount of constant memory," << deviceProp.totalConstMem << "\n";

    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    myfile << "Total amount of shared memory per block," << deviceProp.sharedMemPerBlock << "\n";

    printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    myfile << "Total shared memory per multiprocessor," << deviceProp.sharedMemPerMultiprocessor << "\n";

    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    myfile << "Total number of registers available per block," << deviceProp.regsPerBlock << "\n";

    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    myfile << "Warp size," << deviceProp.warpSize << "\n";

    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    myfile << "Maximum number of threads per multiprocessor," << deviceProp.maxThreadsPerMultiProcessor << "\n";

    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    myfile << "Maximum number of threads per block," << deviceProp.maxThreadsPerBlock << "\n";

    printf("  Max dimension size of a thread block (xyz): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    myfile << "Max dimension size of a thread block (xyz) x," << deviceProp.maxThreadsDim[0] << "\n";
    myfile << "Max dimension size of a thread block (xyz) y," << deviceProp.maxThreadsDim[1] << "\n";
    myfile << "Max dimension size of a thread block (xyz) z," << deviceProp.maxThreadsDim[2] << "\n";

    printf("  Max dimension size of a grid size    (xyz): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    myfile << "Max dimension size of a grid size    (xyz) x," << deviceProp.maxGridSize[0] << "\n";
    myfile << "Max dimension size of a grid size    (xyz) y," << deviceProp.maxGridSize[1] << "\n";
    myfile << "Max dimension size of a grid size    (xyz) z," << deviceProp.maxGridSize[2] << "\n";

    printf("  Maximum memory pitch:                          %zu bytes\n",
           deviceProp.memPitch);
    myfile << "Maximum memory pitch," << deviceProp.memPitch << "\n";

    printf("  Texture alignment:                             %zu bytes\n",
           deviceProp.textureAlignment);
    myfile << "Texture alignment," << deviceProp.textureAlignment << "\n";

    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    myfile << "Concurrent copy and kernel execution," << (deviceProp.deviceOverlap ? "Yes\n" : "No\n") ;
    myfile << "Concurrent copy and kernel execution copy engines," << deviceProp.asyncEngineCount << "\n";

    printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    myfile << "Run time limit on kernels," << (deviceProp.kernelExecTimeoutEnabled ? "Yes\n" : "No\n" ) ;

    printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
    myfile << "Integrated GPU sharing Host Memory," << (deviceProp.integrated ? "Yes\n" : "No\n") ;

    printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");
    myfile << "Support host page-locked memory mapping," << (deviceProp.canMapHostMemory ? "Yes\n" : "No\n") ;

    printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
    myfile << "Alignment requirement for Surfaces," << (deviceProp.surfaceAlignment ? "Yes\n" : "No\n") ;

    printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
    myfile << "Device has ECC support," << (deviceProp.ECCEnabled ? "Enabled\n" : "Disabled\n") ;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
           deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                : "WDDM (Windows Display Driver Model)");
#endif
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");
    myfile << "Device supports Unified Addressing (UVA)," << (deviceProp.unifiedAddressing ? "Yes\n" : "No\n") ;

    printf("  Device supports Managed Memory:                %s\n",
           deviceProp.managedMemory ? "Yes" : "No");
    myfile << "Device supports Managed Memory," << (deviceProp.managedMemory ? "Yes\n" : "No\n") ;

    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    myfile << "Device supports Compute Preemption," << (deviceProp.computePreemptionSupported ? "Yes\n" : "No\n") ;

    printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");
    myfile << "Supports Cooperative Kernel Launch," << (deviceProp.cooperativeLaunch ? "Yes\n" : "No\n") ;

    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    myfile << "Supports MultiDevice Co-op Kernel Launch," << (deviceProp.cooperativeMultiDeviceLaunch ? "Yes\n" : "No\n") ;

    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
    myfile << "Device PCI Domain ID," << deviceProp.pciDomainID << "\n";
    myfile << "Device PCI Bus ID," << deviceProp.pciBusID  << "\n";
    myfile << "Device PCI location ID," << deviceProp.pciDeviceID << "\n";
    myfile.close();


    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Unknown",
        NULL};
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
  }

  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if (deviceCount >= 2) {
    cudaDeviceProp prop[64];
    int gpuid[64];  // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

      // Only boards based on Fermi or later can support P2P
      if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
          // on Windows (64-bit), the Tesla Compute Cluster driver for windows
          // must be enabled to support this
          && prop[i].tccDriver
#endif
      ) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }
          checkCudaErrors(
              cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
          printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                 prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
                 can_access_peer ? "Yes" : "No");
        }
      }
    }
  }

  // csv masterlog info
  // *****************************
  // exe and CUDA driver name
  printf("\n");
  std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
  char cTemp[16];

  // driver version
  sProfileString += ", CUDA Driver Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
  snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
           (driverVersion % 100) / 10);
#endif
  sProfileString += cTemp;

  // Runtime version
  sProfileString += ", CUDA Runtime Version = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
  snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);
#endif
  sProfileString += cTemp;

  // Device count
  sProfileString += ", NumDevs = ";
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  sprintf_s(cTemp, 10, "%d", deviceCount);
#else
  snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
#endif
  sProfileString += cTemp;
  sProfileString += "\n";
  printf("%s", sProfileString.c_str());

  printf("Result = PASS\n");

  // finish
  exit(EXIT_SUCCESS);
}

