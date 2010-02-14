
// Author's : Rakesh Ginjupalli, Felix Rohrer
// Date		: 12/27/09
// we thank Dr Gaurav Khanna for his support

// common SDK header for standard utilities and system libs 
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// Host buffers for demo
// *********************************************************************
float *srcA, *srcB, *dst;        // Host buffers for OpenCL test

bool DEBUG = true;

// OpenCL Vars   
size_t szParmDataBytes;			// Byte size of context information
size_t szKernelLength;			// Byte size of kernel code

int iNumElements = 10;			// Length of float arrays to process

// Forward Declarations
// *********************************************************************
void fillFloatArray(float* arr, int length);
void printFloatArray(float* arr, char* name, int length);

// Main function 
// *********************************************************************
int main(int argc, char **argv)
{
	// set amount of numbers to be calculated
	if(argc > 1){
		iNumElements = atoi(argv[1]);
		printf("Setting numbers to %d\n", iNumElements);
		if(argc == 3){
			DEBUG = false;
		}
	}
    // set and log Global and Local work size dimensions
    size_t szLocalWorkSize = 256;

	float multiplier = iNumElements/(float)szLocalWorkSize;
	if(multiplier > (int)multiplier){
		multiplier += 1;
	}

	size_t szGlobalWorkSize = (int)multiplier * szLocalWorkSize; // rounded up to the nearest multiple of the LocalWorkSize

    // Allocate and initialize host arrays 
    srcA = (float *)malloc(sizeof(float) * szGlobalWorkSize);
    srcB = (float *)malloc(sizeof(float) * szGlobalWorkSize);
    dst = (float *)malloc(sizeof(float) * szGlobalWorkSize);

	fillFloatArray(srcA, iNumElements);
    fillFloatArray(srcB, iNumElements);

	if(DEBUG){
		printFloatArray(srcA, "Field A", iNumElements);
		printFloatArray(srcB, "Field B", iNumElements);
	}

    // Create the OpenCL context on a GPU device
    cl_context cxGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, NULL);

    // Get the list of GPU devices associated with context
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &szParmDataBytes);
    cl_device_id* cdDevices = (cl_device_id*)malloc(szParmDataBytes);
    clGetContextInfo(cxGPUContext, CL_CONTEXT_DEVICES, szParmDataBytes, cdDevices, NULL);

    // Create a command-queue
    cl_command_queue cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[0], 0, 0);

    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cl_mem cmDevSrcA = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, NULL);
    cl_mem cmDevSrcB = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, NULL);
    cl_mem cmDevDst = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, sizeof(cl_float) * szGlobalWorkSize, NULL, NULL);

	char* cSourceCL = " __kernel void VectorAdd(__global const float* a, __global const float* b, __global float* c, int iNumElements) "
					  " { "
					  "   int iGID = get_global_id(0); "
				      "   if (iGID < iNumElements) "
					  "      c[iGID] = a[iGID] + b[iGID]; "
					  " } ";

    // Create the program
    cl_program cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, NULL);

    clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);

    // Create the kernel
    cl_kernel ckKernel = clCreateKernel(cpProgram, "VectorAdd", NULL);

    // Set the Argument values
    clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (float*)&cmDevSrcA);
    clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (float*)&cmDevSrcB);
    clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (float*)&cmDevDst);
    clSetKernelArg(ckKernel, 3, sizeof(cl_int), (float*)&iNumElements);

    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back

    // Asynchronous write of data to GPU device
    clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcA, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, srcA, 0, NULL, NULL);
    clEnqueueWriteBuffer(cqCommandQueue, cmDevSrcB, CL_FALSE, 0, sizeof(cl_float) * szGlobalWorkSize, srcB, 0, NULL, NULL);

    // Launch kernel
    clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, &szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);

    // Synchronous/blocking read of results, and check accumulated errors
    clEnqueueReadBuffer(cqCommandQueue, cmDevDst, CL_TRUE, 0, sizeof(cl_float) * szGlobalWorkSize, dst, 0, NULL, NULL);
	
	if(DEBUG){
		printFloatArray(dst, "Result", iNumElements);
	}

    // Cleanup allocated objects
    if(cdDevices)free(cdDevices);
	if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cmDevSrcA)clReleaseMemObject(cmDevSrcA);
    if(cmDevSrcB)clReleaseMemObject(cmDevSrcB);
    if(cmDevDst)clReleaseMemObject(cmDevDst);

    // Free host memory
    free(srcA); 
    free(srcB);
    free (dst);
}

void fillFloatArray(float* arr, int length){
	for(int i=0;i<length;i++){
		arr[i] = rand() / (float)RAND_MAX;
	}
}

void printFloatArray(float* arr, char* name, int length){
	printf("%s:\n", name);
	for(int i=0;i<length;i++){
		printf("%.1f ", arr[i]);
	}printf("\n\n");
}
