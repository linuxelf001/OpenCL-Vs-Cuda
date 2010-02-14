// Author's : Rakesh Ginjupalli, Felix Rohrer
// Date		: 12/27/09
// We thank Dr Gaurav Khanna for his support


// Includes
#include <stdio.h>

// Variables
float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

int iNumElements = 10;	// Length of float arrays to process
bool DEBUG = true;

// Functions
void fillFloatArray(float* arr, int length);
void printFloatArray(float* arr, char* name, int length);

// Device code
__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
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

    size_t size = iNumElements * sizeof(float);

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize input vectors
    fillFloatArray(h_A, iNumElements);
    fillFloatArray(h_B, iNumElements);

	if(DEBUG){
		printFloatArray(h_A, "Array A",  iNumElements);
		printFloatArray(h_B, "Array B",  iNumElements);
	}

    // Allocate vectors in device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (iNumElements + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, iNumElements);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
	if(DEBUG){
		printFloatArray(h_C, "Result",  iNumElements);
	}

    // Free device memory
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);

    // Free host memory
    if (h_A) free(h_A);
    if (h_B) free(h_B);
    if (h_C) free(h_C);
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
