#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cub/device/device_reduce.cuh>
#include <cub/block/block_reduce.cuh>

#define CHECK_CUDA_ERRORS(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) 
  {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

typedef struct 
{
	double error;
	int iter;
} RESULT;

bool CheckInput(int argc, const char* argv[], double* errorRate, int* gridSize, int* numIterations, int* errorInterval)
{
	static const double checkMaxErrorRate = 0.000001;  // 10^-6
	static const int checkGridSize[] = {128, 256, 512, 1024 };
	//static const int checkGridSize[] = {8, 16, 128, 256, 512, 1024 };
	static const int checkMaxNumIter = 1000000;  // 10^6

	if (argc < 5)
	{
		printf("ERROR\n");
		printf("a.out 'error rate' 'grid size' 'number of iterations'\n");
		return false;
	}

	*errorRate = strtod(argv[1], NULL);
	if (*errorRate < checkMaxErrorRate)
	{
		printf("Max error rate is %f.\n", checkMaxErrorRate);
		return false;
	}

	*gridSize = atoi(argv[2]);
	if (*gridSize != checkGridSize[0] && *gridSize != checkGridSize[1] && *gridSize != checkGridSize[2] && *gridSize != checkGridSize[3]) {
		printf("Possible grid sizes are only %d^2, %d^2, %d^2, %d^2.\n", checkGridSize[0], checkGridSize[1], checkGridSize[2], checkGridSize[3]);
		return false;
	}

	*numIterations = atoi(argv[3]);
	if (*numIterations > checkMaxNumIter)
	{
		printf("Max number of iterations is %d.\n", checkMaxNumIter);
		return false;
	}

	*errorInterval = atoi(argv[4]);
	if (*errorInterval <= 0)
	{
		*errorInterval = 1;
		printf("error interval = 1\n");
	}

	return true;
}

// grid:
// 20   30
//          => diff = 10
// 10   20
void FillingGrid(double* grid, int gridSize)
{
	double leftLower = 10.;
	double leftUpper = 20.;
	double rightUpper = 30.;
	double rightLower = 20.;
	double stepSize = 10. / (gridSize - 1);

	for (int i = 0; i < gridSize; i++) 
	{
		grid[i] = leftUpper + i * stepSize; // uppper line
		grid[(gridSize - 1) * gridSize + i] = leftLower + i * stepSize; // down line
		grid[i * gridSize] = rightLower - i * stepSize; // left line
		grid[i * gridSize + gridSize - 1] = rightUpper - i * stepSize; // right line
	}
}

void printGrid(double* grid, int N)
{
	for(int x = 0; x < N; x++)
	{
		for(int y = 0; y < N; y++)
		{
			printf("%f ", grid[x*N+y]);
		}
		printf("\n");
	}
}

__global__ void CalcNewGrid(double* grid0, double* grid1, int N) 
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	if (x > 0 && x < N - 1 && y > 0 && y < N - 1) 
	{
        int index = y * N + x;
        grid1[index] = 0.25 * (grid0[index - N] + grid0[index + N] + grid0[index - 1] + grid0[index + 1]);
	}
}

__global__ void DifferenceGrids(double* grid0, double* grid1, double* d_diff_grid, int N)
{
    int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	if (x < N && y < N) 
	{
		int index = y * N + x;
		d_diff_grid[index] = fabs(grid0[index] - grid1[index]);
	}
}

double GetError(double* d_diff_grid, int N)
{
	size_t len = N * N * sizeof(int);
	double *d_err = NULL;
    cudaMalloc((void**)&d_err, sizeof(double));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    CHECK_CUDA_ERRORS(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_diff_grid, d_err, len));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    CHECK_CUDA_ERRORS(cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes,  d_diff_grid, d_err, sqrt(len)));
	
	double *h_err = (double*)malloc(sizeof(double));
    CHECK_CUDA_ERRORS(cudaMemcpy(h_err, d_err, sizeof(double), cudaMemcpyDeviceToHost));

	cudaFree(d_err);
    cudaFree(d_temp_storage);

    return *h_err;
}

RESULT HeatEquation(double* h_grid, int N, int numIterations, double errorRate, int errorInterval)
{
	size_t gridNumItems = N * N * sizeof(double);
	double h_err = INFINITY;
	int iter = 0;

	if (errorInterval % 2 == 1) 
	{
		errorInterval++;
	}
	
	CHECK_CUDA_ERRORS(cudaHostRegister(h_grid, gridNumItems, cudaHostRegisterDefault));

	double* d_grid0;
	double* d_grid1;
	double* d_diff_grid;
	CHECK_CUDA_ERRORS(cudaMalloc(&d_grid0, gridNumItems));
	CHECK_CUDA_ERRORS(cudaMalloc(&d_grid1, gridNumItems));
	CHECK_CUDA_ERRORS(cudaMalloc(&d_diff_grid, gridNumItems));

	CHECK_CUDA_ERRORS(cudaMemcpy(d_grid0, h_grid, gridNumItems, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_grid1, h_grid, gridNumItems, cudaMemcpyHostToDevice));

	dim3 blockSize(8, 8);
	dim3 gridSize(ceil((double) N / blockSize.x), ceil((double) N / blockSize.y));

	for (iter = 0; iter < numIterations && h_err > errorRate; iter += 2) 
	{
		if (iter % errorInterval == 0 || iter >= numIterations-1) 
		{
			CalcNewGrid<<<gridSize, blockSize>>>(d_grid0, d_grid1, N);
			CalcNewGrid<<<gridSize, blockSize>>>(d_grid1, d_grid0, N);
			DifferenceGrids<<<gridSize, blockSize>>>(d_grid0, d_grid1, d_diff_grid, N);

            //printf("iter: %d\terr: %f\n", iter, h_err);

			h_err = GetError(d_diff_grid, N);
		}	
		else
		{
			CalcNewGrid<<<gridSize, blockSize>>>(d_grid0, d_grid1, N);
			CalcNewGrid<<<gridSize, blockSize>>>(d_grid1, d_grid0, N);
		}
	}
	CHECK_CUDA_ERRORS(cudaMemcpy(h_grid, d_grid0, gridNumItems, cudaMemcpyDefault));

	cudaHostUnregister(h_grid);
	cudaFree(d_grid0);
	cudaFree(d_grid1);
	cudaFree(d_diff_grid);

	RESULT result;
	result.error = h_err;
	result.iter = iter;

	return result;
}

int main(int argc, const char* argv[])
{
	double errorRate;
	int gridSize, numIterations, errorInterval;

	if (!CheckInput(argc, argv, &errorRate, &gridSize, &numIterations, &errorInterval))
	{
		return 1;
	}

	double* grid = (double*)malloc(sizeof(double) * gridSize * gridSize);
	FillingGrid(grid, gridSize);

	//printGrid(grid, gridSize);

	struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	RESULT result = HeatEquation(grid, gridSize, numIterations, errorRate, errorInterval);

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	int64_t time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
	printf("iter = %d \t error = %f \t time = %ld \n", result.iter, result.error, time);

	return 0;
}
