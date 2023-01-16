#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <cublas_v2.h>

typedef struct {
	double error;
	int iter;
} RESULT;

bool CheckInput(int argc, const char* argv[], double* errorRate, int* gridSize, int* numIterations, int* errorInterval)
{
    static const double checkMaxErrorRate = 0.000001;  // 10^-6
    static const int checkGridSize[] = { 128, 256, 512 };
    static const int checkMaxNumIter = 1000000;  // 10^6

	if (argc < 5) 
	{
        printf("ERROR\n");
		printf("a.out 'error rate' 'grid size' 'number of iterations' 'error interval'\n");
		return false;
	}

    *errorRate = strtod(argv[1], NULL);
    if (*errorRate < checkMaxErrorRate) 
	{
        printf("Max error rate is %f.\n", checkMaxErrorRate);
        return false;
    }

    *gridSize = atoi(argv[2]);
    if (*gridSize != checkGridSize[0] && *gridSize != checkGridSize[1] && *gridSize != checkGridSize[2]) {
        printf("Possible grid sizes are only %d^2, %d^2, %d^2.\n", checkGridSize[0], checkGridSize[1], checkGridSize[2]);
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
    int leftLower = 10;
    int leftUpper = 20;
    int rightUpper = 30;
    int rightLower = 20;
    double stepSize = 10 / (gridSize - 1);

    for (int i = 0; i < gridSize; i++) {
        grid[i] = leftUpper + i * stepSize; // uppper line
        grid[(gridSize - 1) * gridSize + i] = leftLower + i * stepSize; // down line
        grid[i * gridSize] = rightLower - i * stepSize; // left line
        grid[i * gridSize + gridSize - 1] = rightUpper - i * stepSize; // right line
    }
}

RESULT HeatEquation(double* grid, int gridSize, int numIterations, double errorRate, int errorInterval)
{
	int doubleGridSize = gridSize * gridSize;
	double* firstGrid = (double*)malloc(sizeof(double) * doubleGridSize);
	double* secondGrid = (double*)malloc(sizeof(double) * doubleGridSize);
	double error = INFINITY;
	int iter = 0;

	if (errorInterval % 2 == 1) 
	{
		errorInterval++;
	}

    cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) 
    {
		free(firstGrid);
		free(secondGrid);
        printf ("ERROR: CUBLAS initialization failed. Error code %d\n", stat);
        RESULT result;
		result.error = INFINITY;
		result.iter = -1;
		return result;
    }

    #pragma acc data copy(grid [0:doubleGridSize]) create(firstGrid [0:doubleGridSize], secondGrid [0:doubleGridSize])
	{
        #pragma acc kernels loop independent collapse(2)
		for (int i = 0; i < gridSize; i++)
		{
			for (int j = 0; j < gridSize; j++)
			{
				int index = i * gridSize + j;
				firstGrid[index] = grid[index];
				secondGrid[index] = grid[index];
			}
		}

		int failed = 0;
        for (iter = 0; !failed && iter < numIterations && error > errorRate; iter+=2)
		{
            #pragma acc data present(firstGrid [0:doubleGridSize], secondGrid [0:doubleGridSize])
			#pragma acc kernels async 
			{
            	#pragma acc loop independent collapse(2)
				for (int i = 1; i < gridSize - 1; i++)
				{
					for (int j = 1; j < gridSize - 1; j++)
					{
					    int index = i * gridSize + j;
					    secondGrid[index] = 0.25 * (firstGrid[index - gridSize] + firstGrid[index + gridSize] + firstGrid[index - 1] + firstGrid[index + 1]);
				    }
				}

				#pragma acc loop independent collapse(2)
				for (int i = 1; i < gridSize - 1; i++)
			 	{
					for (int j = 1; j < gridSize - 1; j++)
					{
					    int index = i * gridSize + j;
					    firstGrid[index] = 0.25 * (secondGrid[index - gridSize] + secondGrid[index + gridSize] + secondGrid[index - 1] + secondGrid[index + 1]);
				    }
			 	}
			}
			
			int nextIter = iter + 2;
			if (nextIter % errorInterval == 0 || nextIter >= numIterations)
			{
				#pragma acc wait
				#pragma acc data present(grid [0:doubleGridSize], firstGrid [0:doubleGridSize], secondGrid [0:doubleGridSize]) copyout(error)
				#pragma acc host_data use_device(grid, firstGrid, secondGrid, error)
				{
					double* zeroGrid = grid;
					int errorIndex = 0;	
					double a = -1.0;

					stat = cublasDcopy(handle, doubleGridSize, firstGrid, 1, zeroGrid, 1);
					if (stat != CUBLAS_STATUS_SUCCESS) 
                    {
						printf ("CUBLAS grid copy failed with error code %d\n", stat);
						failed = 1;
					}					
					stat = cublasDaxpy(handle, doubleGridSize, &a, secondGrid, 1, zeroGrid, 1);
					if (stat != CUBLAS_STATUS_SUCCESS) 
                    {
						printf ("CUBLAS daxpy failed with error code %d\n", stat);
						failed = 1;
					}					
					stat = cublasIdamax(handle, doubleGridSize, zeroGrid, 1, &errorIndex);	
					if (stat != CUBLAS_STATUS_SUCCESS) 
                    {
						printf ("CUBLAS idamax failed with error code %d\n", stat);
						failed = 1;
					}					
					stat = cublasDcopy(handle, 1, zeroGrid + errorIndex, 1, &error, 1);
					if (stat != CUBLAS_STATUS_SUCCESS) 
                    {
						printf ("CUBLAS error copy failed with error code %d\n", stat);
						failed = 1;
					}
				}
                error = fabs(error);
			}
		}

        #pragma acc kernels loop independent collapse(2)
		for (int i = 0; i < gridSize; i++)
		{
			for (int j = 0; j < gridSize; j++)
			{
				int index = i * gridSize + j;
				grid[index] = firstGrid[index];
			}
		}
	}

	cublasDestroy(handle);
    free(firstGrid);
	free(secondGrid);

	RESULT result;
	result.error = error;
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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    RESULT result = HeatEquation(grid, gridSize, numIterations, errorRate, errorInterval);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int64_t timee = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("iter = %d \t error = %f \t time = %d \n", result.iter, result.error , timee);

	return 0;
}
