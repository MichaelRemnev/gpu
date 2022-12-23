#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

typedef struct {
	double error;
	int iter;
} RESULT;

bool CheckInput(int argc, const char* argv[], double* errorRate, int* gridSize, int* numIterations)
{
    static const double checkMaxErrorRate = 0.000001;  // 10^-6
    static const int checkGridSize[] = { 128, 256, 512 };
    static const int checkMaxNumIter = 1000000;  // 10^6

	if (argc < 3) 
	{
        printf("ERROR\n");
		printf("a.out 'error rate' 'grid size' 'number of iterations'\n");
		return false;
	}

    *errorRate = strtod(argv[1], NULL);
    if (*errorRate < checkMaxErrorRate) {
        printf("Max error rate is %f.\n", checkMaxErrorRate);
        return false;
    }

    *gridSize = atoi(argv[2]);
    if (*gridSize != checkGridSize[0] && *gridSize != checkGridSize[1] && *gridSize != checkGridSize[2]) {
        printf("Possible grid sizes are only %d^2, %d^2, %d^2.\n", checkGridSize[0], checkGridSize[1], checkGridSize[2]);
        return false;
    }

    *numIterations = atoi(argv[3]);
    if (*numIterations > checkMaxNumIter) {
        printf("Max number of iterations is %d.\n", checkMaxNumIter);
        return false;
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

RESULT HeatEquation(double* grid, int gridSize, int numIterations, double errorRate)
{
	int doubleGridSize = gridSize * gridSize;
	double* firstGrid = (double*)malloc(sizeof(double) * doubleGridSize);
	double* secondGrid = (double*)malloc(sizeof(double) * doubleGridSize);
	double error = 1.0;
	int iter = 0;

    #pragma acc data copy(grid [0:doubleGridSize]) create(firstGrid [0:doubleGridSize]) create(secondGrid [0:doubleGridSize])
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

		for (iter = 0; iter < numIterations && error > errorRate; iter++)
		{
            		#pragma acc data present(secondGrid [0:doubleGridSize], firstGrid [0:doubleGridSize])
			{
				if (iter % 50 == 0 || iter == numIterations-1)
				{
					error = 0.0;
                			#pragma acc kernels loop independent collapse(2) reduction(max:error)
					for (int i = 1; i < gridSize - 1; i++)
					{
						for (int j = 1; j < gridSize - 1; j++)
						{
							int index = i * gridSize + j;
							secondGrid[index] = 0.25 * (firstGrid[index - gridSize] + firstGrid[index + gridSize] + firstGrid[index - 1] + firstGrid[index + 1]);
							error = fmax(error, fabs(secondGrid[index] - firstGrid[index]));
						}
					}
				}
				else
                		{
                    		#pragma acc kernels loop independent collapse(2)
					for (int i = 1; i < gridSize - 1; i++)
				    	{
						for (int j = 1; j < gridSize - 1; j++)
						{
						    int index = i * gridSize + j;
						    secondGrid[index] = 0.25 * (firstGrid[index - gridSize] + firstGrid[index + gridSize] + firstGrid[index - 1] + firstGrid[index + 1]);
					    	}
				    	}
                		}
			}

			double* temporaryGrid = firstGrid;
			firstGrid = secondGrid;
			secondGrid = temporaryGrid;
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
    int gridSize, numIterations;
    
    if (!CheckInput(argc, argv, &errorRate, &gridSize, &numIterations))
    {
        return 1;
    }

    double* grid = (double*)malloc(sizeof(double) * gridSize * gridSize);
    FillingGrid(grid, gridSize);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    RESULT result = HeatEquation(grid, gridSize, numIterations, errorRate);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    int64_t timee = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("iter = %d \t error = %f \t time = %d \n", result.iter, result.error , timee);

	return 0;
}
