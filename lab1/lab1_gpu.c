#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//#include <cmath>

#define FLOAT_TYPE float

FLOAT_TYPE
compute_sin_on_cpu() 
{
    const unsigned int N = pow(10, 7);
    FLOAT_TYPE* sin_array = (FLOAT_TYPE *)malloc(sizeof(FLOAT_TYPE) * N);
    FLOAT_TYPE sin_sum = (FLOAT_TYPE)0.0;

    #pragma acc data create(sin_array[0:N]) copy(sin_sum)
    {
    #pragma acc kernels 
        for(int i = 0; i < N; i++)
        {
            sin_array[i] = sin(2*M_PI*i/N);
        }

    #pragma acc kernels loop reduction(+:sin_sum)
        for(int i = 0; i < N; i++) 
        {
            sin_sum += sin_array[i];
        }

    }

    free(sin_array);
    return sin_sum;
}

int main() 
{
    FLOAT_TYPE sin_sum = compute_sin_on_cpu();

    printf("Sum sin = %e\n", sin_sum);
}
