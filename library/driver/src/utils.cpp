#include "../include/utils.h"

void init_matrix(float *A, int M, int N, int lda, int seed)
{
	srand(seed);
	for(int i=0; i<M; i++)
	{
		for(int j=0; j<N; j++)
		{
			A[i*lda + j] = float(rand())/RAND_MAX;
		}
	}
}

void print_matrix(float *A, int M, int N, int lda)
{
	for(int i=0; i<M; i++)
	{
		for(int j=0; j<N; j++)
		{
			printf("%7.3f ", A[i*lda + j]);
		}

		printf("\n");
	}
}
