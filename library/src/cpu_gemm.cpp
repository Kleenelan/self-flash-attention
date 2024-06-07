#include "cpu_gemm.h"

void gemm_nn(float *A, int lda,		//A(M x K) rowMj
	     	 float *B, int ldb,		//B(K x N) rowMj
	     	 float *C, int ldc,		//C(M x N) rowMj
	      	 int M,
			 int N,
			 int K)
{
	for(int i=0; i<M; i++)
	{
		for(int j=0; j<N; j++)
		{
			float sigma = 0.0;

			for(int k=0; k<K; k++)
			{
				sigma += A[i*lda + k] * B[k*ldb + j];
			}

			C[i*ldc + j] = sigma;
		}
	}
}

void gemm_nt(float *A, int lda,		//A(M x K) rowMj
	     	 float *B, int ldb,		//B(N x K) rowMj
	     	 float *C, int ldc,		//C(M x N) rowMj
	      	 int M,
			 int N,
			 int K)
{
	for(int i=0; i<M; i++)
	{
		for(int j=0; j<N; j++)
		{
			float sigma = 0.0;

			for(int k=0; k<K; k++)
			{
				sigma += A[i*lda + k] * B[k + j*ldb];
			}

			C[i*ldc + j] = sigma;
		}
	}
}

void gemm_tn(float *A, int lda,		//A(M x K) rowMj
	     	 float *B, int ldb,		//B(N x K) rowMj
	     	 float *C, int ldc,		//C(M x N) rowMj
	      	 int M,
			 int N,
			 int K)
{
	for(int i=0; i<M; i++)
	{
		for(int j=0; j<N; j++)
		{
			float sigma = 0.0;

			for(int k=0; k<K; k++)
			{
				sigma += A[i + k*lda] * B[k*ldb + j];
			}

			C[i*ldc + j] = sigma;
		}
	}
}
