#pragma once

void gemm_nn(float *A, int lda,		//A(M x K) rowMj
	     	 float *B, int ldb,		//B(K x N) rowMj
	     	 float *C, int ldc,		//C(M x N) rowMj
	      	 int M,
			 int N,
			 int K);

void gemm_nt(float *A, int lda,		//A(M x K) rowMj
	     	 float *B, int ldb,		//B(K x N) rowMj
	     	 float *C, int ldc,		//C(M x N) rowMj
	      	 int M,
			 int N,
			 int K);

void gemm_tn(float *A, int lda,		//A(M x K) rowMj
	     	 float *B, int ldb,		//B(K x N) rowMj
	     	 float *C, int ldc,		//C(M x N) rowMj
	      	 int M,
			 int N,
			 int K);