
#include <stdio.h>
#include <string.h>

#include "cpu_gemm.h"
#include "utils.h"
#include "cpu_softmax.h"
//all matrices are row major.

void cpu_self_attention(float* Q, int ldq,
						float* K, int ldk,
						float* V, int ldv,
						int N, int d)
{
	float* S = nullptr;
	float* P = nullptr;
	float* O = nullptr;

	int lds = N;
	int ldp = N;
	int ldo = d;

	S = (float*)malloc(N*lds*sizeof(float));
	P = (float*)malloc(N*ldp*sizeof(float));
	O = (float*)malloc(N*ldo*sizeof(float));

	memset(S, 0x00, N*lds*sizeof(float));
	memset(P, 0x00, N*ldp*sizeof(float));
	memset(O, 0x00, N*ldo*sizeof(float));

    printf("\n into cpu_self_attention ...\n");
	gemm_nt(Q, ldq, K, ldk, S, lds, N, N, d);// S = Q*K^t     (NxN) = (Nxd) * (dxN)
												printf("\nS =\n");	print_matrix(S, N, N, lds);
	cpu_softmax_column(P, ldp, S, lds, N, N);// P(NxN) = softmax(S(NxN))
												printf("\nP =\n");	print_matrix(P, N, N, ldp);
	gemm_nn(P, ldp, V, ldv, O, ldo, N, d, N);// O = P*V     (Nxd) = (NxN) * (Nxd)
                    							printf("\nO =\n");  print_matrix(O, N, d, ldo);
	printf("\n out cpu_self_attention ...\n");

	free(S);
	free(P);
	free(O);
}

void cublas_self_attention(float* Q, int ldq,
						   float* K, int ldk,
						   float* V, int ldv,
						   int N, int d)
{

}

int main()
{
	int N, d;

	N = 32;//1024;
	d = 8;//64;

	int ldq, ldk, ldv;
	float *Q_h = nullptr;//(Nxd)
	float *K_h = nullptr;//(Nxd)
	float *V_h = nullptr;//(Nxd)

	ldq = d;//512
	ldk = d;//512
	ldv = d;//512

	Q_h = (float*)malloc(N*ldq*sizeof(float));
	K_h = (float*)malloc(N*ldk*sizeof(float));
	V_h = (float*)malloc(N*ldv*sizeof(float));

	init_matrix(Q_h, N, d, ldq, 2022);		printf("\nQ_h =\n");	print_matrix(Q_h, N, d, ldq);
	init_matrix(K_h, N, d, ldk, 2023);		printf("\nK_h =\n");	print_matrix(K_h, N, d, ldk);
	init_matrix(V_h, N, d, ldv, 2024);		printf("\nV_h =\n");	print_matrix(V_h, N, d, ldv);

    cpu_self_attention(Q_h, ldq, K_h, ldk, V_h, ldv, N, d);

	free(Q_h);
	free(K_h);
	free(V_h);

	return 0;
}





