#include <stdio.h>
#include <string.h>

#include "cpu_gemm.h"
#include "utils.h"
#include "cpu_softmax.h"
//all matrices are row major.

void cpu_self_attention(float* Q, int ldq,
						float* K, int ldk,
						float* V, int ldv,
						float* S, int lds,
						float* P, int ldp,
						float* O, int ldo,
						int N, int d)
{
	gemm_nt(Q, ldq, K, ldk, S, lds, N, N, d);// S = Q*K^t     (NxN) = (Nxd) * (dxN)
					printf("\nS =\n");	print_matrix(S, N, N, lds);
	cpu_softmax_column(P, ldp, S, lds, N, N);// P(NxN) = softmax(S(NxN))
					printf("\nP =\n");	print_matrix(S, N, N, lds);
	gemm_nn(P, ldp, V, ldv, O, ldo, N, d, N);// O = P*V     (Nxd) = (NxN) * (Nxd)
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

	N = 7;//1024;
	d = 7;//64;

	int ldq, ldk, ldv;
	float *Q_h = nullptr;//(Nxd)
	float *K_h = nullptr;//(Nxd)
	float *V_h = nullptr;//(Nxd)

	int lds, ldp, ldo;
	float *S_h = nullptr;//(NxN)
	float *P_h = nullptr;//(NxN)
	float *O_h = nullptr;//(Nxd)

	ldq = d;//512
	ldk = d;//512
	ldv = d;//512

	lds = N;
	ldp = N;
	ldo = d;

{
	Q_h = (float*)malloc(N*ldq*sizeof(float));
	K_h = (float*)malloc(N*ldk*sizeof(float));
	V_h = (float*)malloc(N*ldv*sizeof(float));

	S_h = (float*)malloc(N*lds*sizeof(float));
	P_h = (float*)malloc(N*ldp*sizeof(float));
	O_h = (float*)malloc(N*ldo*sizeof(float));

	memset(S_h, 0x00, N*lds*sizeof(float));
	memset(P_h, 0x00, N*ldp*sizeof(float));
	memset(O_h, 0x00, N*ldo*sizeof(float));

	init_matrix(Q_h, N, d, ldq, 2022);
	init_matrix(K_h, N, d, ldk, 2023);
	init_matrix(V_h, N, d, ldv, 2024);

	printf("\nQ_h =\n");	print_matrix(Q_h, N, d, ldq);
	printf("\nK_h =\n");	print_matrix(K_h, N, d, ldk);
	printf("\nV_h =\n");	print_matrix(V_h, N, d, ldv);
}
	cublas_self_attention(Q_h, ldq, K_h, ldk, V_h, ldv, N, d);

	gemm_nt(Q_h, ldq, K_h, ldk, S_h, lds, N, N, d);// S = Q*K^t     (NxN) = (Nxd) * (dxN)
	printf("\nS_h =\n");	print_matrix(S_h, N, N, lds);
	
	cpu_softmax_column(P_h, ldp, S_h, lds, N, N);// P(NxN) = softmax(S(NxN))
	printf("\nP_h =\n");	print_matrix(P_h, N, N, ldp);
	
	gemm_nn(P_h, ldp, V_h, ldv, O_h, ldo, N, d, N);// O = P*V     (Nxd) = (NxN) * (Nxd)
	printf("\nO_h =\n");	print_matrix(O_h, N, d, ldo);







	return 0;
}



