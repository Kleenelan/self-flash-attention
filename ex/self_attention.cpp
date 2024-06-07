#include <stdio.h>
#include <string.h>

#include "cpu_gemm.h"
#include "utils.h"
#include "soft_max.h"
//all matrices are row major.

void cpu_self_attention()
{

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

	gemm_nt(Q_h, ldq, K_h, ldk, S_h, lds, N, N, d);// Q*K^t
	printf("\nS_h =\n");	print_matrix(S_h, N, N, lds);
	soft_max(P_h, ldp, S_h, lds, N, N);//mn
	printf("\nP_h =\n");	print_matrix(S_h, N, N, lds);






	return 0;
}



