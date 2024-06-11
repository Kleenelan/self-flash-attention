#include "cpu_self_attention.h"
#include "../driver/include/cpu_gemm.h"
#include "../driver/include/utils.h"
#include "../driver/include/cpu_softmax.h"
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
