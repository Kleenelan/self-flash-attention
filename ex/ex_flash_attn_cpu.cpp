
#include <stdio.h>
#include <string.h>

#include "cpu_self_attention.h"

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





