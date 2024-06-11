
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <limits>

#include "cpu_self_attention.h"
#include "utils.h"

#define N	32//2048
#define d	16//512
#define M 	(4*4*4*4)//(128*1024)//128K*4 Bytes,   4 = sizeof(float)

template<typename T>
void set_vector_inf_neg(T* m, int len)
{
	float inf_neg = -1*std::numeric_limits<T>::infinity();
	std::cout<<"inf_neg = "<< inf_neg<<std::endl;

	for(int idx=0; idx<len; idx++)
		m[idx] = inf_neg;
}

void load_matrix_block(float* A, int lda, int idx, float* B, int ldb, int row, int col)// copy A(idx, 0)    to  B(0, 0)  B(row x col)       default col = d
{
	for(int i=0; i<row; i++){//row == Bc, Br
		for(int j=0; j<col; j++){//col == d
			B[i*ldb + j] = A[idx*row*lda + j];
		}
	}
}

void load_vector_sgmnt(float* AY, int len_a, int idx, float* BY, int len_cp)
{
	for(int i=0; i<len_cp; i++){
		BY[i] = AY[idx*len_a + i];
	}
}

void flash_attention_cpu(float* Q, int ldq, float* K, int ldk, float* V, int ldv)
{
	//step 01
	constexpr int Br = M/(4*d);
	constexpr int Bc = (M/(4*d))<d? (M/(4*d)): d;
	constexpr int Tr = N/Br;
	constexpr int Tc = N/Bc;

	std::cout<< "Br ="<<Br<<" Bc = "<<Bc<<" Tr = "<<Tr<<" Tc = "<<Tc<<std::endl;
	//step 02
	float* O = nullptr;// O(N x d)
	int ldo = d;
	float* l = nullptr;
	float* m = nullptr;

	O = (float*)malloc(N*ldo*sizeof(float));// O(N x d)
	l = (float*)malloc(N*sizeof(float));
	m = (float*)malloc(N*sizeof(float));
	memset(O, 0x00, N*ldo*sizeof(float));// O(N x d)
	memset(l, 0x00, N*sizeof(float));
	set_vector_inf_neg<float>(m, N);
	//step 03
	// Q => Q_1, Q_2, ..., Q_Tr; Q_i(Br x d)
	// K => K_1, K_2, ..., K_Tc; K_j(Bc x d)
	// V => V_1, V_2, ..., V_Tc; V_j(Bc x d)

	//step 04
	// O => O_1, O_2, ..., O_Tr; O_i(Br x d)
	// l => l_1, l_2, ..., l_Tr; l_i(Br x 1)
	// m => m_1, m_2, ..., m_Tr; m_i(Br x 1)
	//step 05
	float* Kj = nullptr;
	float* Vj = nullptr;
	int ldkj  = d;
	int ldvj  = d;

	Kj = (float*)malloc(Bc*d*sizeof(float));
	Vj = (float*)malloc(Bc*d*sizeof(float));
	/////////////////////////////////////////////////////to step 09
	float* Q_i = nullptr;// Q_i(Br x d)
	float* O_i = nullptr;// O_i(Br x d)
	int ldqi = d;
	int ldoi = d;
	float* l_i = nullptr;// l_i(Br x 1)
	float* m_i = nullptr;// m_i(Br x 1)

	Q_i = (float*)malloc(Br*d*sizeof(float));
	O_i = (float*)malloc(Br*d*sizeof(float));
	l_i = (float*)malloc(Br*sizeof(float));
	m_i = (float*)malloc(Br*sizeof(float));
	/////////////////////////////////////////////////////////
	float* Sij = nullptr;// Sij(Br x Bc)
	int ldsij = Bc;
	Sij = (float*)malloc(Br*ldsij*sizeof(float));
	/////////////////////////////////////////////////////////
	for(int j=0; j<Tc; j++){
		//step 06 load Kj, Vj
		load_matrix_block(K, ldk, j, Kj, ldkj, Bc, d);// Kj(Bc x d)  column all are d;
		load_matrix_block(V, ldv, j, Vj, ldvj, Bc, d);
		//step 07 for
		for(int i=0; i<Tr; i++){
			//step 08 load Q_i, O_i, l_i, m_i from HBM to SRAM
			load_matrix_block(Q, ldq, i, Q_i, ldqi, Br, d);//Q_i(Br x d)
			load_matrix_block(O, ldo, i, O_i, ldoi, Br, d);// O_i(Br x d)
			load_vector_sgmnt(l, Br, i, l_i, Br);			//l_i(Br x 1)
			load_vector_sgmnt(m, Br, i, m_i, Br);			// m_i(Br x 1)
			//step 09 Sij = Qi*(K^t)j ; Sij(Br x Bc)

		}
	}



}

int main()
{
	int NN, dd;

	NN = N;//1024;
	dd = d;//64;

	int ldq, ldk, ldv;
	float *Q_h = nullptr;//(Nxd)
	float *K_h = nullptr;//(Nxd)
	float *V_h = nullptr;//(Nxd)

	ldq = dd;//512
	ldk = dd;//512
	ldv = dd;//512

	Q_h = (float*)malloc(NN*ldq*sizeof(float));
	K_h = (float*)malloc(NN*ldk*sizeof(float));
	V_h = (float*)malloc(NN*ldv*sizeof(float));

	init_matrix(Q_h, NN, dd, ldq, 2025);		printf("\nQ_h =\n");	print_matrix(Q_h, NN, dd, ldq);
	init_matrix(K_h, NN, dd, ldk, 2027);		printf("\nK_h =\n");	print_matrix(K_h, NN, dd, ldk);
	init_matrix(V_h, NN, dd, ldv, 2026);		printf("\nV_h =\n");	print_matrix(V_h, NN, dd, ldv);

    cpu_self_attention(Q_h, ldq, K_h, ldk, V_h, ldv, NN, dd);
	flash_attention_cpu(Q_h, ldq, K_h, ldk, V_h, ldv);

	free(Q_h);
	free(K_h);
	free(V_h);

	return 0;
}
