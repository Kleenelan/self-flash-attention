#include "cpu_softmax.h"
void cpu_softmax_column(float *P, int ldp, float* S, int lds, int M, int N)//P = softmax(S)  P(i,j) = exp(S(i,j))/sigma(exp(S(r,j)));  r=0,1,..,n-1 ;
{
    for(int j=0; j<N; j++){
        float sigma = 0.0f;

        for(int i=0; i<M; i++){
            sigma += exp(S[i*lds + j]);
        }

        for(int i=0; i<M; i++){
            P[i*ldp + j] = S[i*lds + j]/sigma;
        }
    }
}
