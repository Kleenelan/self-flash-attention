#pragma once
#include <math.h>
void cpu_softmax_column(float *P, int ldp, float* S, int lds, int M, int N);
