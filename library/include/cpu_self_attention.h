#pragma once

#include <string.h>


void cpu_self_attention(float* Q, int ldq,
						float* K, int ldk,
						float* V, int ldv,
						int N, int d);
