#pragma once

#include <string.h>
#include "cpu_gemm.h"
#include "utils.h"
#include "cpu_softmax.h"

void cpu_self_attention(float* Q, int ldq,
						float* K, int ldk,
						float* V, int ldv,
						int N, int d);
