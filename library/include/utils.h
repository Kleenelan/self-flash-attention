#pragma once
#include <stdio.h>
#include <stdlib.h>

void init_matrix(float *A, int M, int N, int lda, int seed);
void print_matrix(float *A, int M, int N, int lda);