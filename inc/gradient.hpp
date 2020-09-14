#define BLOCK_SIZE_X 4
#define BLOCK_SIZE_Y 4
#define BLOCK_SIZE_Z 4

//#define N 256
#include "model.hpp"
//#include "algo_rohsa.hpp"
#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime_api.h>
#include "model.hpp"
#include <math.h>
#include <cmath>

//#include "gradient.cu"
void test();
void gradient_L(double* dF_over_dB, int* taille_dF_over_dB, int product_taille_dF_over_dB, double* params, int* taille_params, int product_taille_params, int n_gauss);
void gradient_L_2(double* deriv, int* taille_deriv, int product_taille_deriv, double* params, int* taille_params, int product_taille_params, double* residual, int* taille_residual, int product_residual, double* std_map, int* taille_std_map, int product_std_map, int n_gauss);
//extern void gradient(double* dF_over_dB, int* taille_dF_over_dB, int product_taille_dF_over_dB, double* params, int* taille_params, int product_taille_params, model &M);
