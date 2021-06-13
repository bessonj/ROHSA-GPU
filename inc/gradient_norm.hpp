#ifndef DEF_GRAD_NORM
#define DEF_GRAD_NORM
/*
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 1

#define BLOCK_SIZE_X_BIS 4
#define BLOCK_SIZE_Y_BIS 4
#define BLOCK_SIZE_Z_BIS 8

#define BLOCK_SIZE_L2_X 4
#define BLOCK_SIZE_L2_Y 4
#define BLOCK_SIZE_L2_Z 4

#define BLOCK_SIZE_REDUCTION 256
*/
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 3

#define BLOCK_SIZE_X_BIS 16
#define BLOCK_SIZE_Y_BIS 16
#define BLOCK_SIZE_Z_BIS 3

#define BLOCK_SIZE_L2_X 16
#define BLOCK_SIZE_L2_Y 16
//third one neglected for the product rule 
#define BLOCK_SIZE_L2_Z 8

#define BLOCK_SIZE_REDUCTION 512

//#define N 256
#include "parameters.hpp"
//#include "algo_rohsa.hpp"
#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime_api.h>
#include <math.h>
#include <cmath>

#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]

//#include "gradient.cu"
template <typename T> void regularization_norm(T* beta, T* g_dev, T* b_params, T &f, int dim_x, int dim_y, int dim_v, parameters<T> &M, float* temps_kernel);
template <typename T> void update_array_f_dev_sort_fast_norm(T lambda_amp, T lambda_mu, T lambda_sig, T lambda_var_sig, T* array_f_dev, T* map_conv_amp_dev, T* map_conv_mu_dev, T* map_conv_sig_dev, T* map_image_sig_dev, int indice_x, int indice_y, int k, T* b_params_dev, float* temps);

extern double temps_test;

#endif