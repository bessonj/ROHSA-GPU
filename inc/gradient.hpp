#ifndef DEF_GRAD
#define DEF_GRAD
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
#define BLOCK_SIZE_Z 2

#define BLOCK_SIZE_X_BIS 16
#define BLOCK_SIZE_Y_BIS 16
#define BLOCK_SIZE_Z_BIS 2

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
void init_templates();
template <typename T> void reduction_loop(T* array_in, T* d_array_f, int size_array);
template <typename T> void reduction_loop_save(T* array_in, T* d_array_f, int size_array);
template <typename T> void gradient_L_2_beta(T* deriv, int* taille_deriv, int product_taille_deriv, T* params, int* taille_params, int product_taille_params, T* residual, int* taille_residual, int product_residual, T* std_map, int* taille_std_map, int product_std_map, int n_gauss);
template <typename T> T compute_residual_and_f(T* beta, int* taille_beta, int product_taille_beta, T* cube, int* taille_cube, int product_taille_cube, T* residual, int* taille_residual, int product_taille_residual, T* std_map, int* taille_std_map, int product_taille_std_map, int indice_x, int indice_y, int indice_v, int n_gauss);
template <typename T> void regularisation(T* beta, T* deriv, T* g, T* b_params, T &f, int dim_x, int dim_y, int dim_v, parameters<T> &M, double* temps_bis);
template <typename T> void prepare_for_convolution(T* d_IMAGE, T* d_IMAGE_ext, int dim_x, int dim_y);
template <typename T> void conv_twice_and_copy(T* d_IMAGE_amp_ext, T* d_conv_amp, T* d_conv_conv_amp, int image_x, int image_y);
template <typename T> void update_array_f_dev_sort(T lambda, T lambda_var, T* array_f_dev, T* map_image_dev, T* map_conv_dev, int indice_x, int indice_y, int k, T* b_params);
template <typename T> void update_array_f_dev_sort(T lambda, T* array_f_dev, T* map_dev, int indice_x, int indice_y);

extern double temps_test;

#endif