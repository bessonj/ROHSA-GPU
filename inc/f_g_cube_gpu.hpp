//PART 1/2
#define BLOCK_SIZE_X_GRAD1 32
#define BLOCK_SIZE_Y_GRAD1 32
#define BLOCK_SIZE_Z_GRAD1 1

#define BLOCK_SIZE_X_SORT 8
#define BLOCK_SIZE_Y_SORT 8
#define BLOCK_SIZE_Z_SORT 16

#define BLOCK_SIZE_X_2D_SORT 32
#define BLOCK_SIZE_Y_2D_SORT 32
#define BLOCK_SIZE_Z_2D_SORT 1

#define BLOCK_SIZE_X_2D_SORT_ 32
#define BLOCK_SIZE_Y_2D_SORT_ 32
#define BLOCK_SIZE_Z_2D_SORT_ 1

/*
#define BLOCK_SIZE_X_2D_SORT_ 4
#define BLOCK_SIZE_Y_2D_SORT_ 4
#define BLOCK_SIZE_Z_2D_SORT_ 1
*/

#define BLOCK_SIZE_REDUCTION 256

//#define N 256
#include "parameters.hpp"
//#include "algo_rohsa.hpp"
#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime_api.h>
#include <math.h>
#include <cmath>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdio.h>      /* printf */
#include <math.h>       /* ceil */
#include <omp.h>

#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]

/**
 * @file f_g_cube_gpu.cu
 * @brief These are the functions involved in f_g_cube.
 *
 *  
 *
 */

void gradient_L_3_parallel(double* deriv, int* taille_deriv, int product_taille_deriv, double* params, int* taille_params, int product_taille_params, double* residual, int* taille_residual, int product_residual, double* std_map, int* taille_std_map, int product_std_map, int n_gauss);
void f_g_cube_parallel(parameters &M, double &f, double* g, int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, double* cube_flattened, double temp_conv, double temps_deriv, double temps_tableaux, double temps_res_f);
//void f_g_cube_parallel_lib(const parameters &M, double &f, double* d_g, const int n, double* beta_dev, const int indice_v, const int indice_y, const int indice_x, double* std_map_dev, double* cube_flattened_dev, double* temps);
template <typename T> void f_g_cube_parallel_lib(const parameters &M, T &f, T* g_dev, const int n, T* beta_dev, const int indice_v, const int indice_y, const int indice_x, T* std_map, T* cube_flattened, T* temps);

void conv2D_GPU_sort(double* h_IMAGE, double* h_KERNEL, double* h_RESULTAT_GPU, long int image_x, long int image_y);
void dummyInstantiator_sort(); //!< Initialize the template functions


template <typename T> void compute_residual_and_f_parallel(T* array_f_dev, T* beta_dev, T* cube_dev, T* residual_dev, T* std_map_dev, int indice_x, int indice_y, int indice_v, int n_gauss);
template <typename T> void reduction_loop_parallel(T* array_in, T* d_array_f, int size_array);
template <typename T> void gradient_L_2_beta_parallel(T* deriv_dev, int* taille_deriv, int* taille_deriv_dev, T* beta_modif_dev, int* taille_beta_modif_dev, T* residual_dev, int* taille_residual_dev, T* std_map_dev, int* taille_std_map_dev, int n_gauss);
template <typename T> void conv2D_GPU_all_sort(const parameters &M, T* d_g, const int n_beta, T lambda_var_sig, T* b_params_dev, T* deriv_dev, T* beta_modif_dev, T* array_f_dev, const int image_x, const int image_y, const int n_gauss, float temps_transfert, float temps_mirroirs);
template <typename T> void update_array_f_dev_sort(T lambda, T lambda_var, T* array_f_dev, T* map_image_dev, T* map_conv_dev, int indice_x, int indice_y, int k, T* b_params);
template <typename T> void update_array_f_dev_sort(T lambda, T* array_f_dev, T* map_dev, int indice_x, int indice_y);
template <typename T> void conv_twice_and_copy_sort(T* d_IMAGE_amp_ext, T* d_conv_amp, T* d_conv_conv_amp, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille, dim3 ThreadsParBlock, dim3 BlocksParGrille_frame, dim3 ThreadsParBlock_frame);
template <typename T> void prepare_for_convolution_sort(T* d_IMAGE_amp, T* d_IMAGE_amp_ext, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille_frame, dim3 ThreadsParBlock_frame);
