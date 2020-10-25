//PART 1/2
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define BLOCK_SIZE_Z 1

#define BLOCK_SIZE_L2_X 8
#define BLOCK_SIZE_L2_Y 8
#define BLOCK_SIZE_L2_Z 8

#define BLOCK_SIZE_REDUCTION 256

//PART 2/2
const int BLOCK_SIZE_S     = 16 ;
//POUR LA VERSION SEPARABLE AVEC MEMOIRE SHARED
#define NUMBER_COMPUTED_BLOCK 8
#define NUMBER_EDGE_HALO_BLOCK 1
#define BLOCK_SIZE_ROW_X 4
#define BLOCK_SIZE_ROW_Y 16

#define BLOCK_SIZE_COL_X 8
#define BLOCK_SIZE_COL_Y 16

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

void gradient_L_2_beta_parallel(double* deriv_dev, int* taille_deriv, int* taille_deriv_dev, double* beta_modif_dev, int* taille_beta_modif_dev, double* residual_dev, int* taille_residual_dev, double* std_map_dev, int* taille_std_map_dev, int n_gauss);
void gradient_L_3_parallel(double* deriv, int* taille_deriv, int product_taille_deriv, double* params, int* taille_params, int product_taille_params, double* residual, int* taille_residual, int product_residual, double* std_map, int* taille_std_map, int product_std_map, int n_gauss);
void compute_residual_and_f_parallel(double* array_f_dev, double* beta_dev, double* cube_dev, double* residual_dev, double* std_map_dev, int indice_x, int indice_y, int indice_v, int n_gauss);
void reduction_loop_parallel(double* array_in, double* d_array_f, int size_array);
void f_g_cube_parallel(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, double* cube_flattened);

void conv2D_GPU_sort(double* h_IMAGE, double* h_KERNEL, double* h_RESULTAT_GPU, long int image_x, long int image_y);
void dummyInstantiator_sort();
void prepare_for_convolution_sort(double* IMAGE, double* IMAGE_ext, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille, dim3 ThreadsParBlock);
void conv2D_GPU_all_sort(parameters& M, double* d_g, int n_beta, double* b_params, double* deriv_dev, double* beta_modif_dev, double* array_f_dev, long int image_x, long int image_y, int n_gauss, float temps_transfert, float temps_mirroirs);
void update_array_f_dev_sort(double lambda, double* array_f_dev, double* map_dev, int indice_x, int indice_y);
void update_array_f_dev_sort(double lambda, double* array_f_dev, double* map_image_dev, double* map_conv_dev, int indice_x, int indice_y, int k, double* b_params);