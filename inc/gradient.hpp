#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define BLOCK_SIZE_Z 32

//#define N 256
#include "model.hpp"
//#include "gradient.cu"
void gradient(double* dF_over_dB, int* taille_dF_over_dB, int product_taille_dF_over_dB, double* params, int* taille_params, int product_taille_params, model &M);
//extern void gradient(double* dF_over_dB, int* taille_dF_over_dB, int product_taille_dF_over_dB, double* params, int* taille_params, int product_taille_params, model &M);

/*
__host__ __device__ double value_at_4d_index(double* flattened_4d_tab, int* coordinates, int* taille_tab_4d)
__host__ __device__ void set_at_4d_index(double* flattened_4d_tab, int* coordinates, int* taille_tab_4d, double value);
__host__ __device__ int flattened_index_4d(int* coordinates, int* taille_tab_4d);
*/
