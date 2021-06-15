#include <vector>
#include <omp.h>
#include <stdio.h>
#include "gradient_norm.hpp"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

//#include "convolutions.hpp"

template <typename T> void f_g_cube_cuda_L_clean_templatized_norm(parameters<T> &M, T& f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs, float* temps_kernel);
template <typename T> void f_g_cube_cuda_L_clean_templatized_less_transfers_norm(parameters<T> &M, T& f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs, float* temps_kernel);
//void f_g_cube_cuda_L_clean_f(parameters<float> &M, float &f, float g[], int n, float beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<float>> &std_map, float* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs, double* temps_bis);