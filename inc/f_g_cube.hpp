#include <vector>
#include <omp.h>
#include "gradient.hpp"
#include "convolutions.hpp"


template <typename T> void f_g_cube_fast_unidimensional(parameters<T> &M, T &f, T* g, int n, T* cube, std::vector<std::vector<std::vector<T>>>& cube_for_cache, T* beta, int indice_v, int indice_y, int indice_x, T* std_map, double* temps);	
template <typename T> void f_g_cube_fast_clean(parameters<T> &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, double* temps);
template <typename T> void f_g_cube_not_very_fast_clean(parameters<T> &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, double* temps);
template <typename T> void f_g_cube_fast_clean_optim_CPU_lib(parameters<T> &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T** assist_buffer, double* temps);
template <typename T> void f_g_cube_cuda_L_clean(parameters<T> &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs);
template <typename T> void f_g_cube_cuda_L_clean_lib(parameters<T> &M, T &f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs);

template <typename T> void convolution_2D_mirror_flat(const parameters<T> &M, T* image, T* conv, int dim_y, int dim_x, int dim_k);
template <typename T> void convolution_2D_mirror(const parameters<T> &M, const std::vector<std::vector<T>> &image, std::vector<std::vector<T>> &conv, int dim_y, int dim_x, int dim_k);

template <typename T> void one_D_to_three_D_same_dimensions(T* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v);
template <typename T> void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v);

template <typename T> T myfunc_spec(std::vector<T> &residual);
template <typename T> void myresidual(std::vector<T> & params, std::vector<T> &line, std::vector<T> &residual, int n_gauss_i);