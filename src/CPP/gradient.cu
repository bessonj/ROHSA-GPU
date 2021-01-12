#include "gradient.hpp"
#include "kernels_for_hybrid.cu"
#include "kernel_gradient.cuh"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <omp.h>

void init_templates()
{
/*
  reduce_last_in_one_thread<double><<<1,1>>>(NULL,NULL,0);
  gradient_kernel_2_beta_with_INDEXING<double><<<1,1>>>(NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,0);
  gradient_kernel_2_beta_with_INDEXING<float><<<1,1>>>(NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,0);
*/
}

template <typename T> 
void gradient_L_2_beta(T* deriv, int* taille_deriv, int product_taille_deriv, T* params, int* taille_params, int product_taille_params, T* residual, int* taille_residual, int product_residual, T* std_map, int* taille_std_map, int product_std_map, int n_gauss)
{
   T* params_dev = NULL;
   T* deriv_dev = NULL;
   T* residual_dev = NULL;
   T* std_map_dev = NULL;

   int* taille_params_dev = NULL;
   int* taille_deriv_dev = NULL;
   int* taille_residual_dev = NULL;
   int* taille_std_map_dev = NULL;

   checkCudaErrors(cudaMalloc(&deriv_dev, product_taille_deriv*sizeof(T)));
   checkCudaErrors(cudaMalloc(&residual_dev, product_residual*sizeof(T)));
   checkCudaErrors(cudaMalloc(&params_dev, product_taille_params*sizeof(T)));
   checkCudaErrors(cudaMalloc(&std_map_dev, product_std_map*sizeof(T)));

   checkCudaErrors(cudaMalloc(&taille_deriv_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_params_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
   
   checkCudaErrors(cudaMemcpy(deriv_dev, deriv, product_taille_deriv*sizeof(T), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(residual_dev, residual, product_residual*sizeof(T), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(T), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_std_map*sizeof(T), cudaMemcpyHostToDevice));
   
   checkCudaErrors(cudaMemcpy(taille_deriv_dev, taille_deriv, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map,2*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

   dim3 Dg, Db;

    Db.x = BLOCK_SIZE_X; //x
    Db.y = BLOCK_SIZE_Y; //y
    Db.z = BLOCK_SIZE_Z; //gaussiennes
        //deriv      --> (3g,y,x)  --> (z,y,x)
        //params     --> (3g,y,x)  --> (z,y,x)
    Dg.x = ceil(taille_deriv[2]/T(BLOCK_SIZE_X));
    Dg.y = ceil(taille_deriv[1]/T(BLOCK_SIZE_Y));
    Dg.z = n_gauss;

  cudaDeviceSynchronize();
  gradient_kernel_2_beta_with_INDEXING<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, params_dev, taille_params_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaMemcpy(deriv, deriv_dev, product_taille_deriv*sizeof(T), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
  checkCudaErrors(cudaFree(deriv_dev));
  checkCudaErrors(cudaFree(taille_deriv_dev));
  checkCudaErrors(cudaFree(params_dev));
  checkCudaErrors(cudaFree(taille_params_dev));
  checkCudaErrors(cudaFree(std_map_dev));
  checkCudaErrors(cudaFree(taille_std_map_dev));
  checkCudaErrors(cudaFree(residual_dev));
  checkCudaErrors(cudaFree(taille_residual_dev));
}













//                        map_norm_dev        d_array_f
template <typename T> 
void reduction_loop(T* array_in, T* d_array_f, int size_array){
    int GRID_SIZE_REDUCTION = int(ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION)));
    int N = ceil(log(T(size_array))/log(T(BLOCK_SIZE_REDUCTION)));
//    reduce_last_in_one_thread<T><<<1,1>>>(array_in, d_array_f, size_array);

    checkCudaErrors(cudaDeviceSynchronize());

    if (N==1){
        int size_array_out_kernel = ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION));
        int copy_dev_blocks = ceil(T(size_array_out_kernel)/T(BLOCK_SIZE_REDUCTION));
        T* array_out_kernel=NULL;
        checkCudaErrors(cudaMalloc(&array_out_kernel, size_array_out_kernel*sizeof(T)));

        checkCudaErrors(cudaDeviceSynchronize());

        sum_reduction<T><<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in, array_out_kernel, size_array);

        checkCudaErrors(cudaDeviceSynchronize());

        cpy_first_num_dev<T><<<1,1>>>( array_out_kernel, d_array_f);
        cudaFree(array_out_kernel);

    } else{
        int size_array_out_kernel = ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION));
        T* array_out_kernel=NULL;
        checkCudaErrors(cudaMalloc(&array_out_kernel, size_array_out_kernel*sizeof(T)));

        sum_reduction<T><<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in, array_out_kernel, size_array);

        checkCudaErrors(cudaDeviceSynchronize());

        T* array_in_copied_2;
        checkCudaErrors(cudaMalloc(&array_in_copied_2, size_array_out_kernel*sizeof(T)));

        int copy_dev_blocks = ceil(T(size_array_out_kernel)/T(BLOCK_SIZE_REDUCTION));
        copy_dev<T><<< copy_dev_blocks , BLOCK_SIZE_REDUCTION >>>(array_out_kernel, array_in_copied_2, size_array_out_kernel);

        cudaFree(array_out_kernel);

        checkCudaErrors(cudaDeviceSynchronize());

        T size_array_out_kernel_2 = ceil(T(size_array)/T(pow(BLOCK_SIZE_REDUCTION,2)));
        T* array_out_kernel_2=NULL;
        checkCudaErrors(cudaMalloc(&array_out_kernel_2, size_array_out_kernel_2*sizeof(T)));

        sum_reduction<T><<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in_copied_2, array_out_kernel_2, size_array_out_kernel);

        checkCudaErrors(cudaDeviceSynchronize());

        if(N>2){
        reduce_last_in_one_thread<T><<<1,1>>>(array_out_kernel_2, d_array_f, size_array_out_kernel_2);
        }
        else{
        cpy_first_num_dev<T><<<1,1>>>( array_out_kernel_2, d_array_f);
        }
        cudaFree(array_in_copied_2);
        cudaFree(array_out_kernel_2);
    }
}	






template <typename T> 
T compute_residual_and_f(T* beta, int* taille_beta, int product_taille_beta, T* cube, int* taille_cube, int product_taille_cube, T* residual, int* taille_residual, int product_taille_residual, T* std_map, int* taille_std_map, int product_taille_std_map, int indice_x, int indice_y, int indice_v, int n_gauss)
{
    T* beta_dev = NULL;
    T* cube_dev = NULL;
    T* residual_dev = NULL;
    T* std_map_dev = NULL;

    int* taille_beta_dev = NULL;
    int* taille_cube_dev = NULL;
    int* taille_residual_dev = NULL;
    int* taille_std_map_dev = NULL;

    checkCudaErrors(cudaMalloc(&beta_dev, product_taille_beta*sizeof(T)));
    checkCudaErrors(cudaMalloc(&residual_dev, product_taille_residual*sizeof(T)));
    checkCudaErrors(cudaMalloc(&cube_dev, product_taille_cube*sizeof(T)));
    checkCudaErrors(cudaMalloc(&std_map_dev, product_taille_std_map*sizeof(T)));

    checkCudaErrors(cudaMalloc(&taille_cube_dev, 3*sizeof(int)));
    checkCudaErrors(cudaMalloc(&taille_beta_dev, 3*sizeof(int)));
    checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
    checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
    
    checkCudaErrors(cudaMemcpy(beta_dev, beta, product_taille_beta*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(residual_dev, residual, product_taille_residual*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(cube_dev, cube, product_taille_cube*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_taille_std_map*sizeof(T), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMemcpy(taille_cube_dev, taille_cube, 3*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_beta_dev, taille_beta, 3*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map,2*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

    dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X_BIS; //
    Db.y = BLOCK_SIZE_Y_BIS; //
    Db.z = BLOCK_SIZE_Z_BIS; //

    Dg.x = ceil(T(indice_x)/T(BLOCK_SIZE_X_BIS));
    Dg.y = ceil(T(indice_y)/T(BLOCK_SIZE_Y_BIS));
    Dg.z = ceil(T(indice_v)/T(BLOCK_SIZE_Z_BIS));

    kernel_residual<T><<<Dg,Db>>>(beta_dev, cube_dev, residual_dev,indice_x, indice_y, indice_v, n_gauss);

    checkCudaErrors(cudaMemcpy(residual, residual_dev, product_taille_residual*sizeof(T), cudaMemcpyDeviceToHost));

    dim3 Dg_L2, Db_L2;
    Db_L2.x = BLOCK_SIZE_L2_X;
    Db_L2.y = BLOCK_SIZE_L2_Y;
    Db_L2.z = 1;
    Dg_L2.x = ceil(indice_x/T(BLOCK_SIZE_L2_X));
    Dg_L2.y = ceil(indice_y/T(BLOCK_SIZE_L2_Y));
    Dg_L2.z = 1;

    T* map_norm_dev = NULL;
    checkCudaErrors(cudaMalloc(&map_norm_dev, indice_x*indice_y*sizeof(T)));

    kernel_norm_map_boucle_v<T><<<Dg_L2, Db_L2>>>(map_norm_dev, residual_dev, taille_residual_dev, std_map_dev, indice_x, indice_y, indice_v);

    T* d_array_f=NULL;
    checkCudaErrors(cudaMalloc(&d_array_f, 1*sizeof(T))); // ERREUR ICI
    reduction_loop<T>(map_norm_dev, d_array_f, indice_x*indice_y);
    T* array_f = (T*)malloc(1*sizeof(T));
    checkCudaErrors(cudaMemcpy(array_f, d_array_f, 1*sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(residual, residual_dev, product_taille_residual*sizeof(T), cudaMemcpyDeviceToHost));
    T sum1 = array_f[0];
    free(array_f);

    checkCudaErrors(cudaFree(d_array_f));
    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(beta_dev));
    checkCudaErrors(cudaFree(taille_beta_dev));
    checkCudaErrors(cudaFree(cube_dev));
    checkCudaErrors(cudaFree(taille_cube_dev));
    checkCudaErrors(cudaFree(std_map_dev));
    checkCudaErrors(cudaFree(taille_std_map_dev));
    checkCudaErrors(cudaFree(residual_dev));
    checkCudaErrors(cudaFree(taille_residual_dev));

    return sum1;
  }

template double compute_residual_and_f<double>(double*, int*, int, double*, int*, int, double*, int*, int, double*, int*, int, int, int, int, int);
template void gradient_L_2_beta<double>(double*, int*, int, double*, int*, int, double*, int*, int, double*, int*, int, int);
template void reduction_loop<double>(double*, double*, int);