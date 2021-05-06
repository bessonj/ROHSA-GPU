#include "gradient.hpp"
#include "kernels_for_hybrid.cu"
#include "kernel_gradient.cuh"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <omp.h>

#define Nb_time_mes 10

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
    Dg.z = ceil(taille_deriv[3]/T(BLOCK_SIZE_Z));

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
    int N = ceil(log(T(size_array))/log(T(BLOCK_SIZE_REDUCTION)));
//    printf("N = %d\n", N);

    int GRID_SIZE_REDUCTION = int(ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION)));
    int size_array_out_kernel = ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION));
    checkCudaErrors(cudaDeviceSynchronize());
    T* array_out_kernel=NULL;
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc(&array_out_kernel, size_array_out_kernel*sizeof(T)));    
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemset(array_out_kernel, 0., size_array_out_kernel*sizeof(T)));
    checkCudaErrors(cudaDeviceSynchronize());

    bool reduction_in_one_thread = false;
    if(reduction_in_one_thread){
      reduce_last_in_one_thread<T><<<1,1>>>(array_in, d_array_f, size_array);
    }else{
      sum_reduction<T><<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in, array_out_kernel, size_array);
      checkCudaErrors(cudaDeviceSynchronize());
      if(size_array_out_kernel>1){
        reduce_last_in_one_thread<T><<<1,1>>>(array_out_kernel, d_array_f, size_array_out_kernel);
      }else{
        cpy_first_num_dev<T><<<1,1>>>( array_out_kernel, d_array_f);
      }
    }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(array_out_kernel));

}	

template <typename T> 
void reduction_loop_save(T* array_in, T* d_array_f, int size_array){
    int GRID_SIZE_REDUCTION = int(ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION)));
    int N = ceil(log(T(size_array))/log(T(BLOCK_SIZE_REDUCTION)));
    printf("N = %d\n", N);
    
    reduce_last_in_one_thread<T><<<1,1>>>(array_in, d_array_f, size_array);
    
/*
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
  */
}	






template <typename T> 
T compute_residual_and_f(T* beta, int* taille_beta, int product_taille_beta, T* cube, int* taille_cube, int product_taille_cube, T* residual, int* taille_residual, int product_taille_residual, T* std_map, int* taille_std_map, int product_taille_std_map, int indice_x, int indice_y, int indice_v, int n_gauss)
{
    T* beta_dev = NULL;
    T* cube_dev = NULL;
    T* residual_dev = NULL;
    T* std_map_dev = NULL;

    checkCudaErrors(cudaMalloc(&beta_dev, product_taille_beta*sizeof(T)));
    checkCudaErrors(cudaMalloc(&residual_dev, product_taille_residual*sizeof(T)));
    checkCudaErrors(cudaMalloc(&cube_dev, product_taille_cube*sizeof(T)));
    checkCudaErrors(cudaMalloc(&std_map_dev, product_taille_std_map*sizeof(T)));

    checkCudaErrors(cudaMemcpy(beta_dev, beta, product_taille_beta*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(residual_dev, residual, product_taille_residual*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(cube_dev, cube, product_taille_cube*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_taille_std_map*sizeof(T), cudaMemcpyHostToDevice));
    
    dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X_BIS; //
    Db.y = BLOCK_SIZE_Y_BIS; //
    Db.z = BLOCK_SIZE_Z_BIS; //

    Dg.x = ceil(T(indice_x)/T(BLOCK_SIZE_X_BIS));
    Dg.y = ceil(T(indice_y)/T(BLOCK_SIZE_Y_BIS));
    Dg.z = ceil(T(indice_v)/T(BLOCK_SIZE_Z_BIS));

/*
    T* cube_reconstructed = NULL;
    checkCudaErrors(cudaMalloc(&cube_reconstructed, indice_x*indice_y*indice_v*sizeof(cube_reconstructed[0])));
  	checkCudaErrors(cudaMemset(cube_reconstructed, 0., indice_x*indice_y*indice_v*sizeof(cube_reconstructed[0])));
    checkCudaErrors(cudaDeviceSynchronize());
    kernel_hypercube_reconstructed<T><<<Dg,Db>>>(beta_dev, cube_reconstructed, indice_x, indice_y, indice_v, n_gauss);
    checkCudaErrors(cudaDeviceSynchronize());
    kernel_residual_simple_difference<T><<<Dg,Db>>>(cube_dev, cube_reconstructed, residual_dev, indice_x, indice_y, indice_v);
    checkCudaErrors(cudaFree(cube_reconstructed));
*/

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

    kernel_norm_map_boucle_v<T><<<Dg_L2, Db_L2>>>(map_norm_dev, residual_dev, std_map_dev, indice_x, indice_y, indice_v);

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
    checkCudaErrors(cudaFree(cube_dev));
    checkCudaErrors(cudaFree(std_map_dev));
    checkCudaErrors(cudaFree(residual_dev));

    return sum1;
  }


template <typename T> 
T compute_residual_and_f_cheap_memory_trick(T* beta, int* taille_beta, int product_taille_beta, T* cube, int* taille_cube, int product_taille_cube, T* residual, int* taille_residual, int product_taille_residual, T* std_map, int* taille_std_map, int product_taille_std_map, int indice_x, int indice_y, int indice_v, int n_gauss)
{
    T* beta_dev = NULL;
    T* residual_dev = NULL;
    T* std_map_dev = NULL;

    checkCudaErrors(cudaMalloc(&beta_dev, product_taille_beta*sizeof(T)));
    checkCudaErrors(cudaMalloc(&residual_dev, product_taille_residual*sizeof(T)));
    checkCudaErrors(cudaMalloc(&std_map_dev, product_taille_std_map*sizeof(T)));

    checkCudaErrors(cudaMemcpy(beta_dev, beta, product_taille_beta*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(residual_dev, cube, product_taille_residual*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_taille_std_map*sizeof(T), cudaMemcpyHostToDevice));
    
    dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X_BIS; //
    Db.y = BLOCK_SIZE_Y_BIS; //
    Db.z = BLOCK_SIZE_Z_BIS; //

    Dg.x = ceil(T(indice_x)/T(BLOCK_SIZE_X_BIS));
    Dg.y = ceil(T(indice_y)/T(BLOCK_SIZE_Y_BIS));
    Dg.z = ceil(T(indice_v)/T(BLOCK_SIZE_Z_BIS));


    kernel_residual_cheap_memory_trick<T><<<Dg,Db>>>(beta_dev, residual_dev,indice_x, indice_y, indice_v, n_gauss);

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

    kernel_norm_map_boucle_v<T><<<Dg_L2, Db_L2>>>(map_norm_dev, residual_dev, std_map_dev, indice_x, indice_y, indice_v);

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
    checkCudaErrors(cudaFree(std_map_dev));
    checkCudaErrors(cudaFree(residual_dev));

    return sum1;
  }


template <typename T> 
void regularisation(T* beta, T* deriv, T* g, T* b_params, T &f, int dim_x, int dim_y, int dim_v, parameters<T> &M, double* temps_bis){
  temps_bis[0] = 2.;
	int n_beta = (3*M.n_gauss*dim_x*dim_y)+M.n_gauss;
  T* b_params_dev = NULL;
  T* beta_dev = NULL;
  T* deriv_dev = NULL;
  checkCudaErrors(cudaMalloc(&b_params_dev, M.n_gauss*sizeof(T)));
  checkCudaErrors(cudaMalloc(&beta_dev, n_beta*sizeof(T)));
  checkCudaErrors(cudaMalloc(&deriv_dev, dim_x*dim_y*3*M.n_gauss*sizeof(T)));
  checkCudaErrors(cudaMemcpy(b_params_dev, b_params, M.n_gauss*sizeof(T), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(beta_dev, beta, n_beta*sizeof(T), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(deriv_dev, deriv, dim_x*dim_y*3*M.n_gauss*sizeof(T), cudaMemcpyHostToDevice));
  T* array_f_dev = NULL;
  checkCudaErrors(cudaMalloc(&array_f_dev, 1*sizeof(T)));
  T* array_f = NULL;
  array_f = (T*)malloc(1*sizeof(T));
//  cpy_first_num_dev<T><<<1,1>>>(array_f array_f_dev);
  checkCudaErrors(cudaMemcpy(array_f_dev, array_f, 1*sizeof(T), cudaMemcpyHostToDevice));

  T* d_g = NULL;
  checkCudaErrors(cudaMalloc(&d_g, n_beta*sizeof(T)));
  checkCudaErrors(cudaMemset(d_g, 0., n_beta*sizeof(T)));

  T* d_IMAGE_amp = NULL;
  checkCudaErrors(cudaMalloc(&d_IMAGE_amp, dim_x*dim_y*sizeof(T)));
  T* d_IMAGE_mu = NULL;
  checkCudaErrors(cudaMalloc(&d_IMAGE_mu, dim_x*dim_y*sizeof(T)));
  T* d_IMAGE_sig = NULL;
  checkCudaErrors(cudaMalloc(&d_IMAGE_sig, dim_x*dim_y*sizeof(T)));

  dim3 Dg_2D, Db_2D;
  Db_2D.x = BLOCK_SIZE_L2_X;
  Db_2D.y = BLOCK_SIZE_L2_Y;
  Db_2D.z = 1;
  Dg_2D.x = ceil(dim_x/T(BLOCK_SIZE_L2_X));
  Dg_2D.y = ceil(dim_y/T(BLOCK_SIZE_L2_Y));
  Dg_2D.z = 1;

  checkCudaErrors(cudaDeviceSynchronize());

  for(int k = 0; k<M.n_gauss; k++){
    cudaEvent_t record_event[Nb_time_mes];
    float time_msec[Nb_time_mes];
    for (int i=0;i<Nb_time_mes;i++){
        checkCudaErrors(cudaEventCreate(record_event+i));   
    }
    // Record the start event
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[0], NULL));

    //Slices d_IMAGE_* from beta_dev
    parameter_maps_sliced_from_beta_sort<T><<<Dg_2D, Db_2D>>>(beta_dev, d_IMAGE_amp, d_IMAGE_mu, d_IMAGE_sig, dim_x, dim_y, k);

    T* d_EXT_amp = NULL;
    checkCudaErrors(cudaMalloc(&d_EXT_amp, (dim_x+4)*(dim_y+4)*sizeof(T)));
    T* d_EXT_mu = NULL;
    checkCudaErrors(cudaMalloc(&d_EXT_mu, (dim_x+4)*(dim_y+4)*sizeof(T)));
    T* d_EXT_sig = NULL;
    checkCudaErrors(cudaMalloc(&d_EXT_sig, (dim_x+4)*(dim_y+4)*sizeof(T)));

    T* d_CONV_amp = NULL;
    checkCudaErrors(cudaMalloc(&d_CONV_amp, (dim_x+4)*(dim_y+4)*sizeof(T)));
    T* d_CONV_mu = NULL;
    checkCudaErrors(cudaMalloc(&d_CONV_mu, (dim_x+4)*(dim_y+4)*sizeof(T)));
    T* d_CONV_sig = NULL;
    checkCudaErrors(cudaMalloc(&d_CONV_sig, (dim_x+4)*(dim_y+4)*sizeof(T)));

    T* d_CONV_CONV_amp = NULL;
    checkCudaErrors(cudaMalloc(&d_CONV_CONV_amp, (dim_x+4)*(dim_y+4)*sizeof(T)));
    T* d_CONV_CONV_mu = NULL;
    checkCudaErrors(cudaMalloc(&d_CONV_CONV_mu, (dim_x+4)*(dim_y+4)*sizeof(T)));
    T* d_CONV_CONV_sig = NULL;
    checkCudaErrors(cudaMalloc(&d_CONV_CONV_sig, (dim_x+4)*(dim_y+4)*sizeof(T)));

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventSynchronize(record_event[1]));    
    checkCudaErrors(cudaEventRecord(record_event[1], NULL));

    //Sets d_EXT_* arrays to 0.
    prepare_for_convolution<T>(d_IMAGE_amp, d_EXT_amp, dim_x, dim_y);
    prepare_for_convolution<T>(d_IMAGE_mu, d_EXT_mu, dim_x, dim_y);
    prepare_for_convolution<T>(d_IMAGE_sig, d_EXT_sig, dim_x, dim_y);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    checkCudaErrors(cudaDeviceSynchronize());

    //Does the convolutions for each gaussian parameters of the k-th gaussian index
    conv_twice_and_copy<T>(d_EXT_amp, d_CONV_amp, d_CONV_CONV_amp, dim_x, dim_y);
    conv_twice_and_copy<T>(d_EXT_mu, d_CONV_mu, d_CONV_CONV_mu, dim_x, dim_y);
    conv_twice_and_copy<T>(d_EXT_sig, d_CONV_sig, d_CONV_CONV_sig, dim_x, dim_y);

    checkCudaErrors(cudaDeviceSynchronize());

/*
    display_size<<<1,1>>>(d_CONV_CONV_amp, dim_x*dim_y);
    display_size<<<1,1>>>(d_CONV_CONV_mu, dim_x*dim_y);
    display_size<<<1,1>>>(d_CONV_CONV_sig, dim_x*dim_y);
    display_size<<<1,1>>>(d_CONV_amp, dim_x*dim_y);
    display_size<<<1,1>>>(d_CONV_mu, dim_x*dim_y);
    display_size<<<1,1>>>(d_CONV_sig, dim_x*dim_y);
    display_size<<<1,1>>>(d_IMAGE_amp, dim_x*dim_y);
    display_size<<<1,1>>>(d_IMAGE_mu, dim_x*dim_y);
    display_size<<<1,1>>>(d_IMAGE_sig, dim_x*dim_y);
    display_size<<<1,1>>>(array_f_dev, 1);
*/
    checkCudaErrors(cudaDeviceSynchronize());



    
    checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));
    checkCudaErrors(cudaDeviceSynchronize());

    update_array_f_dev_sort<T>(M.lambda_amp, array_f_dev, d_CONV_amp, dim_x, dim_y);
    checkCudaErrors(cudaDeviceSynchronize());
//    display_size<<<1,1>>>(array_f_dev, 1);
    checkCudaErrors(cudaDeviceSynchronize());

    update_array_f_dev_sort<T>(M.lambda_mu, array_f_dev, d_CONV_mu, dim_x, dim_y);
    checkCudaErrors(cudaDeviceSynchronize());
//    display_size<<<1,1>>>(array_f_dev, 1);
    checkCudaErrors(cudaDeviceSynchronize());
    update_array_f_dev_sort<T>(M.lambda_sig, M.lambda_var_sig, array_f_dev, d_IMAGE_sig, d_CONV_sig, dim_x, dim_y, k, b_params_dev);
    checkCudaErrors(cudaDeviceSynchronize());
//    display_size<<<1,1>>>(array_f_dev, 1);
//    display_size<<<1,1>>>(b_params_dev, M.n_gauss);
//    exit(0);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[4], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[4]));

    dim3 Dg_ud, Db_ud;
    Db_ud.x = BLOCK_SIZE_L2_X;
    Db_ud.y = BLOCK_SIZE_L2_Y;
    Db_ud.z = 1;
    Dg_ud.x = ceil(T(dim_x)/T(BLOCK_SIZE_L2_X));
    Dg_ud.y = ceil(T(dim_y)/T(BLOCK_SIZE_L2_Y));
    Dg_ud.z = 1;

    checkCudaErrors(cudaDeviceSynchronize());

    double temps_test__ = omp_get_wtime();

    kernel_update_deriv_conv_conv_sort<T><<<Dg_ud,Db_ud>>>(deriv_dev, M.lambda_amp, M.lambda_mu, M.lambda_sig, M.lambda_var_sig, d_CONV_CONV_amp, d_CONV_CONV_mu, d_CONV_CONV_sig, d_IMAGE_sig, b_params_dev, dim_y, dim_x, k);

    temps_test += omp_get_wtime() - temps_test__;
    
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[5], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[5]));

    checkCudaErrors(cudaDeviceSynchronize());

    T* d_image_sigma_reduc = NULL;
    checkCudaErrors(cudaMalloc(&d_image_sigma_reduc, 1*sizeof(T)));
  	checkCudaErrors(cudaMemset(d_image_sigma_reduc, 0., 1*sizeof(d_image_sigma_reduc[0])));
    reduction_loop<T>(d_IMAGE_sig, d_image_sigma_reduc, dim_y*dim_x);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[6], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[6]));

    checkCudaErrors(cudaDeviceSynchronize());

    kernel_conv_g_reduction_sort<T><<<1,1>>>(n_beta+1, d_g, d_image_sigma_reduc, M.lambda_var_sig, M.n_gauss, b_params_dev, k, dim_x, dim_y);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_image_sigma_reduc));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[7], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[7]));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[8], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[8]));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[9], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[9]));

    for(int i = 0; i<Nb_time_mes-1; i++){
      checkCudaErrors(cudaEventElapsedTime(time_msec+i, record_event[i], record_event[i+1]));
    }
    for(int i = 0; i<Nb_time_mes-1; i++){
      time_msec[Nb_time_mes-1]+=time_msec[i];
    }

    for(int i = 0; i<Nb_time_mes; i++){
      temps_bis[i+1] += double(time_msec[i]/1000.);
    }



    checkCudaErrors(cudaDeviceSynchronize());

//init_extended_array_sort(T* d_IMAGE_amp, T* d_EXT_amp, int dim_x, int dim_y){

    checkCudaErrors(cudaFree(d_CONV_amp));
    checkCudaErrors(cudaFree(d_CONV_mu));
    checkCudaErrors(cudaFree(d_CONV_sig));
    checkCudaErrors(cudaFree(d_CONV_CONV_amp));
    checkCudaErrors(cudaFree(d_CONV_CONV_mu));
    checkCudaErrors(cudaFree(d_CONV_CONV_sig));
    checkCudaErrors(cudaFree(d_EXT_amp));
    checkCudaErrors(cudaFree(d_EXT_mu));
    checkCudaErrors(cudaFree(d_EXT_sig));
  }

//  display_size<<<1,1>>>(d_g, n_beta);

  checkCudaErrors(cudaDeviceSynchronize());
//  exit(0);

  checkCudaErrors(cudaMemcpy(g, d_g, n_beta*sizeof(T), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaFree(d_g));
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(array_f, array_f_dev, 1*sizeof(T), cudaMemcpyDeviceToHost));
  f += array_f[0];

  checkCudaErrors(cudaFree(array_f_dev));
  free(array_f);

  checkCudaErrors(cudaMemcpy(deriv, deriv_dev, dim_x*dim_y*3*M.n_gauss*sizeof(T), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(b_params_dev));
  checkCudaErrors(cudaFree(beta_dev));
  checkCudaErrors(cudaFree(deriv_dev));
  checkCudaErrors(cudaFree(d_IMAGE_amp));
  checkCudaErrors(cudaFree(d_IMAGE_mu));
  checkCudaErrors(cudaFree(d_IMAGE_sig));
}


template <typename T> 
void prepare_for_convolution(T* d_IMAGE, T* d_IMAGE_ext, int dim_x, int dim_y){
    dim3 Dg_2D, Db_2D;
    Db_2D.x = BLOCK_SIZE_L2_X;
    Db_2D.y = BLOCK_SIZE_L2_Y;
    Db_2D.z = 1;
    Dg_2D.x = ceil((dim_x+4)/T(BLOCK_SIZE_L2_X));
    Dg_2D.y = ceil((dim_y+4)/T(BLOCK_SIZE_L2_Y));
    Dg_2D.z = 1;
  	checkCudaErrors(cudaMemset(d_IMAGE_ext, 0., (dim_x+4)*(dim_y+4)*sizeof(d_IMAGE_ext[0])));
    checkCudaErrors(cudaDeviceSynchronize());
    init_extended_array_sort<T><<<Dg_2D,Db_2D>>>(d_IMAGE, d_IMAGE_ext, dim_x, dim_y);
    checkCudaErrors(cudaDeviceSynchronize());
    extension_mirror_gpu_sort_bis<T><<<Dg_2D,Db_2D>>>(d_IMAGE, d_IMAGE_ext, dim_x, dim_y);
}

template <typename T> void conv_twice_and_copy(T* d_IMAGE_amp_ext, T* d_conv_amp, T* d_conv_conv_amp, int image_x, int image_y)
{
    dim3 Dg_2D_EXT, Db_2D_EXT;
    Db_2D_EXT.x = BLOCK_SIZE_L2_X;
    Db_2D_EXT.y = BLOCK_SIZE_L2_Y;
    Db_2D_EXT.z = 1;
    Dg_2D_EXT.x = ceil((image_x+4)/T(BLOCK_SIZE_L2_X));
    Dg_2D_EXT.y = ceil((image_y+4)/T(BLOCK_SIZE_L2_Y));
    Dg_2D_EXT.z = 1;

    dim3 Dg_2D, Db_2D;
    Db_2D.x = BLOCK_SIZE_L2_X;
    Db_2D.y = BLOCK_SIZE_L2_Y;
    Db_2D.z = 1;
    Dg_2D.x = ceil((image_x)/T(BLOCK_SIZE_L2_X));
    Dg_2D.y = ceil((image_y)/T(BLOCK_SIZE_L2_Y));
    Dg_2D.z = 1;

    int size_i = (image_x+4)  * (image_y+4)  * sizeof(T);

    T* d_RESULTAT_first_conv;
    cudaMalloc((void**)&d_RESULTAT_first_conv, size_i);
  	checkCudaErrors(cudaMemset(d_RESULTAT_first_conv, 0., size_i));
    T* d_RESULTAT_second_conv;
    cudaMalloc((void**)&d_RESULTAT_second_conv, size_i);
	  checkCudaErrors(cudaMemset(d_RESULTAT_second_conv, 0., size_i));

    checkCudaErrors(cudaDeviceSynchronize());
    ConvKernel<T><<<Dg_2D_EXT, Db_2D_EXT>>>(d_RESULTAT_first_conv,  d_IMAGE_amp_ext, image_x+4, image_y+4);
    checkCudaErrors(cudaDeviceSynchronize());
    copy_gpu<T><<<Dg_2D, Db_2D>>>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
    checkCudaErrors(cudaDeviceSynchronize());
    prepare_for_convolution<T>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
    checkCudaErrors(cudaDeviceSynchronize());
    ConvKernel<T><<<Dg_2D_EXT, Db_2D_EXT>>>(d_RESULTAT_second_conv,  d_RESULTAT_first_conv, image_x+4, image_y+4);
    checkCudaErrors(cudaDeviceSynchronize());
    copy_gpu<T><<<Dg_2D, Db_2D>>>(d_conv_conv_amp, d_RESULTAT_second_conv, image_x, image_y);

    checkCudaErrors(cudaFree(d_RESULTAT_first_conv));
    checkCudaErrors(cudaFree(d_RESULTAT_second_conv));

}

template <typename T> void update_array_f_dev_sort(T lambda, T lambda_var, T* array_f_dev, T* map_image_dev, T* map_conv_dev, int indice_x, int indice_y, int k, T* b_params){

    T* array_f_dev_bis = NULL;
    cudaMalloc((void**)&array_f_dev_bis, 1*sizeof(T));
	  checkCudaErrors(cudaMemset(array_f_dev_bis, 0., 1*sizeof(T)));

    int size_j = (indice_x)  * (indice_y)  * sizeof(T);
    T* map_norm_dev = NULL;
    cudaMalloc((void**)&map_norm_dev, size_j);
  	checkCudaErrors(cudaMemset(map_norm_dev, 0., indice_x*indice_y*sizeof(T)));

    dim3 Dg_L2, Db_L2;

    Db_L2.x = BLOCK_SIZE_L2_X;
    Db_L2.y = BLOCK_SIZE_L2_Y;
    Db_L2.z = 1;

    Dg_L2.x = ceil(T(indice_x)/T(BLOCK_SIZE_L2_X));
    Dg_L2.y = ceil(T(indice_y)/T(BLOCK_SIZE_L2_Y));
    Dg_L2.z = 1;

    kernel_norm_map_simple_sort<T><<<Dg_L2,Db_L2>>>(lambda, lambda_var, map_norm_dev, map_conv_dev, map_image_dev, indice_x, indice_y, k, b_params);

    checkCudaErrors(cudaDeviceSynchronize());

    reduction_loop<T>(map_norm_dev, array_f_dev_bis, indice_x*indice_y);

    checkCudaErrors(cudaDeviceSynchronize());

    add_first_elements_sort<T><<<1,1>>>(array_f_dev_bis, array_f_dev);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(array_f_dev_bis));
}

template <typename T> void update_array_f_dev_sort(T lambda, T* array_f_dev, T* map_dev, int indice_x, int indice_y){

    T* array_f_dev_bis = NULL;
    cudaMalloc((void**)&array_f_dev_bis, 1*sizeof(T));
  	checkCudaErrors(cudaMemset(array_f_dev_bis, 0., 1*sizeof(array_f_dev_bis[0])));

    unsigned long int size_j = (indice_x)  * (indice_y)  * sizeof(T);
    T* map_norm_dev = NULL;
    cudaMalloc((void**)&map_norm_dev, size_j);
	  checkCudaErrors(cudaMemset(map_norm_dev, 0., indice_x*indice_y*sizeof(map_norm_dev[0])));

    dim3 Dg_L2, Db_L2;

    Db_L2.x = BLOCK_SIZE_L2_X;
    Db_L2.y = BLOCK_SIZE_L2_Y;
    Db_L2.z = 1;

    Dg_L2.x = ceil(T(indice_x)/T(BLOCK_SIZE_L2_X));
    Dg_L2.y = ceil(T(indice_y)/T(BLOCK_SIZE_L2_Y));
    Dg_L2.z = 1;

    kernel_norm_map_simple_sort<T><<<Dg_L2,Db_L2>>>(lambda, map_norm_dev, map_dev, indice_x, indice_y);

    checkCudaErrors(cudaDeviceSynchronize());

    reduction_loop<T>(map_norm_dev, array_f_dev_bis, indice_x*indice_y);

    checkCudaErrors(cudaDeviceSynchronize());

    add_first_elements_sort<T><<<1,1>>>(array_f_dev_bis, array_f_dev);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(array_f_dev_bis));
}

template double compute_residual_and_f_cheap_memory_trick<double>(double*, int*, int, double*, int*, int, double*, int*, int, double*, int*, int, int, int, int, int);
template double compute_residual_and_f<double>(double*, int*, int, double*, int*, int, double*, int*, int, double*, int*, int, int, int, int, int);
template void gradient_L_2_beta<double>(double*, int*, int, double*, int*, int, double*, int*, int, double*, int*, int, int);
template void reduction_loop<double>(double*, double*, int);
template void regularisation<double>(double*, double*, double*, double*, double&, int, int, int, parameters<double>&, double* temps_bis);
template void prepare_for_convolution<double>(double*, double*, int, int);
template void conv_twice_and_copy<double>(double*, double*, double*, int, int);
template void update_array_f_dev_sort<double>(double, double, double*, double*, double*, int, int, int, double*);
template void update_array_f_dev_sort<double>(double, double*, double*, int, int);

template float compute_residual_and_f_cheap_memory_trick<float>(float*, int*, int, float*, int*, int, float*, int*, int, float*, int*, int, int, int, int, int);
template float compute_residual_and_f<float>(float*, int*, int, float*, int*, int, float*, int*, int, float*, int*, int, int, int, int, int);
template void gradient_L_2_beta<float>(float*, int*, int, float*, int*, int, float*, int*, int, float*, int*, int, int);
template void reduction_loop<float>(float*, float*, int);
template void regularisation<float>(float*, float*, float*, float*, float&, int, int, int, parameters<float>&, double* temps_bis);
template void prepare_for_convolution<float>(float*, float*, int, int);
template void conv_twice_and_copy<float>(float*, float*, float*, int, int);
template void update_array_f_dev_sort<float>(float, float, float*, float*, float*, int, int, int, float*);
template void update_array_f_dev_sort<float>(float, float*, float*, int, int);
