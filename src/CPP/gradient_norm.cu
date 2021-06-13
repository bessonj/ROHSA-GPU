#include "gradient_norm.hpp"
//#include "kernel_gradient.cuh"
#include <cuda_runtime.h>
#include "kernels_for_hybrid.cu"
#include "kernels_for_hybrid_norm.cu"
#include <helper_cuda.h>
#include <cuda.h>
#include <omp.h>

#define Nb_time_mes 10


template <typename T> 
void regularization_norm(T* beta_dev, T* g_dev, T* b_params, T &f, int dim_x, int dim_y, int dim_v, parameters<T> &M, float* temps_kernel_regu){
//  printf("début f = %.26f\n",f);

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEventRecord(start);

	int n_beta = (3*M.n_gauss*dim_x*dim_y)+M.n_gauss;
  T* b_params_dev = NULL;
  checkCudaErrors(cudaMalloc(&b_params_dev, M.n_gauss*sizeof(T)));
  checkCudaErrors(cudaMemcpy(b_params_dev, b_params, M.n_gauss*sizeof(T), cudaMemcpyHostToDevice));

  T* array_f_dev = NULL;
  checkCudaErrors(cudaMalloc(&array_f_dev, 1*sizeof(T)));
  checkCudaErrors(cudaMemset(array_f_dev, 0., 1*sizeof(T)));

  T* array_f = NULL;
  array_f = (T*)malloc(1*sizeof(T));
  array_f[0]=0.;
//  cpy_first_num_dev<T><<<1,1>>>(array_f array_f_dev);

//  checkCudaErrors(cudaMemcpy(array_f_dev, array_f, 1*sizeof(T), cudaMemcpyHostToDevice));

  T* d_IMAGE_amp = NULL;
  checkCudaErrors(cudaMalloc(&d_IMAGE_amp, dim_x*dim_y*sizeof(T)));
  checkCudaErrors(cudaMemset(d_IMAGE_amp, 0., dim_x*dim_y*sizeof(T)));
  T* d_IMAGE_mu = NULL;
  checkCudaErrors(cudaMalloc(&d_IMAGE_mu, dim_x*dim_y*sizeof(T)));
  checkCudaErrors(cudaMemset(d_IMAGE_mu, 0., dim_x*dim_y*sizeof(T)));
  T* d_IMAGE_sig = NULL;
  checkCudaErrors(cudaMalloc(&d_IMAGE_sig, dim_x*dim_y*sizeof(T)));
  checkCudaErrors(cudaMemset(d_IMAGE_sig, 0., dim_x*dim_y*sizeof(T)));
  T* d_IMAGE_sig_square = NULL;
  checkCudaErrors(cudaMalloc(&d_IMAGE_sig_square, dim_x*dim_y*sizeof(T)));

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
  checkCudaErrors(cudaEventRecord(stop));
  float tmp = 0.;
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&tmp, start, stop));
  checkCudaErrors(cudaDeviceSynchronize());
  temps_kernel_regu[8]+=tmp; //reduction

  dim3 Dg_2D, Db_2D;
  Db_2D.x = BLOCK_SIZE_L2_X;
  Db_2D.y = BLOCK_SIZE_L2_Y;
  Db_2D.z = 1;
  Dg_2D.x = ceil(dim_x/T(BLOCK_SIZE_L2_X));
  Dg_2D.y = ceil(dim_y/T(BLOCK_SIZE_L2_Y));
  Dg_2D.z = 1;

  checkCudaErrors(cudaDeviceSynchronize());

  for(int k = 0; k<M.n_gauss; k++){
    cudaEvent_t record_event[8];
    float time_msec[7] = {0.,0.,0.,0.,0.,0.,0.};
    for (int i=0;i<8;i++){
        checkCudaErrors(cudaEventCreate(record_event+i));   
    }
    // Record the start event
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[0], NULL));
    checkCudaErrors(cudaDeviceSynchronize());

    get_gaussian_parameter_maps<T><<<Dg_2D, Db_2D>>>(beta_dev, d_IMAGE_amp, d_IMAGE_mu, d_IMAGE_sig, dim_x, dim_y, k);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[1], NULL));

    checkCudaErrors(cudaMemset(d_CONV_amp, 0., (dim_x+4)*(dim_y+4)*sizeof(T)));
    checkCudaErrors(cudaMemset(d_CONV_mu, 0., (dim_x+4)*(dim_y+4)*sizeof(T)));
    checkCudaErrors(cudaMemset(d_CONV_sig, 0., (dim_x+4)*(dim_y+4)*sizeof(T)));

    checkCudaErrors(cudaMemset(d_CONV_CONV_amp, 0., (dim_x+4)*(dim_y+4)*sizeof(T)));
    checkCudaErrors(cudaMemset(d_CONV_CONV_mu, 0., (dim_x+4)*(dim_y+4)*sizeof(T)));
    checkCudaErrors(cudaMemset(d_CONV_CONV_sig, 0., (dim_x+4)*(dim_y+4)*sizeof(T)));

    //Sets d_EXT_* arrays to 0.


    //Does the convolutions for each gaussian parameters of the k-th gaussian index
    float tmp_temps_mirror_and_conv[6] = {0.,0.,0.,0.,0.,0.};
    conv_twice_and_copy<T>(d_IMAGE_amp, d_CONV_amp, d_CONV_CONV_amp, dim_x, dim_y,tmp_temps_mirror_and_conv);
    conv_twice_and_copy<T>(d_IMAGE_mu, d_CONV_mu, d_CONV_CONV_mu, dim_x, dim_y,tmp_temps_mirror_and_conv);
    conv_twice_and_copy<T>(d_IMAGE_sig, d_CONV_sig, d_CONV_CONV_sig, dim_x, dim_y,tmp_temps_mirror_and_conv);



    float tmp_temps_R[2] = {0.,0.};
    update_array_f_dev_sort_fast_norm<T>(M.lambda_amp, M.lambda_mu, M.lambda_sig, M.lambda_var_sig, array_f_dev, d_CONV_amp, d_CONV_mu, d_CONV_sig, d_IMAGE_sig, dim_x, dim_y, k, b_params_dev,tmp_temps_R);
//    display_size<<<1,1>>>(array_f_dev, 1);

//    display_size<<<1,1>>>(array_f_dev, 1);
//    display_size<<<1,1>>>(b_params_dev, M.n_gauss);
//    exit(0);

    dim3 Dg_ud, Db_ud;
    Db_ud.x = BLOCK_SIZE_L2_X;
    Db_ud.y = BLOCK_SIZE_L2_Y;
    Db_ud.z = 1;
    Dg_ud.x = ceil(T(dim_x)/T(BLOCK_SIZE_L2_X));
    Dg_ud.y = ceil(T(dim_y)/T(BLOCK_SIZE_L2_Y));
    Dg_ud.z = 1;

    checkCudaErrors(cudaDeviceSynchronize());

    double temps_test__ = omp_get_wtime();

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaDeviceSynchronize());

    compute_nabla_R_wrt_theta_norm<T><<<Dg_ud,Db_ud>>>(g_dev, M.lambda_amp, M.lambda_mu, M.lambda_sig, M.lambda_var_sig, d_CONV_CONV_amp, d_CONV_CONV_mu, d_CONV_CONV_sig, d_IMAGE_sig, b_params_dev, dim_y, dim_x, k);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaDeviceSynchronize());

    temps_test += omp_get_wtime() - temps_test__;
    

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[4], NULL));
    checkCudaErrors(cudaDeviceSynchronize());

    T* d_image_sigma_reduc = NULL;
    checkCudaErrors(cudaMalloc(&d_image_sigma_reduc, 1*sizeof(T)));
  	checkCudaErrors(cudaMemset(d_image_sigma_reduc, 0., 1*sizeof(T)));

    compute_square<T><<<Dg_ud,Db_ud>>>(d_IMAGE_sig, d_IMAGE_sig_square, dim_x, dim_y);
    checkCudaErrors(cudaDeviceSynchronize());
    reduction_loop<T>(d_IMAGE_sig, d_image_sigma_reduc, dim_y*dim_x);

    T* d_image_sigma_reduc_square = NULL;
    checkCudaErrors(cudaMalloc(&d_image_sigma_reduc_square, 1*sizeof(T)));
  	checkCudaErrors(cudaMemset(d_image_sigma_reduc_square, 0., 1*sizeof(T)));

    checkCudaErrors(cudaDeviceSynchronize());

    reduction_loop<T>(d_IMAGE_sig_square, d_image_sigma_reduc_square, dim_y*dim_x);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[5], NULL));
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[6], NULL));
    checkCudaErrors(cudaDeviceSynchronize());

    compute_nabla_R_wrt_m_norm<T><<<1,1>>>(n_beta+1, g_dev, d_image_sigma_reduc, d_image_sigma_reduc_square, M.lambda_var_sig, M.n_gauss, b_params_dev, k, dim_x, dim_y);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_image_sigma_reduc));
    checkCudaErrors(cudaFree(d_image_sigma_reduc_square));


    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[7], NULL));
    
    checkCudaErrors(cudaEventSynchronize(record_event[1]));    
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));
    checkCudaErrors(cudaEventSynchronize(record_event[4]));
    checkCudaErrors(cudaEventSynchronize(record_event[5]));
    checkCudaErrors(cudaEventSynchronize(record_event[6]));
    checkCudaErrors(cudaEventSynchronize(record_event[7]));

    for(int i = 0; i<8-1; i++){
      checkCudaErrors(cudaEventElapsedTime(time_msec+i, record_event[i], record_event[i+1]));
    }

    temps_kernel_regu[0]+=time_msec[0]; //get_gaussian_parameter_maps
    temps_kernel_regu[1]+=tmp_temps_mirror_and_conv[0]+tmp_temps_mirror_and_conv[3]; //perform_mirror_effect_before_convolution
    temps_kernel_regu[2]+=time_msec[4]+tmp_temps_mirror_and_conv[1]+tmp_temps_mirror_and_conv[4]; //ConvKernel
    temps_kernel_regu[3]+=tmp_temps_mirror_and_conv[2]+tmp_temps_mirror_and_conv[5]; //copy_gpu
    temps_kernel_regu[4]+=tmp_temps_R[0]; //compute_R_map
    temps_kernel_regu[5]+=time_msec[2]; //compute_nabla_R_wrt_theta
    temps_kernel_regu[6]+=time_msec[6]; //compute_nabla_R_wrt_m
    temps_kernel_regu[7]+=time_msec[4]+tmp_temps_R[1]; //reduction

    checkCudaErrors(cudaDeviceSynchronize());

//init_extended_array_sort(T* d_IMAGE_amp, T* d_EXT_amp, int dim_x, int dim_y){

  }


//  display_size<<<1,1>>>(d_g, n_beta);

  checkCudaErrors(cudaDeviceSynchronize());
//  exit(0);

  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemcpy(array_f, array_f_dev, 1*sizeof(T), cudaMemcpyDeviceToHost));
  f += array_f[0];
//  printf("f = %.26f\n",f);
//	std::cin.ignore();

  checkCudaErrors(cudaFree(array_f_dev));
  free(array_f);

  checkCudaErrors(cudaFree(b_params_dev));
  checkCudaErrors(cudaFree(d_IMAGE_amp));
  checkCudaErrors(cudaFree(d_IMAGE_mu));
  checkCudaErrors(cudaFree(d_IMAGE_sig));
  checkCudaErrors(cudaFree(d_IMAGE_sig_square));
  checkCudaErrors(cudaFree(d_CONV_amp));
  checkCudaErrors(cudaFree(d_CONV_mu));
  checkCudaErrors(cudaFree(d_CONV_sig));
  checkCudaErrors(cudaFree(d_CONV_CONV_amp));
  checkCudaErrors(cudaFree(d_CONV_CONV_mu));
  checkCudaErrors(cudaFree(d_CONV_CONV_sig));
}


template <typename T> void update_array_f_dev_sort_fast_norm(T lambda_amp, T lambda_mu, T lambda_sig, T lambda_var_sig, T* array_f_dev, T* map_conv_amp_dev, T* map_conv_mu_dev, T* map_conv_sig_dev, T* map_image_sig_dev, int indice_x, int indice_y, int k, T* b_params_dev, float* temps){

    cudaEvent_t record_event[3];
    float time_msec[2] = {0.,0.};
    for (int i=0;i<3;i++){
        checkCudaErrors(cudaEventCreate(record_event+i));   
    }
    bool print = false;

    // Record the start event
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[0], NULL));

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

    compute_R_map_norm<T><<<Dg_L2,Db_L2>>>(lambda_amp, lambda_mu, lambda_sig, lambda_var_sig, map_norm_dev, map_conv_amp_dev, map_conv_mu_dev, map_conv_sig_dev, map_image_sig_dev, indice_x, indice_y, k, b_params_dev);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventSynchronize(record_event[1]));    
    checkCudaErrors(cudaEventRecord(record_event[1], NULL));
    checkCudaErrors(cudaDeviceSynchronize());

/*
    printf("--> Début print un morceau de map_conv_dev :\n");
    checkCudaErrors(cudaDeviceSynchronize());
    display_dev_complete_sort<T><<<1,1>>>(map_conv_dev,4);//indice_x*indice_y);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("--> Fin print un morceau de map_conv_dev :\n");
    checkCudaErrors(cudaDeviceSynchronize());
*/
//    printf("Dg_ud = %d, %d, %d ; Db_ud = %d, %d, %d\n",Dg_ud.x,Dg_ud.y,Dg_ud.z,Db_ud.x,Db_ud.y,Db_ud.z);

    checkCudaErrors(cudaDeviceSynchronize());

    reduction_loop<T>(map_norm_dev, array_f_dev_bis, indice_x*indice_y);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    checkCudaErrors(cudaDeviceSynchronize());


/*
    if(indice_x>=256 && print){
        checkCudaErrors(cudaDeviceSynchronize());
        printf("Début f convolution :\n");
        checkCudaErrors(cudaDeviceSynchronize());
        display_dev_sort<T><<<1,1>>>(array_f_dev_bis);
        display_dev_complete_sort<T><<<1,1>>>(map_conv_amp_dev,15);
        checkCudaErrors(cudaDeviceSynchronize());
        printf("Fin f convolution\n");
        checkCudaErrors(cudaDeviceSynchronize());
        std::cin.ignore();
    }
*/

    add_first_elements_sort<T><<<1,1>>>(array_f_dev_bis, array_f_dev);

    checkCudaErrors(cudaDeviceSynchronize());

    for(int i = 0; i<3-1; i++){
      checkCudaErrors(cudaEventElapsedTime(time_msec+i, record_event[i], record_event[i+1]));
    }

    temps[0]+=time_msec[0];
    temps[1]+=time_msec[1];
 
    
    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(array_f_dev_bis));
}

template void regularization_norm<double>(double*, double*, double*, double&, int, int, int, parameters<double>&, float*);
template void update_array_f_dev_sort_fast_norm<double>(double, double, double, double, double*, double*, double*, double*, double*, int, int, int, double*, float*);

template void regularization_norm<float>(float*, float*, float*, float&, int, int, int, parameters<float>&, float*);
template void update_array_f_dev_sort_fast_norm<float>(float, float, float, float, float*, float*, float*, float*, float*, int, int, int, float*, float*);
