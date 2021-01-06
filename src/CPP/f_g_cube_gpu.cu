#include "f_g_cube_gpu.hpp"
#include "kernel_gradient_sort.cuh"
#include "kernel_conv_sort.cu"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <omp.h>

void dummyInstantiator_sort(){
    initialize_b_params<float><<<1,1>>>(NULL, NULL, 0,0,0);
    initialize_array<float><<<1,1>>>(NULL, 0, 0.);
    ConvKernel_sort<float><<<1,1>>>(NULL, NULL, 0, 0);
    copy_gpu_sort<float><<<1,1>>>(NULL,NULL,0,0);
    extension_mirror_gpu_sort_bis<float><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_gpu_sort<float><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_gpu_sort_save<float><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_sort<float>(NULL, NULL, 0, 0);
    init_extended_array_sort<float><<<1,1>>>(NULL, NULL, 0, 0);
    parameter_maps_sliced_from_beta_sort<float><<<1,1>>>(NULL,NULL,0,0,0,0,0);
    print_device_array_sort<float><<<1,1>>>(NULL,0,0);
    parameter_maps_sliced_from_beta_sort<float><<<1,1>>>(NULL, NULL, NULL, NULL, 0, 0, 0);
    fill_gpu_sort<float><<<1,1>>>(NULL,NULL,0);

    initialize_b_params<double><<<1,1>>>(NULL, NULL, 0,0,0);
    initialize_array<double><<<1,1>>>(NULL, 0, 0.);
    ConvKernel_sort<double><<<1,1>>>(NULL, NULL, 0, 0);
    copy_gpu_sort<double><<<1,1>>>(NULL,NULL,0,0);
    extension_mirror_gpu_sort_bis<double><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_gpu_sort<double><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_gpu_sort_save<float><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_sort<double>(NULL, NULL, 0, 0);
    init_extended_array_sort<double><<<1,1>>>(NULL, NULL, 0, 0);
    parameter_maps_sliced_from_beta_sort<double><<<1,1>>>(NULL,NULL,0,0,0,0,0);
    print_device_array_sort<double><<<1,1>>>(NULL,0,0);
    parameter_maps_sliced_from_beta_sort<double><<<1,1>>>(NULL, NULL, NULL, NULL, 0, 0, 0);
    fill_gpu_sort<double><<<1,1>>>(NULL,NULL,0);
}



template <typename T> void prepare_for_convolution_sort(T* d_IMAGE_amp, T* d_IMAGE_amp_ext, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille_frame, dim3 ThreadsParBlock_frame)
{
    bool print = false;

    init_extended_array_sort<T><<<BlocksParGrille_init,ThreadsParBlock_init>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);

    if(print){
            checkCudaErrors(cudaDeviceSynchronize());
        printf("display d_IMAGE_amp :\n");
            checkCudaErrors(cudaDeviceSynchronize());
        display_dev_complete_sort<<<1,1>>>(d_IMAGE_amp, 600);
            checkCudaErrors(cudaDeviceSynchronize());
        std::cin.ignore();
    }

    if(print){
            checkCudaErrors(cudaDeviceSynchronize());
        printf("display d_IMAGE_amp_ext :\n");
            checkCudaErrors(cudaDeviceSynchronize());
        display_dev_complete_sort<<<1,1>>>(d_IMAGE_amp_ext, 600);
            checkCudaErrors(cudaDeviceSynchronize());
        std::cin.ignore();
    }

    checkCudaErrors(cudaDeviceSynchronize());
//    printf("BlocksParGrille_init = %d, %d, %d ; ThreadsParBlock_init = %d, %d, %d\n",BlocksParGrille_init.x,BlocksParGrille_init.y,BlocksParGrille_init.z,ThreadsParBlock_init.x,ThreadsParBlock_init.y,ThreadsParBlock_init.z);

//    extension_mirror_gpu_sort_save<T><<<BlocksParGrille_init,ThreadsParBlock_init>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
    extension_mirror_gpu_sort_bis<T><<<BlocksParGrille_init,ThreadsParBlock_init>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
}



template <typename T> void conv_twice_and_copy_sort(T* d_IMAGE_amp_ext, T* d_conv_amp, T* d_conv_conv_amp, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille, dim3 ThreadsParBlock, dim3 BlocksParGrille_frame, dim3 ThreadsParBlock_frame)
{
    int size_i = (image_x+4)  * (image_y+4)  * sizeof(T);

    T* d_RESULTAT_first_conv;
    cudaMalloc((void**)&d_RESULTAT_first_conv, size_i);
	checkCudaErrors(cudaMemset(d_RESULTAT_first_conv, 0., (image_x+4)*(image_y+4)*sizeof(d_RESULTAT_first_conv[0])));
    T* d_RESULTAT_second_conv;
    cudaMalloc((void**)&d_RESULTAT_second_conv, size_i);
	checkCudaErrors(cudaMemset(d_RESULTAT_second_conv, 0., (image_x+4)*(image_y+4)*sizeof(d_RESULTAT_second_conv[0])));

    checkCudaErrors(cudaDeviceSynchronize());
    ConvKernel_sort<T><<<BlocksParGrille_init, ThreadsParBlock_init>>>(d_RESULTAT_first_conv,  d_IMAGE_amp_ext, image_x+4, image_y+4);
    checkCudaErrors(cudaDeviceSynchronize());
    copy_gpu_sort<T><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
    checkCudaErrors(cudaDeviceSynchronize());
    prepare_for_convolution_sort<T>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille_frame, ThreadsParBlock_frame);
    checkCudaErrors(cudaDeviceSynchronize());
    ConvKernel_sort<T><<<BlocksParGrille_init, ThreadsParBlock_init>>>(d_RESULTAT_second_conv,  d_RESULTAT_first_conv, image_x+4, image_y+4);
    checkCudaErrors(cudaDeviceSynchronize());
    copy_gpu_sort<T><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_conv_amp, d_RESULTAT_second_conv, image_x, image_y);

//    extension_mirror_gpu_sort<T><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);

    checkCudaErrors(cudaFree(d_RESULTAT_first_conv));
    checkCudaErrors(cudaFree(d_RESULTAT_second_conv));

}





//    update_array_f_dev_sort(M.lambda_sig, M.lambda_var_sig, array_f_dev, d_IMAGE_sig, d_conv_sig, image_x, image_y, k, b_params_dev);
template <typename T> void update_array_f_dev_sort(T lambda, T lambda_var, T* array_f_dev, T* map_image_dev, T* map_conv_dev, int indice_x, int indice_y, int k, T* b_params){

    bool print = false;

    T* array_f_dev_bis = NULL;
    cudaMalloc((void**)&array_f_dev_bis, 1*sizeof(T));
	checkCudaErrors(cudaMemset(array_f_dev_bis, 0., 1*sizeof(T)));

    int size_j = (indice_x)  * (indice_y)  * sizeof(T);
    T* map_norm_dev = NULL;
    cudaMalloc((void**)&map_norm_dev, size_j);
	checkCudaErrors(cudaMemset(map_norm_dev, 0., indice_x*indice_y*sizeof(T)));

    dim3 Dg_L2, Db_L2;

    Db_L2.x = BLOCK_SIZE_X_2D_SORT_;
    Db_L2.y = BLOCK_SIZE_Y_2D_SORT_;
    Db_L2.z = 1;

    Dg_L2.x = ceil(T(indice_x)/T(BLOCK_SIZE_X_2D_SORT_));
    Dg_L2.y = ceil(T(indice_y)/T(BLOCK_SIZE_Y_2D_SORT_));
    Dg_L2.z = 1;

    kernel_norm_map_simple_sort<<<Dg_L2,Db_L2>>>(lambda, lambda_var, map_norm_dev, map_conv_dev, map_image_dev, indice_x, indice_y, k, b_params);

/*
    printf("--> Début print un morceau de map_conv_dev :\n");
    checkCudaErrors(cudaDeviceSynchronize());
    display_dev_complete_sort<<<1,1>>>(map_conv_dev,4);//indice_x*indice_y);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("--> Fin print un morceau de map_conv_dev :\n");
    checkCudaErrors(cudaDeviceSynchronize());
*/
//    printf("Dg_ud = %d, %d, %d ; Db_ud = %d, %d, %d\n",Dg_ud.x,Dg_ud.y,Dg_ud.z,Db_ud.x,Db_ud.y,Db_ud.z);

    checkCudaErrors(cudaDeviceSynchronize());

    reduction_loop_parallel<T>(map_norm_dev, array_f_dev_bis, indice_x*indice_y);

    checkCudaErrors(cudaDeviceSynchronize());

    if(indice_x>=256 && print){
        checkCudaErrors(cudaDeviceSynchronize());
        printf("Début f convolution :\n");
        checkCudaErrors(cudaDeviceSynchronize());
        display_dev_sort<<<1,1>>>(array_f_dev_bis);
        display_dev_complete_sort<<<1,1>>>(map_conv_dev,15);
        checkCudaErrors(cudaDeviceSynchronize());
        printf("Fin f convolution\n");
        checkCudaErrors(cudaDeviceSynchronize());
        std::cin.ignore();
    }

    add_first_elements_sort<<<1,1>>>(array_f_dev_bis, array_f_dev);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(array_f_dev_bis));
}


template <typename T> void update_array_f_dev_sort(T lambda, T* array_f_dev, T* map_dev, int indice_x, int indice_y){

    bool print = false;

    T* array_f_dev_bis = NULL;
    cudaMalloc((void**)&array_f_dev_bis, 1*sizeof(T));
	checkCudaErrors(cudaMemset(array_f_dev_bis, 0., 1*sizeof(array_f_dev_bis[0])));

    unsigned long int size_j = (indice_x)  * (indice_y)  * sizeof(T);
    T* map_norm_dev = NULL;
    cudaMalloc((void**)&map_norm_dev, size_j);
	checkCudaErrors(cudaMemset(map_norm_dev, 0., indice_x*indice_y*sizeof(map_norm_dev[0])));

    dim3 Dg_L2, Db_L2;

    Db_L2.x = BLOCK_SIZE_X_2D_SORT_;
    Db_L2.y = BLOCK_SIZE_Y_2D_SORT_;
    Db_L2.z = 1;//BLOCK_SIZE_L2_Z;

    Dg_L2.x = ceil(T(indice_x)/T(BLOCK_SIZE_X_2D_SORT_));
    Dg_L2.y = ceil(T(indice_y)/T(BLOCK_SIZE_Y_2D_SORT_));
    Dg_L2.z = 1;//ceil(indice_x/T(BLOCK_SIZE_L2_Z));

    kernel_norm_map_simple_sort<<<Dg_L2,Db_L2>>>(lambda, map_norm_dev, map_dev, indice_x, indice_y);

    checkCudaErrors(cudaDeviceSynchronize());

/*
    if(indice_x>128){
        checkCudaErrors(cudaDeviceSynchronize());
        printf("--> Début print un morceau de map_dev :\n");
        checkCudaErrors(cudaDeviceSynchronize());
        display_dev_complete_sort<<<1,1>>>(map_dev,4);//indice_x*indice_y);
//        display_dev_complete_sort<<<1,1>>>(map_norm_dev,4);//indice_x*indice_y);
        checkCudaErrors(cudaDeviceSynchronize());
        printf("--> Fin print un morceau de map_dev\n");
        checkCudaErrors(cudaDeviceSynchronize());
    }
*/

    reduction_loop_parallel<T>(map_norm_dev, array_f_dev_bis, indice_x*indice_y);

    if(indice_x>=256 && print){
        checkCudaErrors(cudaDeviceSynchronize());
        printf("Début f convolution :\n");
        checkCudaErrors(cudaDeviceSynchronize());
        display_dev_sort<<<1,1>>>(array_f_dev_bis);
        display_dev_complete_sort<<<1,1>>>(map_dev,15);
        checkCudaErrors(cudaDeviceSynchronize());
        printf("Fin f convolution\n");
        checkCudaErrors(cudaDeviceSynchronize());
        std::cin.ignore();
    }

    checkCudaErrors(cudaDeviceSynchronize());

    add_first_elements_sort<<<1,1>>>(array_f_dev_bis, array_f_dev);

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(array_f_dev_bis));
}





//beta_modif_dev
//
//
template <typename T> void conv2D_GPU_all_sort(const parameters &M, T* d_g, const int n_beta, T lambda_var_sig, T* b_params_dev, T* deriv_dev, T* beta_modif_dev, T* array_f_dev, const int image_x, const int image_y, const int n_gauss, float temps_transfert, float temps_mirroirs)
{
    bool print = false;
//    printf("\n beta_modif_dev below \n");
//    display_dev_complete_sort<<<1,1>>>(beta_modif_dev, 30);

    dummyInstantiator_sort();
    T Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    int nb_cycles = 1;
    long int kernel_x = 3;
    long int kernel_y = 3;
    long int kernel_radius_x = 1;
    long int kernel_radius_y = 1;

    int size_i = (image_x+4)  * (image_y+4)  * sizeof(T);
    int size_j = (image_x)  * (image_y)  * sizeof(T);
    int size_k = (kernel_x)  * (kernel_y)  * sizeof(T);

/*    int blocks_x  = ceil(T(image_x) / T(BLOCK_SIZE_X_2D_SORT));
    int blocks_y  = ceil(T(image_y) / T(BLOCK_SIZE_Y_2D_SORT));
    int blocks_x_init = ceil(T(image_x+4) / T(BLOCK_SIZE_X_2D_SORT));
    int blocks_y_init = ceil(T(image_y+4) / T(BLOCK_SIZE_Y_2D_SORT));*/

    int grid_frame = ceil(T(max(image_x, image_y)) / T(256));
    dim3 ThreadsParBlock_frame, BlocksParGrille_frame;
    ThreadsParBlock_frame.x = 256;
    ThreadsParBlock_frame.y = 2;
    ThreadsParBlock_frame.z = 1;    
    BlocksParGrille_frame.x = grid_frame;
    BlocksParGrille_frame.y = 2;
    BlocksParGrille_frame.z = 1;

for(int k = 0; k<M.n_gauss; k++){

    checkCudaErrors(cudaDeviceSynchronize());

    double temps_temp = (double)(omp_get_wtime());

    T* d_IMAGE_amp;
    cudaMalloc((void**)&d_IMAGE_amp, size_j);
	checkCudaErrors(cudaMemset(d_IMAGE_amp, 0., image_x*image_y*sizeof(d_IMAGE_amp[0])));
    T* d_IMAGE_amp_ext;
    cudaMalloc((void**)&d_IMAGE_amp_ext, size_i);
	checkCudaErrors(cudaMemset(d_IMAGE_amp_ext, 0., (image_x+4)*(image_y+4)*sizeof(d_IMAGE_amp_ext[0])));

    T* d_IMAGE_mu;
    cudaMalloc((void**)&d_IMAGE_mu, size_j);
	checkCudaErrors(cudaMemset(d_IMAGE_mu, 0., image_x*image_y*sizeof(d_IMAGE_mu[0])));
    T* d_IMAGE_mu_ext;
    cudaMalloc((void**)&d_IMAGE_mu_ext, size_i);
	checkCudaErrors(cudaMemset(d_IMAGE_mu_ext, 0., (image_x+4)*(image_y+4)*sizeof(d_IMAGE_mu_ext[0])));
    
    T* d_IMAGE_sig;
    cudaMalloc((void**)&d_IMAGE_sig, size_j);
	checkCudaErrors(cudaMemset(d_IMAGE_sig, 0., image_x*image_y*sizeof(d_IMAGE_sig[0])));
    T* d_IMAGE_sig_ext;
    cudaMalloc((void**)&d_IMAGE_sig_ext, size_i);
	checkCudaErrors(cudaMemset(d_IMAGE_sig_ext, 0., (image_x+4)*(image_y+4)*sizeof(d_IMAGE_sig_ext[0])));

    T* d_conv_amp;
    cudaMalloc((void**)&d_conv_amp, size_j);
	checkCudaErrors(cudaMemset(d_conv_amp, 0., image_x*image_y*sizeof(d_conv_amp[0])));
    T* d_conv_mu;
    cudaMalloc((void**)&d_conv_mu, size_j);
	checkCudaErrors(cudaMemset(d_conv_mu, 0., image_x*image_y*sizeof(d_conv_mu[0])));
    T* d_conv_sig;
    cudaMalloc((void**)&d_conv_sig, size_j);
	checkCudaErrors(cudaMemset(d_conv_sig, 0., image_x*image_y*sizeof(d_conv_sig[0])));

    T* d_conv_conv_amp;
    cudaMalloc((void**)&d_conv_conv_amp, size_j);
	checkCudaErrors(cudaMemset(d_conv_conv_amp, 0., image_x*image_y*sizeof(d_conv_conv_amp[0])));
    T* d_conv_conv_mu;
    cudaMalloc((void**)&d_conv_conv_mu, size_j);
	checkCudaErrors(cudaMemset(d_conv_conv_mu, 0., image_x*image_y*sizeof(d_conv_conv_mu[0])));
    T* d_conv_conv_sig;
    cudaMalloc((void**)&d_conv_conv_sig, size_j);
	checkCudaErrors(cudaMemset(d_conv_conv_sig, 0., image_x*image_y*sizeof(d_conv_conv_sig[0])));

    int blocks_x  = ceil(T(image_x) / T(BLOCK_SIZE_X_2D_SORT_));
    int blocks_y  = ceil(T(image_y) / T(BLOCK_SIZE_Y_2D_SORT_));
    int blocks_x_init = ceil(T(image_x+4) / T(BLOCK_SIZE_X_2D_SORT_));
    int blocks_y_init = ceil(T(image_y+4) / T(BLOCK_SIZE_Y_2D_SORT_));

    dim3 ThreadsParBlock(BLOCK_SIZE_X_2D_SORT_, BLOCK_SIZE_Y_2D_SORT_, 1);
    dim3 BlocksParGrille(blocks_x , blocks_y , 1);

    dim3 ThreadsParBlock_init(BLOCK_SIZE_X_2D_SORT_, BLOCK_SIZE_Y_2D_SORT_, 1);
    dim3 BlocksParGrille_init(blocks_x_init , blocks_y_init, 1);

    checkCudaErrors(cudaDeviceSynchronize());

    parameter_maps_sliced_from_beta_sort<T><<<BlocksParGrille, ThreadsParBlock>>>(beta_modif_dev, d_IMAGE_amp, d_IMAGE_mu, d_IMAGE_sig, image_x, image_y, k);
/*
    display_dev_complete_sort<<<1,1>>>(d_IMAGE_amp, image_x*image_y);
    display_dev_complete_sort<<<1,1>>>(d_IMAGE_mu, image_x*image_y);
    display_dev_complete_sort<<<1,1>>>(d_IMAGE_sig, image_x*image_y);
*/
    checkCudaErrors(cudaDeviceSynchronize());
    prepare_for_convolution_sort<T>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille_frame, ThreadsParBlock_frame);
    checkCudaErrors(cudaDeviceSynchronize());
    prepare_for_convolution_sort<T>(d_IMAGE_mu, d_IMAGE_mu_ext, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille_frame, ThreadsParBlock_frame);
    checkCudaErrors(cudaDeviceSynchronize());
    prepare_for_convolution_sort<T>(d_IMAGE_sig, d_IMAGE_sig_ext, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille_frame, ThreadsParBlock_frame);

    checkCudaErrors(cudaDeviceSynchronize());

    temps_mirroirs = omp_get_wtime() - temps_temp;

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t record_event[5];
    float time_msec[4];
    for (int i=0;i<5;i++){
        checkCudaErrors(cudaEventCreate(record_event+i));   
    }

    // Record the start event
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[0], NULL));


     checkCudaErrors(cudaEventSynchronize(record_event[1]));    
     checkCudaErrors(cudaEventRecord(record_event[1], NULL));
        
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));

    checkCudaErrors(cudaDeviceSynchronize());    
    conv_twice_and_copy_sort<T>(d_IMAGE_amp_ext, d_conv_amp, d_conv_conv_amp, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock, BlocksParGrille_frame, ThreadsParBlock_frame);
    checkCudaErrors(cudaDeviceSynchronize());
    conv_twice_and_copy_sort<T>(d_IMAGE_mu_ext, d_conv_mu, d_conv_conv_mu, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock, BlocksParGrille_frame, ThreadsParBlock_frame);
    checkCudaErrors(cudaDeviceSynchronize());
    conv_twice_and_copy_sort<T>(d_IMAGE_sig_ext, d_conv_sig, d_conv_conv_sig, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock, BlocksParGrille_frame, ThreadsParBlock_frame);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));    

    checkCudaErrors(cudaEventRecord(record_event[4], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[4]));
        
    //---------------------------------------------------------------------------------------------------------------------------    
checkCudaErrors(cudaEventElapsedTime(time_msec+0, record_event[0], record_event[1]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+1, record_event[1], record_event[2]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+2, record_event[2], record_event[3]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+3, record_event[3], record_event[4]));
    time_msec[4]=time_msec[0]+time_msec[1]+time_msec[2]+time_msec[3];

//    printf("\n\n\t Memory transfer speed :");
            
        float total_mem_time         = time_msec[0]+ time_msec[1]+time_msec[3];
        float total_time            = time_msec[4] ;
        float upload_speed           = float(((size_i )*1e-9) / ( time_msec[1] *1e-3));
        float download_speed         = float(( size_i*1e-9) / (time_msec[3]*1e-3));
        float average_speed          = (upload_speed + download_speed) / 2 ;
        float percentage_vs_memory  = ( total_mem_time / total_time )*100;

        temps_transfert += total_time/1000;

//    printf("\n IMAGE EXT\n");
/*
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_amp_ext, image_x+4,image_y+4);
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_mu_ext, image_x+4,image_y+4);
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_sig_ext, image_x+4,image_y+4);

    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_mu, image_x,image_y);
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_sig, image_x,image_y);

    display_dev_complete_2D_sort<<<1,1>>>(d_conv_conv_amp, image_x, image_y);
    display_dev_complete_2D_sort<<<1,1>>>(d_conv_conv_mu, image_x,image_y);
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_sig, image_x,image_y);

    display_dev_complete_sort<<<1,1>>>(b_params_dev, M.n_gauss);
    display_dev_complete_sort<<<1,1>>>(d_g, n_beta);
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_sig, image_x, image_y);

    checkCudaErrors(cudaDeviceSynchronize());
    printf("print f avant update_array_f_dev_sort\n");
    display_dev_complete_sort<<<1,1>>>(array_f_dev, 1);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("print f amp\n");

        checkCudaErrors(cudaDeviceSynchronize());
    printf("print amp map\n");
    display_dev_complete_sort<<<1,1>>>(d_IMAGE_amp_ext, 660);
        checkCudaErrors(cudaDeviceSynchronize());
    printf("print amp map\n");
    if(print){
        std::cin.ignore();
            checkCudaErrors(cudaDeviceSynchronize());
        printf("print amp map\n");
            checkCudaErrors(cudaDeviceSynchronize());
        display_dev_complete_sort<<<1,1>>>(d_IMAGE_amp, 300);
            checkCudaErrors(cudaDeviceSynchronize());
        printf("print mu map\n");
            checkCudaErrors(cudaDeviceSynchronize());
        display_dev_complete_sort<<<1,1>>>(d_IMAGE_mu, 300);
            checkCudaErrors(cudaDeviceSynchronize());
        printf("print sig map\n");
            checkCudaErrors(cudaDeviceSynchronize());
        display_dev_complete_sort<<<1,1>>>(d_IMAGE_sig, 300);
            checkCudaErrors(cudaDeviceSynchronize());
    }
*/
    if(print){
        printf("print conv amp map\n");
            checkCudaErrors(cudaDeviceSynchronize());
        display_dev_complete_sort<<<1,1>>>(d_conv_amp, 300);
            checkCudaErrors(cudaDeviceSynchronize());
    }
/*
    printf("print conv mu map\n");
        checkCudaErrors(cudaDeviceSynchronize());
    display_dev_complete_sort<<<1,1>>>(d_conv_mu, 300);
        checkCudaErrors(cudaDeviceSynchronize());
    printf("print conv sig map\n");
        checkCudaErrors(cudaDeviceSynchronize());
    display_dev_complete_sort<<<1,1>>>(d_conv_sig, 300);
        checkCudaErrors(cudaDeviceSynchronize());
    printf("display at index d_conv_amp[idx] :\n");
        checkCudaErrors(cudaDeviceSynchronize());
    display_dev_at_index<<<1,1>>>(d_conv_amp, image_x*image_y-2);
    display_dev_at_index<<<1,1>>>(d_conv_amp, image_x*image_y-1);
    display_dev_at_index<<<1,1>>>(d_conv_amp, image_x*image_y);
*/
        checkCudaErrors(cudaDeviceSynchronize());
    update_array_f_dev_sort<T>(M.lambda_amp, array_f_dev, d_conv_amp, image_x, image_y);
        checkCudaErrors(cudaDeviceSynchronize());
/*
    printf("print f mu\n");
    checkCudaErrors(cudaDeviceSynchronize());
*/
    update_array_f_dev_sort<T>(M.lambda_mu, array_f_dev, d_conv_mu, image_x, image_y);
//    update_array_f_dev_sort<T>(M.lambda_mu, array_f_dev, d_conv_mu, image_x, image_y);
    checkCudaErrors(cudaDeviceSynchronize());
/*
    printf("print f sig\n");
    checkCudaErrors(cudaDeviceSynchronize());
*/
//    update_array_f_dev_sort<T>(0., 0., array_f_dev, d_IMAGE_sig, d_conv_sig, image_x, image_y, k, b_params_dev);
    update_array_f_dev_sort<T>(M.lambda_sig, M.lambda_var_sig, array_f_dev, d_IMAGE_sig, d_conv_sig, image_x, image_y, k, b_params_dev);


    dim3 Dg_ud, Db_ud;
    Db_ud.x = BLOCK_SIZE_X_2D_SORT; //x
    Db_ud.y = BLOCK_SIZE_Y_2D_SORT; //y
    Dg_ud.x = ceil(T(image_x)/T(BLOCK_SIZE_X_2D_SORT));
    Dg_ud.y = ceil(T(image_y)/T(BLOCK_SIZE_Y_2D_SORT));
//    printf("Dg_ud = %d, %d, %d ; Db_ud = %d, %d, %d\n",Dg_ud.x,Dg_ud.y,Dg_ud.z,Db_ud.x,Db_ud.y,Db_ud.z);

    checkCudaErrors(cudaDeviceSynchronize());

//    display_dev_complete_sort<<<1,1>>>(d_conv_conv_amp, 30);
    kernel_update_deriv_conv_conv_sort<<<Dg_ud,Db_ud>>>(deriv_dev, M.lambda_amp, M.lambda_mu, M.lambda_sig, M.lambda_var_sig, d_conv_conv_amp, d_conv_conv_mu, d_conv_conv_sig, d_IMAGE_sig, b_params_dev, int(image_y), int(image_x),k);
//    display_dev_complete_sort<<<1,1>>>(deriv_dev, 30);
//	std::cin.ignore();

    cudaDeviceSynchronize();

    T* d_image_sigma_reduc = NULL;
    checkCudaErrors(cudaMalloc(&d_image_sigma_reduc, 1*sizeof(T)));
	checkCudaErrors(cudaMemset(d_image_sigma_reduc, 0., 1*sizeof(d_image_sigma_reduc[0])));
    reduction_loop_parallel<T>(d_IMAGE_sig, d_image_sigma_reduc, image_y*image_x);

    cudaDeviceSynchronize();
    int grid_k = ceil(T(n_gauss)/T(256));
    kernel_conv_g_reduction_sort<<<1,1>>>(n_beta, d_g, d_image_sigma_reduc, M.lambda_var_sig, n_gauss, b_params_dev, k, image_x, image_y);
//    kernel_conv_g_sort<<<Dg_ud,Db_ud>>>(n_beta, d_g, d_IMAGE_sig, M.lambda_var_sig, n_gauss, b_params_dev, k, image_x, image_y);

    checkCudaErrors(cudaFree(d_image_sigma_reduc));
    checkCudaErrors(cudaFree(d_conv_amp));
    checkCudaErrors(cudaFree(d_conv_mu));
    checkCudaErrors(cudaFree(d_conv_sig));
    checkCudaErrors(cudaFree(d_conv_conv_amp));
    checkCudaErrors(cudaFree(d_conv_conv_mu));
    checkCudaErrors(cudaFree(d_conv_conv_sig));
    checkCudaErrors(cudaFree(d_IMAGE_amp));
    checkCudaErrors(cudaFree(d_IMAGE_mu));
    checkCudaErrors(cudaFree(d_IMAGE_sig));
    checkCudaErrors(cudaFree(d_IMAGE_amp_ext));
    checkCudaErrors(cudaFree(d_IMAGE_mu_ext));
    checkCudaErrors(cudaFree(d_IMAGE_sig_ext));
}

}






























template <typename T> void gradient_L_2_beta_parallel(T* deriv_dev, int* taille_deriv, int* taille_deriv_dev, T* beta_modif_dev, int* taille_beta_modif_dev, T* residual_dev, int* taille_residual_dev, T* std_map_dev, int* taille_std_map_dev, int n_gauss)
{
   dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X_GRAD1; //x
    Db.y = BLOCK_SIZE_Y_GRAD1; //y
    Db.z = BLOCK_SIZE_Z_GRAD1; //gaussiennes
        //deriv_dev      --> (3g,y,x)  --> (z,y,x)
        //params     --> (3g,y,x)  --> (z,y,x)
    Dg.x = ceil(T(taille_deriv[2])/T(BLOCK_SIZE_X_GRAD1));
    Dg.y = ceil(T(taille_deriv[1])/T(BLOCK_SIZE_Y_GRAD1));
    Dg.z = ceil(T(n_gauss)/T(BLOCK_SIZE_Z_GRAD1));


//  printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);

//gradient_kernel_2_beta_working_sort<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, beta_modif_dev, taille_beta_modif_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);

//    display_dev_complete_sort<<<1,1>>>(beta_modif_dev,taille_deriv[2]*taille_deriv[1]*taille_deriv[0]);

//display_dev_sort<<<1,1>>>(beta_modif_dev);
//    display_dev_complete_sort<<<1,1>>>(beta_modif_dev,10);

    gradient_kernel_2_beta_with_INDEXING_sort<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, beta_modif_dev, taille_beta_modif_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);
//    gradient_kernel_2_beta_with_INDEXING_sort_edit<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, beta_modif_dev, taille_beta_modif_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);


//printf("DEBUG !!\n");

   }










   void gradient_L_3_parallel(double* deriv, int* taille_deriv, int product_taille_deriv, double* params, int* taille_params, int product_taille_params, double* residual, int* taille_residual, int product_residual, double* std_map, int* taille_std_map, int product_std_map, int n_gauss)
{
   double* params_dev = NULL;
   double* deriv_dev = NULL;
   double* residual_dev = NULL;
   double* std_map_dev = NULL;

   int* taille_params_dev = NULL;
   int* taille_deriv_dev = NULL;
   int* taille_residual_dev = NULL;
   int* taille_std_map_dev = NULL;

   checkCudaErrors(cudaMalloc(&deriv_dev, product_taille_deriv*sizeof(double)));
   checkCudaErrors(cudaMalloc(&residual_dev, product_residual*sizeof(double)));
   checkCudaErrors(cudaMalloc(&params_dev, product_taille_params*sizeof(double)));
   checkCudaErrors(cudaMalloc(&std_map_dev, product_std_map*sizeof(double)));

   checkCudaErrors(cudaMalloc(&taille_deriv_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_params_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
   
   checkCudaErrors(cudaMemcpy(deriv_dev, deriv, product_taille_deriv*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(residual_dev, residual, product_residual*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_std_map*sizeof(double), cudaMemcpyHostToDevice));
   
   checkCudaErrors(cudaMemcpy(taille_deriv_dev, taille_deriv, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map,2*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

   dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X_SORT; //gaussiennes
    Db.y = BLOCK_SIZE_Y_SORT; //y
    Db.z = BLOCK_SIZE_Z_SORT; //x
        //deriv      --> (y,x,3g)  --> (y,z,x)
        //params     --> (y,x,3g)  --> (y,z,x)
    Dg.x = ceil(n_gauss/double(BLOCK_SIZE_X_SORT));
    Dg.y = ceil(taille_deriv[1]/double(BLOCK_SIZE_Y_SORT));
    Dg.z = ceil(taille_deriv[2]/double(BLOCK_SIZE_Z_SORT));

//    printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);
///    printf("taille_dF_over_dB = %d, %d, %d\n",taille_dF_over_dB[0],taille_dF_over_dB[1],taille_dF_over_dB[2]);

  gradient_kernel_3_sort<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, params_dev, taille_params_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);

  checkCudaErrors(cudaMemcpy(deriv, deriv_dev, product_taille_deriv*sizeof(double), cudaMemcpyDeviceToHost));

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
template <typename T> void reduction_loop_parallel(T* array_in, T* d_array_f, int size_array){
    int GRID_SIZE_REDUCTION = int(ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION)));
    int N = ceil(log(T(size_array))/log(T(BLOCK_SIZE_REDUCTION)));
    reduce_last_in_one_thread_sort<<<1,1>>>(array_in, d_array_f, size_array);


/*
    checkCudaErrors(cudaDeviceSynchronize());

    if (N==1){
        int size_array_out_kernel = ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION));
        int copy_dev_blocks = ceil(T(size_array_out_kernel)/T(BLOCK_SIZE_REDUCTION));
        T* array_out_kernel=NULL;
        checkCudaErrors(cudaMalloc(&array_out_kernel, size_array_out_kernel*sizeof(T)));

        checkCudaErrors(cudaDeviceSynchronize());

        sum_reduction_sort<<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in, array_out_kernel, size_array);

        checkCudaErrors(cudaDeviceSynchronize());

        cpy_first_num_dev_sort<<<1,1>>>( array_out_kernel, d_array_f);
        cudaFree(array_out_kernel);

    } else{
        int size_array_out_kernel = ceil(T(size_array)/T(BLOCK_SIZE_REDUCTION));
        T* array_out_kernel=NULL;
        checkCudaErrors(cudaMalloc(&array_out_kernel, size_array_out_kernel*sizeof(T)));

        sum_reduction_sort<<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in, array_out_kernel, size_array);

        checkCudaErrors(cudaDeviceSynchronize());

        T* array_in_copied_2;
        checkCudaErrors(cudaMalloc(&array_in_copied_2, size_array_out_kernel*sizeof(T)));

        int copy_dev_blocks = ceil(T(size_array_out_kernel)/T(BLOCK_SIZE_REDUCTION));
        copy_dev_sort<<< copy_dev_blocks , BLOCK_SIZE_REDUCTION >>>(array_out_kernel, array_in_copied_2, size_array_out_kernel);

        cudaFree(array_out_kernel);

        checkCudaErrors(cudaDeviceSynchronize());

        T size_array_out_kernel_2 = ceil(T(size_array)/T(pow(BLOCK_SIZE_REDUCTION,2)));
        T* array_out_kernel_2=NULL;
        checkCudaErrors(cudaMalloc(&array_out_kernel_2, size_array_out_kernel_2*sizeof(T)));

        sum_reduction_sort<<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in_copied_2, array_out_kernel_2, size_array_out_kernel);

        checkCudaErrors(cudaDeviceSynchronize());

        if(N>2){
        reduce_last_in_one_thread_sort<<<1,1>>>(array_out_kernel_2, d_array_f, size_array_out_kernel_2);
        }
        else{
        cpy_first_num_dev_sort<<<1,1>>>( array_out_kernel_2, d_array_f);
        }
        cudaFree(array_in_copied_2);
        cudaFree(array_out_kernel_2);
    }
    */
}	








//f = compute_residual_and_f(beta_modif_dev, cube_flattened_dev, residual_dev, std_map_dev, indice_x, indice_y, indice_v, M.n_gauss);
template <typename T> void compute_residual_and_f_parallel(T* array_f_dev, T* beta_dev, T* cube_dev, T* residual_dev, T* std_map_dev, int indice_x, int indice_y, int indice_v, int n_gauss)
  {
    dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X_SORT; //
    Db.y = BLOCK_SIZE_Y_SORT; //
    Db.z = BLOCK_SIZE_Z_SORT; //

    Dg.x = ceil(T(indice_x)/T(BLOCK_SIZE_X_SORT));
    Dg.y = ceil(T(indice_y)/T(BLOCK_SIZE_Y_SORT));
    Dg.z = ceil(T(indice_v)/T(BLOCK_SIZE_Z_SORT));


    T* cube_reconstructed = NULL;
    checkCudaErrors(cudaMalloc(&cube_reconstructed, indice_x*indice_y*indice_v*sizeof(cube_reconstructed[0])));
	checkCudaErrors(cudaMemset(cube_reconstructed, 0., indice_x*indice_y*indice_v*sizeof(cube_reconstructed[0])));
  
    checkCudaErrors(cudaDeviceSynchronize());
    kernel_hypercube_reconstructed<<<Dg,Db>>>(beta_dev, cube_reconstructed, indice_x, indice_y, indice_v, n_gauss);
    checkCudaErrors(cudaDeviceSynchronize());
    kernel_residual_simple_difference<<<Dg,Db>>>(cube_dev, cube_reconstructed, residual_dev, indice_x, indice_y, indice_v);

//    kernel_residual_sort<<<Dg,Db>>>(beta_dev, cube_dev, residual_dev,indice_x, indice_y, indice_v, n_gauss);

    dim3 Dg_L2, Db_L2;
    Db_L2.x = BLOCK_SIZE_X_2D_SORT;
    Db_L2.y = BLOCK_SIZE_Y_2D_SORT;
    Db_L2.z = 1;
    Dg_L2.x = ceil(T(indice_x)/T(BLOCK_SIZE_X_2D_SORT));
    Dg_L2.y = ceil(T(indice_y)/T(BLOCK_SIZE_Y_2D_SORT));
    Dg_L2.z = 1;

    T* map_norm_dev = NULL;
    checkCudaErrors(cudaMalloc(&map_norm_dev, indice_x*indice_y*sizeof(map_norm_dev[0])));
	checkCudaErrors(cudaMemset(map_norm_dev, 0., indice_x*indice_y*sizeof(map_norm_dev[0])));

    checkCudaErrors(cudaDeviceSynchronize());

    kernel_norm_map_boucle_v_sort<<<Dg_L2, Db_L2>>>(map_norm_dev, residual_dev, std_map_dev, indice_x, indice_y, indice_v);

    checkCudaErrors(cudaDeviceSynchronize());

    reduction_loop_parallel<T>(map_norm_dev, array_f_dev, indice_x*indice_y);

    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(cube_reconstructed));
  }











void f_g_cube_parallel(parameters &M, double &f, double* g, int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, double* cube_flattened, double temps_conv, double temps_deriv, double temps_tableaux, double temps_res_f)   
  {
    int i,k,j,l,p;

	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {indice_x, indice_y, 3*M.n_gauss};
	int taille_beta_modif[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_cube[] = {indice_v, indice_y, indice_x};
	int taille_image_conv[] = {indice_y, indice_x};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
	int product_cube = taille_cube[0]*taille_cube[1]*taille_cube[2]; 
	int product_beta = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_beta_modif = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_image_conv = taille_image_conv[0]*taille_image_conv[1];

	size_t size_deriv = product_deriv * sizeof(double);
	size_t size_res = product_residual * sizeof(double);
	size_t size_std = product_std_map_ * sizeof(double);
	size_t size_beta_modif = product_beta_modif * sizeof(double);
	size_t size_image_conv = product_image_conv * sizeof(double);
	size_t size_b_params = M.n_gauss * sizeof(double);

	double* b_params = (double*)malloc(size_b_params);
	double* deriv = (double*)malloc(size_deriv);
	double* residual = (double*)malloc(size_res);
	double* std_map_ = (double*)malloc(size_std);
	double* beta_modif = (double*)malloc(size_beta_modif);

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;
	double temps1_tableaux = omp_get_wtime();

	for(int i = 0; i<product_deriv; i++){
		deriv[i]=0.;
	}

	for(i=0; i<indice_y; i++){
		for(j=0; j<indice_x; j++){
			std_map_[i*indice_x+j]=std_map[i][j];
		}
	}

	for(int i = 0; i< n_beta; i++){
		g[i]=0.;
	}
	f=0.;

//beta est de taille : x,y,3g
//params est de taille : 3g,y,x

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	double temps_modification_beta1 = omp_get_wtime();
/*
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for(p=0; p<3*M.n_gauss; p++){
				beta_modif[j*indice_y*3*M.n_gauss+i*3*M.n_gauss+p] = beta[j*indice_y*3*M.n_gauss+i*3*M.n_gauss+p];
			}
		}
	}
*/
   double temps_modification_beta2 = omp_get_wtime();

   double temps2_tableaux = omp_get_wtime();

   double* d_image_sigma_reduc = NULL;
   checkCudaErrors(cudaMalloc(&d_image_sigma_reduc, product_image_conv*sizeof(double)));
   double* d_g = NULL;
   checkCudaErrors(cudaMalloc(&d_g,n_beta*sizeof(double)));
   checkCudaErrors(cudaMemcpy(d_g, g, n_beta*sizeof(double), cudaMemcpyHostToDevice));

   double* beta_modif_dev = NULL;
   checkCudaErrors(cudaMalloc(&beta_modif_dev, product_beta_modif*sizeof(double)));
   checkCudaErrors(cudaMemcpy(beta_modif_dev, beta, product_beta_modif*sizeof(double), cudaMemcpyHostToDevice));

   double* b_params_dev = NULL;
   checkCudaErrors(cudaMalloc(&b_params_dev, M.n_gauss*sizeof(double)));
   checkCudaErrors(cudaMemcpy(b_params_dev, b_params, M.n_gauss*sizeof(double), cudaMemcpyHostToDevice));

   double* cube_flattened_dev = NULL;
   checkCudaErrors(cudaMalloc(&cube_flattened_dev, product_cube*sizeof(double)));
   checkCudaErrors(cudaMemcpy(cube_flattened_dev, cube_flattened, product_cube*sizeof(double), cudaMemcpyHostToDevice));

   double* residual_dev = NULL;
   checkCudaErrors(cudaMalloc(&residual_dev, product_residual*sizeof(double)));
   checkCudaErrors(cudaMemcpy(residual_dev, residual, product_residual*sizeof(double), cudaMemcpyHostToDevice));

   double* std_map_dev = NULL;
   checkCudaErrors(cudaMalloc(&std_map_dev, product_std_map_*sizeof(double)));
   checkCudaErrors(cudaMemcpy(std_map_dev, std_map_, product_std_map_*sizeof(double), cudaMemcpyHostToDevice));

   double* array_f_dev = NULL;
   checkCudaErrors(cudaMalloc(&array_f_dev, 1*sizeof(double)));
   init_dev_sort<<<1,1>>>(array_f_dev, 0.);

   double* deriv_dev = NULL;
   checkCudaErrors(cudaMalloc(&deriv_dev, product_deriv*sizeof(double)));
   checkCudaErrors(cudaMemcpy(deriv_dev, deriv, product_deriv*sizeof(double), cudaMemcpyHostToDevice));

   int* taille_beta_modif_dev = NULL;
   checkCudaErrors(cudaMalloc(&taille_beta_modif_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMemcpy(taille_beta_modif_dev, taille_beta_modif, 3*sizeof(int), cudaMemcpyHostToDevice));

   int* taille_deriv_dev = NULL;
   checkCudaErrors(cudaMalloc(&taille_deriv_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMemcpy(taille_deriv_dev, taille_deriv, 3*sizeof(int), cudaMemcpyHostToDevice));

   int* taille_residual_dev = NULL;
   checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

   int* taille_std_map_dev = NULL;
   checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
   checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map_, 2*sizeof(int), cudaMemcpyHostToDevice));

//    display_dev_complete_sort<<<1,1>>>(beta_modif_dev,20);

   double temps1_res_f = omp_get_wtime();

   compute_residual_and_f_parallel(array_f_dev, beta_modif_dev, cube_flattened_dev, residual_dev, std_map_dev, indice_x, indice_y, indice_v, M.n_gauss);

    double temps2_res_f = omp_get_wtime();

    temps_res_f += temps2_res_f - temps1_res_f;

/*
    display_dev_complete_sort<<<1,1>>>(residual_dev,3);
    display_dev_complete_sort<<<1,1>>>(beta_modif_dev,10);
    display_dev_complete_sort<<<1,1>>>(std_map_dev,10);
    display_dev_complete_sort<<<1,1>>>(deriv_dev,10);
*/

    double temps2_dF_dB = omp_get_wtime();

    double temps1_deriv = omp_get_wtime();

    gradient_L_2_beta_parallel(deriv_dev, taille_deriv, taille_deriv_dev, beta_modif_dev, taille_beta_modif_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, M.n_gauss);


	double temps2_deriv = omp_get_wtime();
	double temps1_conv = omp_get_wtime();

//    display_dev_sort<<<1,1>>>(deriv_dev);


    conv2D_GPU_all_sort(M, d_g, n_beta, M.lambda_var_sig, b_params_dev, deriv_dev, beta_modif_dev, array_f_dev, indice_x, indice_y, M.n_gauss, 0,0);//temps_transfert, temps_mirroirs);

	double temps2_conv = omp_get_wtime();

//printf("suite \n");
//if(indice_x != indice_y || (indice_x == 128 && indice_x == 128)){
//   display_dev_complete_sort<<<1,1>>>(deriv_dev,30);
//   display_dev_sort<<<1,1>>>(deriv_dev);
//}
/*
   printf("indice_x = %d , indice_y = %d\n", indice_x, indice_y);
   for(int i = 0; i<indice_x; i++){
       for(int j = 0; j<indice_y; j++){
           printf("beta_modif[%d] = %f\n", indice_x*j+i, beta_modif[indice_x*j+i]);
       }
    }

   for(int i = 0; i<indice_x; i++){
       for(int j = 0; j<indice_y; j++){
           printf("beta_modif[%d] = %f\n", 1*indice_x*indice_y+indice_x*j+i, beta_modif[1*indice_x*indice_y+indice_x*j+i]);
       }
    }

   for(int i = 0; i<indice_x; i++){
       for(int j = 0; j<indice_y; j++){
           printf("beta_modif[%d] = %f\n", 2*indice_x*indice_y+indice_x*j+i, beta_modif[2*indice_x*indice_y+indice_x*j+i]);
       }
    }
*/
//	std::cin.ignore();


    checkCudaErrors(cudaMemcpy(deriv, deriv_dev, product_deriv*sizeof(double), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(g, d_g, n_beta*sizeof(double), cudaMemcpyDeviceToHost));

    for(int k(0); k<indice_x; k++)
    {
        for(int j(0); j<indice_y; j++)
        {
	        for(int i(0); i<3*M.n_gauss; i++)
			{
	            g[i*indice_y*indice_x+j*indice_x+k] = deriv[i*indice_y*indice_x+j*indice_x+k];
	    	}
        }
    }

    double* array_f = NULL;
    array_f = (double*)(malloc(1*sizeof(double)));
    checkCudaErrors(cudaMemcpy(array_f, array_f_dev, 1*sizeof(double), cudaMemcpyDeviceToHost));

    f = array_f[0];

    checkCudaErrors(cudaFree(b_params_dev));
    checkCudaErrors(cudaFree(d_g));
    checkCudaErrors(cudaFree(d_image_sigma_reduc));
    checkCudaErrors(cudaFree(beta_modif_dev));
    checkCudaErrors(cudaFree(cube_flattened_dev));
    checkCudaErrors(cudaFree(array_f_dev));
    checkCudaErrors(cudaFree(deriv_dev));
    checkCudaErrors(cudaFree(taille_beta_modif_dev));
    checkCudaErrors(cudaFree(taille_deriv_dev));
    checkCudaErrors(cudaFree(std_map_dev));
    checkCudaErrors(cudaFree(taille_std_map_dev));
    checkCudaErrors(cudaFree(residual_dev));
    checkCudaErrors(cudaFree(taille_residual_dev));


    //  free(taille_beta_modif);
    free(b_params);
    free(deriv);
    free(residual);
    free(std_map_);
    free(beta_modif);

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;


}






























template <typename T> void f_g_cube_parallel_lib(const parameters &M, T &f, T* d_g, const int n, T* beta_dev, const int indice_v, const int indice_y, const int indice_x, T* std_map_dev, T* cube_flattened_dev, T* temps)   
  {
    bool print = false;
//    bool print = true;
    int rang = 8;
    int rang_print = 100;

/*
    if(indice_x>128){
        print = true;
    }
*/

    dummyInstantiator_sort();

    int i,k,j,l,p;

	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_cube[] = {indice_v, indice_y, indice_x};
	int taille_image_conv[] = {indice_y, indice_x};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
	int product_cube = taille_cube[0]*taille_cube[1]*taille_cube[2]; 
	int product_beta_modif = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_image_conv = taille_image_conv[0]*taille_image_conv[1];

	size_t size_deriv = product_deriv * sizeof(T);
	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_beta_modif = product_beta_modif * sizeof(T);
	size_t size_image_conv = product_image_conv * sizeof(T);
	size_t size_b_params = M.n_gauss * sizeof(T);

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;
	double temps1_tableaux = omp_get_wtime();
    //beta est de taille : x,y,3g
    //params est de taille : 3g,y,x


    // Allocate CUDA events that we'll use for timing
    cudaEvent_t record_event[5];
    float time_msec[4];
    for (int i=0;i<5;i++){
        checkCudaErrors(cudaEventCreate(record_event+i));   
    }

    // Record the start event
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[0], NULL));

    double temps2_tableaux = omp_get_wtime();

    T* b_params_dev = nullptr;
    checkCudaErrors(cudaMalloc(&b_params_dev, M.n_gauss*sizeof(T)));
	checkCudaErrors(cudaMemset(b_params_dev, 0., M.n_gauss*sizeof(b_params_dev[0])));

    T* residual_dev = nullptr;
    checkCudaErrors(cudaMalloc(&residual_dev, product_residual*sizeof(T)));
	checkCudaErrors(cudaMemset(residual_dev, 0., product_residual*sizeof(residual_dev[0])));

    T* array_f_dev = nullptr;
    checkCudaErrors(cudaMalloc(&array_f_dev, 1*sizeof(T)));
	checkCudaErrors(cudaMemset(array_f_dev, 0., 1*sizeof(array_f_dev[0])));

    T* deriv_dev = nullptr;
    checkCudaErrors(cudaMalloc(&deriv_dev, product_deriv*sizeof(T)));
	checkCudaErrors(cudaMemset(deriv_dev, 0., product_deriv*sizeof(deriv_dev[0])));

    int* taille_beta_dev = nullptr;
    checkCudaErrors(cudaMalloc(&taille_beta_dev, 3*sizeof(int)));
    checkCudaErrors(cudaMemcpy(taille_beta_dev, taille_beta, 3*sizeof(int), cudaMemcpyHostToDevice));

    int* taille_deriv_dev = nullptr;
    checkCudaErrors(cudaMalloc(&taille_deriv_dev, 3*sizeof(int)));
    checkCudaErrors(cudaMemcpy(taille_deriv_dev, taille_deriv, 3*sizeof(int), cudaMemcpyHostToDevice));

    int* taille_residual_dev = nullptr;
    checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
    checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

    int* taille_std_map_dev = nullptr;
    checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
    checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map_, 2*sizeof(int), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemset(d_g, 0., n_beta*sizeof(d_g[0])));
    f=0.;

    checkCudaErrors(cudaDeviceSynchronize());
    //on peut changer 64 en autre chose
    int grid_initialize_b_params = ceil(T(M.n_gauss)/T(64)); //=1 (trivial)
    initialize_b_params<T><<<grid_initialize_b_params, 64>>>(b_params_dev, beta_dev, M.n_gauss, n_beta, M.n_gauss);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventSynchronize(record_event[1]));    
    checkCudaErrors(cudaEventRecord(record_event[1], NULL));

    if(print)
    {
        checkCudaErrors(cudaDeviceSynchronize());
        printf("début\n");
        display_dev_complete_sort<<<1,1>>>(beta_dev, rang_print);
        display_dev_complete_sort<<<1,1>>>(array_f_dev, 1);
//        display_dev_complete_sort<<<1,1>>>(beta_dev, 30);
        std::cin.ignore();
    }

    checkCudaErrors(cudaDeviceSynchronize());

    compute_residual_and_f_parallel<T>(array_f_dev, beta_dev, cube_flattened_dev, residual_dev, std_map_dev, indice_x, indice_y, indice_v, M.n_gauss);

    checkCudaErrors(cudaDeviceSynchronize());
    
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    

    gradient_L_2_beta_parallel<T>(deriv_dev, taille_deriv, taille_deriv_dev, beta_dev, taille_beta_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, M.n_gauss);

    checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));    

    if(print)
    {
        checkCudaErrors(cudaDeviceSynchronize());
        printf("milieu\n");
        display_dev_complete_sort<<<1,1>>>(deriv_dev, rang_print);
        display_dev_complete_sort<<<1,1>>>(array_f_dev, 1);
        std::cin.ignore();
    }

    checkCudaErrors(cudaDeviceSynchronize());

    conv2D_GPU_all_sort<T>(M, d_g, n_beta, M.lambda_var_sig, b_params_dev, deriv_dev, beta_dev, array_f_dev, indice_x, indice_y, M.n_gauss, 0,0);//temps_transfert, temps_mirroirs);

    checkCudaErrors(cudaDeviceSynchronize());

    dim3 Dg_L2, Db_L2;
    Db_L2.x = 256;
    Db_L2.y = 1;
    Db_L2.z = 1;
    Dg_L2.x = ceil(T(3*M.n_gauss*indice_x*indice_y)/T(256));
    Dg_L2.y = 1;
    Dg_L2.z = 1;
    fill_gpu_sort<T><<<Dg_L2, Db_L2>>>(d_g, deriv_dev, 3*M.n_gauss*indice_x*indice_y);

//    display_dev_sort<<<1,1>>>(array_f_dev);
//    display_dev_complete_sort<<<1,1>>>(beta_dev, n_beta);
//    std::cin.ignore();

    T* array_f = NULL;
    array_f = (T*)(malloc(1*sizeof(T)));
    checkCudaErrors(cudaMemcpy(array_f, array_f_dev, 1*sizeof(T), cudaMemcpyDeviceToHost));

    f = array_f[0];
//    exit(0);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(record_event[4], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[4]));
        
    checkCudaErrors(cudaEventElapsedTime(time_msec+0, record_event[0], record_event[1]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+1, record_event[1], record_event[2]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+2, record_event[2], record_event[3]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+3, record_event[3], record_event[4]));
    time_msec[4]=time_msec[0]+time_msec[1]+time_msec[2]+time_msec[3];

    double total_mem_time 		= time_msec[0]+ time_msec[1]+time_msec[3];
    double mem_time_0 		= time_msec[0];
    double mem_time_1 		= time_msec[1];
    double mem_time_2 		= time_msec[2];
    double mem_time_3 		= time_msec[3];
    double mem_time_4 		= time_msec[4];

    temps[0] += mem_time_0;
    temps[1] += mem_time_1;
    temps[2] += mem_time_2;
    temps[3] += mem_time_3;
    temps[4] += mem_time_4;

    if(print)
    {
        checkCudaErrors(cudaDeviceSynchronize());
        printf("fin\n");
        display_dev_complete_sort<<<1,1>>>(d_g, rang_print);
        display_dev_complete_fin_sort<<<1,1>>>(d_g, n_beta, 40);
        display_dev_complete_sort<<<1,1>>>(array_f_dev, 1);
//        display_dev_complete_sort<<<1,1>>>(beta_dev, 30);
        std::cin.ignore();
    }

    checkCudaErrors(cudaFree(b_params_dev));
    checkCudaErrors(cudaFree(array_f_dev));
    checkCudaErrors(cudaFree(deriv_dev));
    checkCudaErrors(cudaFree(taille_beta_dev));
    checkCudaErrors(cudaFree(taille_deriv_dev));
    checkCudaErrors(cudaFree(taille_std_map_dev));
    checkCudaErrors(cudaFree(residual_dev));
    checkCudaErrors(cudaFree(taille_residual_dev));

    free(array_f);

}

 	template void f_g_cube_parallel_lib<double>(const parameters&, double&, double*, const int, double*, const int, const int, const int, double*, double*, double*);
    template void compute_residual_and_f_parallel<double>(double*, double*, double*, double*, double*, int, int, int, int);
    template void reduction_loop_parallel<double>(double*, double*, int);
    template void gradient_L_2_beta_parallel<double>(double*, int*, int*, double*, int*, double*, int*, double*, int*, int);
    template void conv2D_GPU_all_sort<double>(const parameters&, double*, const int, double, double*, double*, double*, double*, const int, const int, const int, float, float);
    template void update_array_f_dev_sort<double>(double,double, double*, double*, double*, int, int, int, double*);
    template void update_array_f_dev_sort<double>(double, double*, double*, int, int);
    template void conv_twice_and_copy_sort<double>(double*, double*, double*, int, int, dim3, dim3, dim3, dim3, dim3, dim3);
    template void prepare_for_convolution_sort<double>(double*, double*, int, int, dim3, dim3, dim3, dim3);



