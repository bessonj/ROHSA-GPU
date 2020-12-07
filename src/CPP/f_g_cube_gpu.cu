#include "f_g_cube_gpu.hpp"
#include "kernel_gradient_sort.cuh"
#include "kernel_conv_sort.cu"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <omp.h>



void dummyInstantiator_sort(){
    ConvKernel_sort<float><<<1,1>>>(NULL, NULL, 0, 0);
    copy_gpu_sort<float><<<1,1>>>(NULL,NULL,0,0);
    extension_mirror_gpu_sort<float><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_gpu_sort_save<float><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_sort<float>(NULL, NULL, 0, 0);
    init_extended_array_sort<float><<<1,1>>>(NULL, NULL, 0, 0);
    parameter_maps_sliced_from_beta_sort<float><<<1,1>>>(NULL,NULL,0,0,0,0,0);
    print_device_array_sort<float><<<1,1>>>(NULL,0,0);
    parameter_maps_sliced_from_beta_sort<float><<<1,1>>>(NULL, NULL, NULL, NULL, 0, 0, 0);

    ConvKernel_sort<double><<<1,1>>>(NULL, NULL, 0, 0);
    copy_gpu_sort<double><<<1,1>>>(NULL,NULL,0,0);
    extension_mirror_gpu_sort<double><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_gpu_sort_save<float><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror_sort<double>(NULL, NULL, 0, 0);
    init_extended_array_sort<double><<<1,1>>>(NULL, NULL, 0, 0);
    parameter_maps_sliced_from_beta_sort<double><<<1,1>>>(NULL,NULL,0,0,0,0,0);
    print_device_array_sort<double><<<1,1>>>(NULL,0,0);
    parameter_maps_sliced_from_beta_sort<double><<<1,1>>>(NULL, NULL, NULL, NULL, 0, 0, 0);
}

void prepare_for_convolution_sort(double* d_IMAGE_amp, double* d_IMAGE_amp_ext, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille_frame, dim3 ThreadsParBlock_frame)
{
/*
	printf("BlocksParGrille.x =      %d, BlocksParGrille.y =       %d, BlocksParGrille.z =       %d\n", BlocksParGrille.x, BlocksParGrille.y, BlocksParGrille.z);
    printf("ThreadsParBlock.x =      %d, ThreadsParBlock.y =      %d, ThreadsParBlock.z =      %d\n", ThreadsParBlock.x, ThreadsParBlock.y, ThreadsParBlock.z);
    printf("BlocksParGrille_init.x = %d, BlocksParGrille_init.y =  %d, BlocksParGrille_init.z =  %d\n", BlocksParGrille_init.x, BlocksParGrille_init.y, BlocksParGrille_init.z);
    printf("ThreadsParBlock_init.x = %d, ThreadsParBlock_init.y = %d, ThreadsParBlock_init.z =  %d\n", ThreadsParBlock_init.x, ThreadsParBlock_init.y, ThreadsParBlock_init.z);
*/
    init_extended_array_sort<double><<<BlocksParGrille_init,ThreadsParBlock_init>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
//    extension_mirror_gpu_sort_save<double><<<BlocksParGrille_frame,ThreadsParBlock_frame>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
    extension_mirror_gpu_sort<double><<<BlocksParGrille_frame,ThreadsParBlock_frame>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
}

void conv_twice_and_copy_sort(double* d_IMAGE_amp_ext, double* d_conv_amp, double* d_conv_conv_amp, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille, dim3 ThreadsParBlock, dim3 BlocksParGrille_frame, dim3 ThreadsParBlock_frame)
{
    int size_i = (image_x+4)  * (image_y+4)  * sizeof(double);

    double* d_RESULTAT_first_conv;        
    cudaMalloc((void**)&d_RESULTAT_first_conv, size_i);
    cudaMemset ( d_RESULTAT_first_conv, 0 , size_i) ;

    double* d_RESULTAT_second_conv;        
    cudaMalloc((void**)&d_RESULTAT_second_conv, size_i);
    cudaMemset ( d_RESULTAT_second_conv, 0 , size_i) ;

    ConvKernel_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_first_conv,  d_IMAGE_amp_ext, image_x+4, image_y+4);
    copy_gpu_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
//    init_extended_array_sort<double><<<BlocksParGrille_init,ThreadsParBlock_init>>>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
//    extension_mirror_gpu_sort<double><<<BlocksParGrille,ThreadsParBlock>>>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
    prepare_for_convolution_sort(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille_frame, ThreadsParBlock_frame);
    ConvKernel_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_second_conv,  d_RESULTAT_first_conv, image_x+4, image_y+4);
    copy_gpu_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_conv_amp, d_RESULTAT_second_conv, image_x, image_y);

//    extension_mirror_gpu_sort<double><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);

    checkCudaErrors(cudaFree(d_RESULTAT_first_conv));
    checkCudaErrors(cudaFree(d_RESULTAT_second_conv));

}





//    update_array_f_dev_sort(M.lambda_sig, M.lambda_var_sig, array_f_dev, d_IMAGE_sig, d_conv_sig, image_x, image_y, k, b_params_dev);
void update_array_f_dev_sort(double lambda, double lambda_var, double* array_f_dev, double* map_image_dev, double* map_conv_dev, int indice_x, int indice_y, int k, double* b_params){

    double* array_f_dev_bis = NULL;
    cudaMalloc((void**)&array_f_dev_bis, 1*sizeof(double));

    int size_j = (indice_x)  * (indice_y)  * sizeof(double);
    double* map_norm_dev = NULL;
    cudaMalloc((void**)&map_norm_dev, size_j);

    dim3 Dg_L2, Db_L2;

    Db_L2.x = BLOCK_SIZE_X_2D_SORT;
    Db_L2.y = BLOCK_SIZE_Y_2D_SORT;
    Db_L2.z = 1;//;

    Dg_L2.x = ceil(double(indice_x)/double(BLOCK_SIZE_X_2D_SORT));
    Dg_L2.y = ceil(double(indice_y)/double(BLOCK_SIZE_Y_2D_SORT));
    Dg_L2.z = 1;//ceil(indice_x/double());

    kernel_norm_map_simple_sort<<<Dg_L2,Db_L2>>>(lambda, lambda_var, map_norm_dev, map_conv_dev, map_image_dev, indice_x, indice_y, k, b_params);

    reduction_loop_parallel(map_norm_dev, array_f_dev_bis, indice_x*indice_y);

    add_first_elements_sort<<<1,1>>>(array_f_dev_bis, array_f_dev);

    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(array_f_dev_bis));
}

void update_array_f_dev_sort(double lambda, double* array_f_dev, double* map_dev, int indice_x, int indice_y){

    double* array_f_dev_bis = NULL;
    cudaMalloc((void**)&array_f_dev_bis, 1*sizeof(double));
    unsigned long int size_j = (indice_x)  * (indice_y)  * sizeof(double);
    double* map_norm_dev = NULL;
    cudaMalloc((void**)&map_norm_dev, size_j);

    dim3 Dg_L2, Db_L2;

    Db_L2.x = BLOCK_SIZE_X_2D_SORT;
    Db_L2.y = BLOCK_SIZE_Y_2D_SORT;
    Db_L2.z = 1;//BLOCK_SIZE_L2_Z;

    Dg_L2.x = ceil(double(indice_x)/double(BLOCK_SIZE_X_2D_SORT));
    Dg_L2.y = ceil(double(indice_y)/double(BLOCK_SIZE_Y_2D_SORT));
    Dg_L2.z = 1;//ceil(indice_x/double(BLOCK_SIZE_L2_Z));

    kernel_norm_map_simple_sort<<<Dg_L2,Db_L2>>>(lambda, map_norm_dev, map_dev, indice_x, indice_y);

    reduction_loop_parallel(map_norm_dev, array_f_dev_bis, indice_x*indice_y);

    add_first_elements_sort<<<1,1>>>(array_f_dev_bis, array_f_dev);

    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(array_f_dev_bis));
}



//beta_modif_dev
//
//
void conv2D_GPU_all_sort(parameters& M, double* d_g, int n_beta, double lambda_var_sig, double* b_params_dev, double* deriv_dev, double* beta_modif_dev, double* array_f_dev, int image_x, int image_y, int n_gauss, float temps_transfert, float temps_mirroirs)
{
//    printf("\n beta_modif_dev below \n");
//    display_dev_complete_sort<<<1,1>>>(beta_modif_dev, 30);

    dummyInstantiator_sort();
    double Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    int nb_cycles = 1;
    long int kernel_x = 3;
    long int kernel_y = 3;
    long int kernel_radius_x = 1;
    long int kernel_radius_y = 1;

    int size_i = (image_x+4)  * (image_y+4)  * sizeof(double);
    int size_j = (image_x)  * (image_y)  * sizeof(double);
    int size_k = (kernel_x)  * (kernel_y)  * sizeof(double);

    int blocks_x  = ceil(double(image_x) / double(BLOCK_SIZE_X_2D_SORT));
    int blocks_y  = ceil(double(image_y) / double(BLOCK_SIZE_Y_2D_SORT));
    int blocks_x_init = ceil(double(image_x+4) / double(BLOCK_SIZE_X_2D_SORT));
    int blocks_y_init = ceil(double(image_y+4) / double(BLOCK_SIZE_Y_2D_SORT));//2;//ceil(double(2) / double(BLOCK_SIZE_S));

    int grid_frame = ceil(double(max(image_x, image_y)) / double(256));
    dim3 ThreadsParBlock_frame, BlocksParGrille_frame;
    ThreadsParBlock_frame.x = 256;
    ThreadsParBlock_frame.y = 2;
    ThreadsParBlock_frame.z = 1;    
    BlocksParGrille_frame.x = grid_frame;
    BlocksParGrille_frame.y = 1;
    BlocksParGrille_frame.z = 1;

for(int k = 0; k<M.n_gauss; k++){

    double temps_temp = (double)(omp_get_wtime());

    double* d_IMAGE_amp;
    cudaMalloc((void**)&d_IMAGE_amp, size_j);
    double* d_IMAGE_amp_ext;
    cudaMalloc((void**)&d_IMAGE_amp_ext, size_i);

    double* d_IMAGE_mu;
    cudaMalloc((void**)&d_IMAGE_mu, size_j);
    double* d_IMAGE_mu_ext;
    cudaMalloc((void**)&d_IMAGE_mu_ext, size_i);
    
    double* d_IMAGE_sig;
    cudaMalloc((void**)&d_IMAGE_sig, size_j);
    double* d_IMAGE_sig_ext;
    cudaMalloc((void**)&d_IMAGE_sig_ext, size_i);

    double* d_conv_amp;
    cudaMalloc((void**)&d_conv_amp, size_j);
    double* d_conv_mu;
    cudaMalloc((void**)&d_conv_mu, size_j);
    double* d_conv_sig;
    cudaMalloc((void**)&d_conv_sig, size_j);

    double* d_conv_conv_amp;
    cudaMalloc((void**)&d_conv_conv_amp, size_j);
    double* d_conv_conv_mu;
    cudaMalloc((void**)&d_conv_conv_mu, size_j);
    double* d_conv_conv_sig;
    cudaMalloc((void**)&d_conv_conv_sig, size_j);

    int blocks_x  = ceil(double(image_x) / double(BLOCK_SIZE_X_2D_SORT));
    int blocks_y  = ceil(double(image_y) / double(BLOCK_SIZE_Y_2D_SORT));
    int blocks_x_init = ceil(double(image_x+4) / double(BLOCK_SIZE_X_2D_SORT));
    int blocks_y_init = ceil(double(image_y+4) / double(BLOCK_SIZE_Y_2D_SORT));//2;//ceil(double(2) / double(BLOCK_SIZE_S));

    dim3 ThreadsParBlock(BLOCK_SIZE_X_2D_SORT, BLOCK_SIZE_Y_2D_SORT);
    dim3 BlocksParGrille(blocks_x , blocks_y , 1);

    dim3 ThreadsParBlock_init(BLOCK_SIZE_X_2D_SORT, BLOCK_SIZE_Y_2D_SORT);
    dim3 BlocksParGrille_init(blocks_x_init , blocks_y_init, 1);

    parameter_maps_sliced_from_beta_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(beta_modif_dev, d_IMAGE_amp, d_IMAGE_mu, d_IMAGE_sig, image_x, image_y, k);
/*
    display_dev_complete_sort<<<1,1>>>(d_IMAGE_amp, image_x*image_y);
    display_dev_complete_sort<<<1,1>>>(d_IMAGE_mu, image_x*image_y);
    display_dev_complete_sort<<<1,1>>>(d_IMAGE_sig, image_x*image_y);
*/
    prepare_for_convolution_sort(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille_frame, ThreadsParBlock_frame);
    prepare_for_convolution_sort(d_IMAGE_mu, d_IMAGE_mu_ext, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille_frame, ThreadsParBlock_frame);
    prepare_for_convolution_sort(d_IMAGE_sig, d_IMAGE_sig_ext, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille_frame, ThreadsParBlock_frame);

    temps_mirroirs = omp_get_wtime() - temps_temp;

//    T c_Kernel[9] = {0,-1,0,-1,4,-1,0,-1,0};
//    long int c_image_x = 32;
//    long int c_image_y = 32;
//    long int c_kernel_x = 3;
//    long int c_kernel_y = 3;
//    long int c_kernel_radius_x = 1;
//    long int c_kernel_radius_y = 1;

    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\n GPU-Based 2D Image convolution");
    
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Allocating and intializing memory");

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
    
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Transfering HOST data to GPU");
    
    
    //---------------------------------------------------------------------------------------------------------------------------        
//    printf("\n\t Transfering CONSTANTS to GPU");
    
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Running GPU 2D-Convolution algorithm for %lu iterations\n", nb_cycles);
    
    conv_twice_and_copy_sort(d_IMAGE_amp_ext, d_conv_amp, d_conv_conv_amp, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock, BlocksParGrille_frame, ThreadsParBlock_frame);
    conv_twice_and_copy_sort(d_IMAGE_mu_ext, d_conv_mu, d_conv_conv_mu, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock, BlocksParGrille_frame, ThreadsParBlock_frame);
    conv_twice_and_copy_sort(d_IMAGE_sig_ext, d_conv_sig, d_conv_conv_sig, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock, BlocksParGrille_frame, ThreadsParBlock_frame);

/*
    ConvKernel_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_first_conv,  d_IMAGE_amp_ext, image_x+4, image_y+4);
    ConvKernel_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_second_conv,  d_conv_amp, image_x+4, image_y+4);
    copy_gpu_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
    copy_gpu_sort<double><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_conv_amp, d_RESULTAT_second_conv, image_x, image_y);
*/



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
//        printf("temps = %f \n", total_time);
/*
        printf("\n\t\t Time taken by memory transfer = %.2f ms", total_mem_time);
        printf("\n\t\t Upload  : %.2f GB/s  Download : %.2f GB/s",upload_speed ,download_speed );
        printf("\n\t\t Average : %.2f GB/s", average_speed);
        printf("\n\t\t memory transfer part: %.1f pct", percentage_vs_memory);
        
        printf("\n\n\t Total GPU (calc+memory) mean execution time   : %.2f ms", total_time);
        printf("\n\t Total GPU (calc+memory) mean processing speed : %.1f Mpx/s\n", 1e-6 * image_x * image_x / (total_time * 1e-3) );


    //---------------------------------------------------------------------------------------------------------------------------
    printf("\n\n\t Freeing GPU Memory \n");
*/

//    printf("\n IMAGE EXT\n");
/*
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_amp_ext, image_x+4,image_y+4);
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_mu_ext, image_x+4,image_y+4);
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_sig_ext, image_x+4,image_y+4);

    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_mu, image_x,image_y);
    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_sig, image_x,image_y);



    display_dev_complete_2D_sort<<<1,1>>>(d_conv_conv_amp, image_x, image_y);
    display_dev_complete_2D_sort<<<1,1>>>(d_conv_conv_mu, image_x,image_y);
    display_dev_complete_2D_sort<<<1,1>>>(d_conv_conv_sig, image_x,image_y);
*/
//if(image_x != image_y){
//    printf("\n IMAGE \n");
//    display_dev_complete_2D_sort<<<1,1>>>(d_IMAGE_amp, image_x,image_y);
//    printf("\n CONV \n");
//    display_dev_complete_2D_sort<<<1,1>>>(d_conv_amp, image_x, image_y);
//}

//    display_dev_complete_2D_sort<<<1,1>>>(d_conv_mu, image_x,image_y);
//    display_dev_complete_2D_sort<<<1,1>>>(d_conv_sig, image_x,image_y);

//  exit(0);

    update_array_f_dev_sort(M.lambda_amp ,array_f_dev, d_conv_amp, image_x, image_y);
    update_array_f_dev_sort(M.lambda_mu ,array_f_dev, d_conv_mu, image_x, image_y);
    update_array_f_dev_sort(M.lambda_sig, M.lambda_var_sig, array_f_dev, d_IMAGE_sig, d_conv_sig, image_x, image_y, k, b_params_dev);


    dim3 Dg_ud, Db_ud;
    Db_ud.x = BLOCK_SIZE_X_2D_SORT; //x
    Db_ud.y = BLOCK_SIZE_Y_2D_SORT; //y
    Dg_ud.x = ceil(double(image_x)/double(BLOCK_SIZE_X_2D_SORT));
    Dg_ud.y = ceil(double(image_y)/double(BLOCK_SIZE_Y_2D_SORT));
//    printf("Dg_ud = %d, %d, %d ; Db_ud = %d, %d, %d\n",Dg_ud.x,Dg_ud.y,Dg_ud.z,Db_ud.x,Db_ud.y,Db_ud.z);
    kernel_update_deriv_conv_conv_sort<<<Dg_ud,Db_ud>>>(deriv_dev, M.lambda_amp, M.lambda_mu, M.lambda_sig, M.lambda_var_sig, d_conv_conv_amp, d_conv_conv_mu, d_conv_conv_sig, d_IMAGE_sig, b_params_dev, int(image_y), int(image_x),k);

    double* d_image_sigma_reduc_f = NULL;
    checkCudaErrors(cudaMalloc(&d_image_sigma_reduc_f, 1*sizeof(double)));

    reduction_loop_parallel(d_IMAGE_sig, d_image_sigma_reduc_f, image_y*image_x);

    kernel_conv_g_sort<<<Dg_ud,Db_ud>>>(n_beta, d_g, d_IMAGE_sig, M.lambda_var_sig, n_gauss, b_params_dev, k, image_x, image_y);



/*
   reduction_loop_parallel(d_image_sigma_reduc, d_image_sigma_reduc_f, product_image_conv);
   g[n_beta − M.n_gauss + k]+ = lambda var sig × (b params[index k] − image sig[index y][index x])

   display_dev_sort<<<1,1>>>(deriv_dev);
    printf("\n END OF DISPLAY \n");
*/

//£



    checkCudaErrors(cudaFree(d_image_sigma_reduc_f));
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






























void gradient_L_2_beta_parallel(double* deriv_dev, int* taille_deriv, int* taille_deriv_dev, double* beta_modif_dev, int* taille_beta_modif_dev, double* residual_dev, int* taille_residual_dev, double* std_map_dev, int* taille_std_map_dev, int n_gauss)
{
   dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X_GRAD1; //x
    Db.y = BLOCK_SIZE_Y_GRAD1; //y
    Db.z = BLOCK_SIZE_Z_GRAD1; //gaussiennes
        //deriv_dev      --> (3g,y,x)  --> (z,y,x)
        //params     --> (3g,y,x)  --> (z,y,x)
    Dg.x = ceil(double(taille_deriv[2])/double(BLOCK_SIZE_X_GRAD1));
    Dg.y = ceil(double(taille_deriv[1])/double(BLOCK_SIZE_Y_GRAD1));
    Dg.z = ceil(double(n_gauss)/double(BLOCK_SIZE_Z_GRAD1));

//  printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);

//gradient_kernel_2_beta_working_sort<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, beta_modif_dev, taille_beta_modif_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);

//    display_dev_complete_sort<<<1,1>>>(beta_modif_dev,taille_deriv[2]*taille_deriv[1]*taille_deriv[0]);

//display_dev_sort<<<1,1>>>(beta_modif_dev);
//ùùù
//    display_dev_complete_sort<<<1,1>>>(beta_modif_dev,10);

    gradient_kernel_2_beta_with_INDEXING_sort<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, beta_modif_dev, taille_beta_modif_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);

//  gradient_kernel_2_beta_sort<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, beta_modif_dev, taille_beta_modif_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);
//  printf("DEBUG !!\n");
//  printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);


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
void reduction_loop_parallel(double* array_in, double* d_array_f, int size_array){

    int GRID_SIZE_REDUCTION = int(ceil(double(size_array)/double(BLOCK_SIZE_REDUCTION)));
    int N = ceil(log(double(size_array))/log(double(BLOCK_SIZE_REDUCTION)));

//if (N==1){
if (true){//N==1){
    reduce_last_in_one_thread_sort<<<1,1>>>(array_in, d_array_f, size_array);
//    cpy_first_num_dev_sort<<<1,1>>>( array_out_kernel, d_array_f);
/*
    double* array_in_copied = NULL;
    checkCudaErrors(cudaMalloc(&array_in_copied, size_array*sizeof(double)));
    copy_dev_sort<<< GRID_SIZE_REDUCTION , BLOCK_SIZE_REDUCTION >>>(array_in, array_in_copied, size_array);

    int size_array_out_kernel = ceil(double(size_array)/double(BLOCK_SIZE_REDUCTION));
    int copy_dev_blocks = ceil(double(size_array_out_kernel)/double(BLOCK_SIZE_REDUCTION));
    double* array_out_kernel=NULL;
    checkCudaErrors(cudaMalloc(&array_out_kernel, size_array_out_kernel*sizeof(double)));

    sum_reduction_sort<<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in_copied, array_out_kernel, size_array);

    cpy_first_num_dev_sort<<<1,1>>>( array_out_kernel, d_array_f);
    cudaFree(array_in_copied);
    cudaFree(array_out_kernel);
*/

} else{
    double* array_in_copied = NULL;
    checkCudaErrors(cudaMalloc(&array_in_copied, size_array*sizeof(double)));
    copy_dev_sort<<< GRID_SIZE_REDUCTION , BLOCK_SIZE_REDUCTION >>>(array_in, array_in_copied, size_array);

    int size_array_out_kernel = ceil(double(size_array)/double(BLOCK_SIZE_REDUCTION));
    double* array_out_kernel=NULL;
    checkCudaErrors(cudaMalloc(&array_out_kernel, size_array_out_kernel*sizeof(double)));

    sum_reduction_sort<<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in_copied, array_out_kernel, size_array);

    cudaFree(array_in_copied);
    double* array_in_copied_2;
    checkCudaErrors(cudaMalloc(&array_in_copied_2, size_array_out_kernel*sizeof(double)));

    int copy_dev_blocks = ceil(double(size_array_out_kernel)/double(BLOCK_SIZE_REDUCTION));
    copy_dev_sort<<< copy_dev_blocks , BLOCK_SIZE_REDUCTION >>>(array_out_kernel, array_in_copied_2, size_array_out_kernel);

    cudaFree(array_out_kernel);

    double size_array_out_kernel_2 = ceil(double(size_array)/double(pow(BLOCK_SIZE_REDUCTION,2)));
    double* array_out_kernel_2=NULL;
    checkCudaErrors(cudaMalloc(&array_out_kernel_2, size_array_out_kernel_2*sizeof(double)));

    sum_reduction_sort<<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(array_in_copied_2, array_out_kernel_2, size_array_out_kernel);

    if(N>2){
      reduce_last_in_one_thread_sort<<<1,1>>>(array_out_kernel_2, d_array_f, size_array_out_kernel_2);
    }
    else{
      cpy_first_num_dev_sort<<<1,1>>>( array_out_kernel_2, d_array_f);
    }
    cudaFree(array_in_copied_2);
    cudaFree(array_out_kernel_2);
}


}	










//f = compute_residual_and_f(beta_modif_dev, cube_flattened_dev, residual_dev, std_map_dev, indice_x, indice_y, indice_v, M.n_gauss);
void compute_residual_and_f_parallel(double* array_f_dev, double* beta_dev, double* cube_dev, double* residual_dev, double* std_map_dev, int indice_x, int indice_y, int indice_v, int n_gauss)
  {
   dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X_SORT; //
    Db.y = BLOCK_SIZE_Y_SORT; //
    Db.z = BLOCK_SIZE_Z_SORT; //

    Dg.x = ceil(double(indice_x)/double(BLOCK_SIZE_X_SORT));
    Dg.y = ceil(double(indice_y)/double(BLOCK_SIZE_Y_SORT));
    Dg.z = ceil(double(indice_v)/double(BLOCK_SIZE_Z_SORT));

// index_x -> indice_x
// index_y -> indice_y
// index_z -> indice_v
//    printf("beta[0] = %f \n", beta[0]);

    kernel_residual_sort<<<Dg,Db>>>(beta_dev, cube_dev, residual_dev,indice_x, indice_y, indice_v, n_gauss);


//    printf("residual[0] = %f \n",residual[0]);



    dim3 Dg_L2, Db_L2;
    Db_L2.x = BLOCK_SIZE_X_2D_SORT;
    Db_L2.y = BLOCK_SIZE_Y_2D_SORT;
    Db_L2.z = 1;//BLOCK_SIZE_L2_Z;
    Dg_L2.x = ceil(double(indice_x)/double(BLOCK_SIZE_X_2D_SORT));
    Dg_L2.y = ceil(double(indice_y)/double(BLOCK_SIZE_Y_2D_SORT));
    Dg_L2.z = 1;//ceil(indice_x/double(BLOCK_SIZE_L2_Z));

    double* map_norm_dev = NULL;
    checkCudaErrors(cudaMalloc(&map_norm_dev, indice_x*indice_y*sizeof(double)));

    kernel_norm_map_boucle_v_sort<<<Dg_L2, Db_L2>>>(map_norm_dev, residual_dev, std_map_dev, indice_x, indice_y, indice_v);


/*
//vérification kernel_norm_map_boucle_v
    double* map_norm_host = NULL;
    map_norm_host = (double*)malloc(indice_x*indice_y*sizeof(double));
    checkCudaErrors(cudaMemcpy(map_norm_host, map_norm_dev, indice_x*indice_y*sizeof(double), cudaMemcpyDeviceToHost));
    for(int p = 0; p<indice_x*indice_y; p++)
    {
      f+= map_norm_host[p];
    }
    printf("f = %f \n", f);
exit(0);
  
    printf("indice_x = %d , indice_y = %d , indice_v = %d , BLOCK_SIZE_REDUCTION = %d \n", indice_x, indice_y, indice_v, BLOCK_SIZE_REDUCTION);
    printf("int(ceil(double(indice_x*indice_y)/double(BLOCK_SIZE_REDUCTION))) = %d \n", int(ceil(double(indice_x*indice_y)/double(BLOCK_SIZE_REDUCTION))));
    printf("Dg = %d , Db = %d\n",int(ceil(double(indice_x*indice_y)/double(BLOCK_SIZE_REDUCTION))), BLOCK_SIZE_REDUCTION);
*/

/*
    int GRID_SIZE_REDUCTION = int(ceil(double(indice_x*indice_y)/double(BLOCK_SIZE_REDUCTION)));

    double* tab_cpy_cpu = NULL;
    tab_cpy_cpu = (double*)malloc(GRID_SIZE_REDUCTION*sizeof(double));

    double* tab_test_dev_out=NULL;
    checkCudaErrors(cudaMalloc(&tab_test_dev_out, GRID_SIZE_REDUCTION*sizeof(double)));

    sum_reduction_sort<<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(map_norm_dev, tab_test_dev_out, indice_x*indice_y);

    checkCudaErrors(cudaMemcpy(tab_cpy_cpu, tab_test_dev_out, GRID_SIZE_REDUCTION*sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.;
    for(int o = 0; o<GRID_SIZE_REDUCTION ; o++)
    {
      sum+=tab_cpy_cpu[o];
    }

    double* d_array_f=NULL;
    checkCudaErrors(cudaMalloc(&d_array_f, 1*sizeof(double))); // ERREUR ICI
*/

    reduction_loop_parallel(map_norm_dev, array_f_dev, indice_x*indice_y);

/*
    sum_sort<<<1,1>>>(map_norm_dev, indice_x*indice_y);

    double* array_f = (double*)malloc(1*sizeof(double));
    checkCudaErrors(cudaMemcpy(array_f, d_array_f, 1*sizeof(double), cudaMemcpyDeviceToHost));

    printf("--> array_f[0] = %f\n",array_f[0]);
    printf("sum = %f\n",sum);
    free(array_f);
*/


//limite taille à 3300*3300 :/
/*
    int N_test = 1.1*1e7;
    double* test_cpu = (double*)malloc(N_test*sizeof(double));
    double* test_gpu = (double*)malloc(N_test*sizeof(double));
    double* f_cpu = (double*)malloc(1*sizeof(double));
    double* d_array_f_2=NULL;
    checkCudaErrors(cudaMalloc(&d_array_f_2, 1*sizeof(double))); // ERREUR ICI
    for(int i = 0; i<N_test; i++){
       test_cpu[i] = 0.001;
    }
    double* d_test=NULL;
    checkCudaErrors(cudaMalloc(&d_test, N_test*sizeof(double))); // ERREUR ICI
    checkCudaErrors(cudaMemcpy(d_test, test_cpu, N_test*sizeof(double), cudaMemcpyHostToDevice));
    double temps_1 = omp_get_wtime();
    reduction_loop_parallel(d_test, d_array_f_2, N_test);
    double temps_2 = omp_get_wtime();
    checkCudaErrors(cudaMemcpy(f_cpu, d_array_f_2, 1*sizeof(double), cudaMemcpyDeviceToHost));
    double sum_test = 0;
    double temps_3 = omp_get_wtime();
    for(int i = 0; i<N_test; i++){
       sum_test += test_cpu[i];
    }
    double temps_4 = omp_get_wtime();
    printf("temps cpu =     %f\n",temps_4-temps_3);
    printf("temps gpu =     %f\n",temps_2-temps_1);
    printf("sum_test =     %f\n",sum_test);
    printf("d_array_f[0] = %f\n",f_cpu[0]);
*/


/*
    printf("beta[0] = %f \n",beta[0]);
    printf("residual[0] = %f \n",residual[0]);
    printf("residual[1] = %f \n",residual[1]);
    printf("residual[2] = %f \n",residual[2]);
    printf("residual[3] = %f \n",residual[3]);
*/

//    checkCudaErrors(cudaFree(tab_test_dev_out));
//    checkCudaErrors(cudaFree(d_array_f));
    checkCudaErrors(cudaFree(map_norm_dev));

//    init_dev_sort<<<1,1>>>(array_f_dev, sum);
//    cpy_first_num_dev_sort<<<1,1>>>(d_array_f,array_f_dev);
//    init_dev_sort<<<1,1>>>(array_f_dev, );

  }










/*

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

*/


void f_g_cube_parallel(parameters &M, double &f, double* g, int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, double* cube_flattened, double temps_conv, double temps_deriv, double temps_tableaux, double temps_res_f)   
  {
//    printf("indice_x = %d , indice_y = %d , indice_v = %d \n", indice_x, indice_y, indice_v);

    std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));


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

//exit(0);
	double temps_modification_beta1 = omp_get_wtime();

	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for(p=0; p<3*M.n_gauss; p++){
				beta_modif[j*indice_y*3*M.n_gauss+i*3*M.n_gauss+p] = beta[j*indice_y*3*M.n_gauss+i*3*M.n_gauss+p];
			}
		}
	}

   double temps_modification_beta2 = omp_get_wtime();

   double temps2_tableaux = omp_get_wtime();

   double* d_image_sigma_reduc = NULL;
   checkCudaErrors(cudaMalloc(&d_image_sigma_reduc, product_image_conv*sizeof(double)));
   double* d_g = NULL;
   checkCudaErrors(cudaMalloc(&d_g,n_beta*sizeof(double)));
   checkCudaErrors(cudaMemcpy(d_g, g, n_beta*sizeof(double), cudaMemcpyHostToDevice));

   double* beta_modif_dev = NULL;
   checkCudaErrors(cudaMalloc(&beta_modif_dev, product_beta_modif*sizeof(double)));
   checkCudaErrors(cudaMemcpy(beta_modif_dev, beta_modif, product_beta_modif*sizeof(double), cudaMemcpyHostToDevice));

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

//if(indice_x ==4){//indice_x != indice_y){
//    display_dev_sort<<<1,1>>>(array_f_dev);
//}

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


//if(indice_x ==4){//indice_x != indice_y){
//   printf("indice_x, indice_y = %d, %d\n", indice_x, indice_y);
//   display_dev_complete_sort<<<1,1>>>(deriv_dev,30);
//}
/*
//    display_dev_sort<<<1,1>>>(deriv_dev);
//    display_dev_sort<<<1,1>>>(residual_dev);
//    display_dev_sort<<<1,1>>>(cube_dev);
//  display_dev_sort<<<1,1>>>(deriv_dev);
*/

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

/*    
	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				printf("deriv[%d] = %f \n", l*indice_y*indice_x+i*indice_x+j, deriv[l*indice_y*indice_x+i*indice_x+j]);// + dR_over_dB[l][i][j];
			}
		}
	}
	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				g_3D[l][i][j] = deriv[l*indice_y*indice_x+i*indice_x+j];// + dR_over_dB[l][i][j];
			}
		}
	}
    for(int k(0); k<indice_x; k++)
    {
        for(int j(0); j<indice_y; j++)
        {
	        for(int i(0); i<3*M.n_gauss; i++)
			{
	            g[k*indice_y*(3*M.n_gauss)+j*(3*M.n_gauss)+i] += g_3D[i][j][k];
	    	}
        }
    }
*/

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

//if(indice_x ==4){//indice_x != indice_y){
//	printf("après conv \n");

//	printf("Avant display b_params \n");
//    display_dev_complete_sort<<<1,1>>>(b_params_dev, 20);

//	for(int i=0; i<indice_x*indice_y*3*M.n_gauss; i++){
//		printf("g[%d] = %f \n",i,g[i]);
//	}
//   	printf("f = %f \n",f);

//    std::cin.ignore();   
//}


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


//checkCudaErrors(cudaMemcpy(residual, residual_dev, product_residual*sizeof(double), cudaMemcpyDeviceToHost));

}