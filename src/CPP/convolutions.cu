#include "convolutions.hpp"
#include "kernel_conv.cu"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

void conv_twice_and_copy(double* d_IMAGE_amp, double* d_IMAGE_amp_ext, double* d_conv_amp, double* d_conv_conv_amp, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille, dim3 ThreadsParBlock)
{
    unsigned long int size_i = (image_x+4)  * (image_y+4)  * sizeof(double);

    double* d_RESULTAT_first_conv;        
    cudaMalloc((void**)&d_RESULTAT_first_conv, size_i);
    cudaMemset ( d_RESULTAT_first_conv, 0 , size_i) ;

    double* d_RESULTAT_second_conv;        
    cudaMalloc((void**)&d_RESULTAT_second_conv, size_i);
    cudaMemset ( d_RESULTAT_second_conv, 0 , size_i) ;

    ConvKernel<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_first_conv,  d_IMAGE_amp_ext, image_x+4, image_y+4);
    ConvKernel<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_second_conv,  d_conv_amp, image_x+4, image_y+4);
    copy_gpu<double><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
    copy_gpu<double><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_conv_amp, d_RESULTAT_second_conv, image_x, image_y);
}


void prepare_for_convolution(float* d_IMAGE_amp, float* d_IMAGE_amp_ext, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille, dim3 ThreadsParBlock)
{
/*
	printf("BlocksParGrille.x = %d, BlocksParGrille.y = %d, BlocksParGrille.z = %d\n", BlocksParGrille.x, BlocksParGrille.y, BlocksParGrille.z);
    printf("ThreadsParBlock.x = %d, ThreadsParBlock.y = %d, ThreadsParBlock.z = %d\n", ThreadsParBlock.x, ThreadsParBlock.y, ThreadsParBlock.z);
    printf("BlocksParGrille_init.x = %d, BlocksParGrille_init.y = %d, BlocksParGrille_init.z = %d\n", BlocksParGrille_init.x, BlocksParGrille_init.y, BlocksParGrille_init.z);
    printf("ThreadsParBlock_init.x = %d, ThreadsParBlock_init.y = %d, ThreadsParBlock_init.z = %d\n", ThreadsParBlock_init.x, ThreadsParBlock_init.y, ThreadsParBlock_init.z);
*/
//    print_device_array<float><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp, image_x, image_y);
    init_extended_array<float><<<BlocksParGrille_init,ThreadsParBlock_init>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
//    print_device_array<float><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp, image_x, image_y);
    extension_mirror_gpu<float><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
//    print_device_array<float><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp_ext, image_x+4, image_y+4);
}

void prepare_for_convolution(double* d_IMAGE_amp, double* d_IMAGE_amp_ext, int image_x, int image_y, dim3 BlocksParGrille_init, dim3 ThreadsParBlock_init, dim3 BlocksParGrille, dim3 ThreadsParBlock)
{
/*
	printf("BlocksParGrille.x = %d, BlocksParGrille.y = %d, BlocksParGrille.z = %d\n", BlocksParGrille.x, BlocksParGrille.y, BlocksParGrille.z);
    printf("ThreadsParBlock.x = %d, ThreadsParBlock.y = %d, ThreadsParBlock.z = %d\n", ThreadsParBlock.x, ThreadsParBlock.y, ThreadsParBlock.z);
    printf("BlocksParGrille_init.x = %d, BlocksParGrille_init.y = %d, BlocksParGrille_init.z = %d\n", BlocksParGrille_init.x, BlocksParGrille_init.y, BlocksParGrille_init.z);
    printf("ThreadsParBlock_init.x = %d, ThreadsParBlock_init.y = %d, ThreadsParBlock_init.z = %d\n", ThreadsParBlock_init.x, ThreadsParBlock_init.y, ThreadsParBlock_init.z);
*/
//    print_device_array<double><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp, image_x, image_y);
    init_extended_array<double><<<BlocksParGrille_init,ThreadsParBlock_init>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
//    print_device_array<double><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp, image_x, image_y);
    extension_mirror_gpu<double><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y);
//    print_device_array<double><<<BlocksParGrille,ThreadsParBlock>>>(d_IMAGE_amp_ext, image_x+4, image_y+4);
}

void dummyInstantiator(){
    ConvKernel<float><<<1,1>>>(NULL, NULL, 0, 0);
    copy_gpu<float><<<1,1>>>(NULL,NULL,0,0);
    extension_mirror_gpu<float><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror<float>(NULL, NULL, 0, 0);
    init_extended_array<float><<<1,1>>>(NULL, NULL, 0, 0);
    parameter_maps_sliced_from_beta<float><<<1,1>>>(NULL,NULL,0,0,0,0,0);
    print_device_array<float><<<1,1>>>(NULL,0,0);
    parameter_maps_sliced_from_beta<float><<<1,1>>>(NULL, NULL, NULL, NULL, 0, 0, 0);

    ConvKernel<double><<<1,1>>>(NULL, NULL, 0, 0);
    copy_gpu<double><<<1,1>>>(NULL,NULL,0,0);
    extension_mirror_gpu<double><<<1,1>>>(NULL, NULL, 0, 0);
    extension_mirror<double>(NULL, NULL, 0, 0);
    init_extended_array<double><<<1,1>>>(NULL, NULL, 0, 0);
    parameter_maps_sliced_from_beta<double><<<1,1>>>(NULL,NULL,0,0,0,0,0);
    print_device_array<double><<<1,1>>>(NULL,0,0);
    parameter_maps_sliced_from_beta<double><<<1,1>>>(NULL, NULL, NULL, NULL, 0, 0, 0);
}

void conv2D_GPU(double *h_IMAGE, double* h_KERNEL, double* h_RESULTAT_GPU, long int image_x, long int image_y, float temps_transfert, float temps_mirroirs)
{
    dummyInstantiator();
    double Kernel[9] = {0,-1,0,-1,4,-1,0,-1,0};
    long int kernel_x = 3;
    long int kernel_y = 3;
    long int kernel_radius_x = 1;
    long int kernel_radius_y = 1;

    int nb_cycles = 1;

    double* h_IMAGE_extended = (double*)malloc((image_x+4)*(image_y+4)*sizeof(double));

	for(int j(0); j<image_y+4; j++)
	{
		for(int i(0); i<image_x+4; i++)
		{
			h_IMAGE_extended[(image_x+4)*(j)+i]=0.;
		}
	}

	for(int j(0); j<image_y; j++)
	{
		for(int i(0); i<image_x; i++)
		{
			h_IMAGE_extended[(image_x+4)*(2+j)+i+2]=h_IMAGE[image_x*j+i];
		}
	}

	for(int j(0); j<2; j++)
	{
		for(int i(0); i<image_x; i++)
		{
			h_IMAGE_extended[(image_x+4)*j+i+2] = h_IMAGE[image_x*j+i];
		}
	}

	for(int j(0); j<image_y; j++)
	{
    	for(int i(0); i<2; i++)
		{
			h_IMAGE_extended[(image_x+4)*(2+j)+i] = h_IMAGE[image_x*j+i];
		}
	}

   	for(int j=image_y; j<image_y+2; j++)
	{
    	for(int i=0; i<image_x; i++)
		{
			h_IMAGE_extended[(image_x+4)*(2+j)+i+2]=h_IMAGE[image_x*(j-2)+i];
		}
	}

	for(int j(0); j<image_y; j++)
	{
		for(int i=image_x; i<image_x+2; i++)
		{
			h_IMAGE_extended[(image_x+4)*(2+j)+2+i]=h_IMAGE[image_x*j+i-2];
		}
	}
    

/*
    printf("h_IMAGE_extended : \n");
	for(int j(0); j<image_y+4; j++)
	{
		for(int i(0); i<image_x+4; i++)
		{
            printf("%f ", h_IMAGE_extended[i+(image_x+4)*j]);
		}
        printf("\n");
	}
    exit(0);
*/


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

        unsigned long int size_i = (image_x+4)  * (image_y+4)  * sizeof(double);
        unsigned long int size_j = (image_x)  * (image_y)  * sizeof(double);
        unsigned long int size_k = kernel_x * kernel_y * sizeof(double);
    
        double* d_IMAGE;            
        cudaMalloc((void**)&d_IMAGE,    size_i);
        double* d_RESULTAT;        
        cudaMalloc((void**)&d_RESULTAT, size_i);
        double* d_RESULTAT_RESHAPED;        
        cudaMalloc((void**)&d_RESULTAT_RESHAPED, size_j);  
        
        
        
        cudaMemset ( d_RESULTAT, 0 , size_i) ;
        cudaMemset ( d_IMAGE, 0 , size_i) ;
                
     checkCudaErrors(cudaEventSynchronize(record_event[1]));    
     checkCudaErrors(cudaEventRecord(record_event[1], NULL));
    
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Transfering HOST data to GPU");
    
    cudaMemcpy(d_IMAGE,    h_IMAGE_extended,   size_i, cudaMemcpyHostToDevice);
    
    //---------------------------------------------------------------------------------------------------------------------------        
//    printf("\n\t Transfering CONSTANTS to GPU");
    
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Running GPU 2D-Convolution algorithm for %lu iterations\n", nb_cycles);
        
        int    threads = BLOCK_SIZE;
        int blocks_x  = (image_x+4) / BLOCK_SIZE;
        int blocks_y  = (image_y+4) / BLOCK_SIZE;

        dim3 ThreadsParBlock(threads, threads);
        dim3 BlocksParGrille(blocks_x , blocks_y );

        ConvKernel<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT,  d_IMAGE, image_x+4, image_y+4);

            checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));    

    copy_gpu<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_RESHAPED, d_RESULTAT, image_x, image_y);

    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Transfering GPU Data back to HOST");
    
    
    cudaMemcpy(h_RESULTAT_GPU, d_RESULTAT_RESHAPED, size_j, cudaMemcpyDeviceToHost);
            
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

    cudaFree(d_IMAGE);
    cudaFree(d_RESULTAT);
    cudaFree(d_RESULTAT_RESHAPED);

    free(h_IMAGE_extended);
}












void conv2D_GPU(float *h_IMAGE, float* h_KERNEL, float* h_RESULTAT_GPU, long int image_x, long int image_y, float temps_transfert, float temps_mirroirs)
{
    dummyInstantiator();
    float Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    long int kernel_x = 3;
    long int kernel_y = 3;
    long int kernel_radius_x = 1;
    long int kernel_radius_y = 1;

    double temps_temp = omp_get_wtime();

    float* h_IMAGE_extended = (float*)malloc((image_x+4)*(image_y+4)*sizeof(float));

    //extension_mirror<float>(h_IMAGE, h_IMAGE_extended, image_x, image_y);
	for(int j(0); j<image_y+4; j++)
	{
		for(int i(0); i<image_x+4; i++)
		{
			h_IMAGE_extended[i+(image_x+4)*(j)]=0.;
		}
	}
    
	for(int j(0); j<image_y; j++)
	{
		for(int i(0); i<image_x; i++)
		{
			h_IMAGE_extended[2+i+(image_x+4)*(2+j)]=h_IMAGE[i+image_x*j];
		}
	}

	for(int j(0); j<2; j++)
	{
		for(int i(0); i<image_x; i++)
		{
			h_IMAGE_extended[2+i+(image_x+4)*j] = h_IMAGE[i+image_x*j];
		}
	}

	for(int j(0); j<image_y; j++)
	{
    	for(int i(0); i<2; i++)
		{
			h_IMAGE_extended[i+(image_x+4)*(2+j)] = h_IMAGE[i+image_x*j];
		}
	}

   	for(int j=image_y; j<image_y+2; j++)
	{
    	for(int i=0; i<image_x; i++)
		{
			h_IMAGE_extended[2+i+(image_x+4)*(2+j)]=h_IMAGE[i+image_x*(j-2)];
		}
	}

	for(int j(0); j<image_y; j++)
	{
		for(int i=image_x; i<image_x+2; i++)
		{
			h_IMAGE_extended[2+i+(image_x+4)*(2+j)]=h_IMAGE[i-2+image_x*j];
		}
	}

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

        unsigned long int size_i = (image_x+4)  * (image_y+4)  * sizeof(float);
        unsigned long int size_j = (image_x)  * (image_y)  * sizeof(float);
        unsigned long int size_k = kernel_x * kernel_y * sizeof(float);
    
        float* d_IMAGE;            
        cudaMalloc((void**)&d_IMAGE,    size_i);
        float* d_RESULTAT;        
        cudaMalloc((void**)&d_RESULTAT, size_i);
        float* d_RESULTAT_RESHAPED;        
        cudaMalloc((void**)&d_RESULTAT_RESHAPED, size_j);  

        cudaMemset ( d_RESULTAT, 0 , size_i) ;
        cudaMemset ( d_IMAGE, 0 , size_i) ;
                
     checkCudaErrors(cudaEventSynchronize(record_event[1]));    
     checkCudaErrors(cudaEventRecord(record_event[1], NULL));
    
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Transfering HOST data to GPU");
    
    cudaMemcpy(d_IMAGE, h_IMAGE_extended,   size_i, cudaMemcpyHostToDevice);
    
    //---------------------------------------------------------------------------------------------------------------------------        
//    printf("\n\t Transfering CONSTANTS to GPU");
    
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Running GPU 2D-Convolution algorithm for %lu iterations\n", nb_cycles);
        
        int    threads = BLOCK_SIZE;
        int blocks_x  = ceil(double(image_x+4) / double(BLOCK_SIZE));
        int blocks_y  = ceil(double(image_y+4) / double(BLOCK_SIZE));

        dim3 ThreadsParBlock(threads, threads);
        dim3 BlocksParGrille(blocks_x , blocks_y );

        ConvKernel<<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT,  d_IMAGE, image_x+4, image_y+4);

        checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));    

    copy_gpu<float><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_RESHAPED, d_RESULTAT, image_x, image_y);

    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Transfering GPU Data back to HOST\n");
    
    cudaMemcpy(h_RESULTAT_GPU, d_RESULTAT_RESHAPED, size_j, cudaMemcpyDeviceToHost);
            
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

    cudaFree(d_IMAGE);
    cudaFree(d_RESULTAT);
    cudaFree(d_RESULTAT_RESHAPED);
    free(h_IMAGE_extended);
   
}















void conv2D_GPU_cpu(float *h_IMAGE, float* h_KERNEL, float* h_RESULTAT_GPU, long int image_x, long int image_y, float temps_transfert, float temps_mirroirs)
{
    dummyInstantiator();
    float Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    long int kernel_x = 3;
    long int kernel_y = 3;
    long int kernel_radius_x = 1;
    long int kernel_radius_y = 1;

    int nb_cycles = 1;

    double temps_temp = omp_get_wtime();

    float* h_IMAGE_extended = (float*)malloc((image_x+4)*(image_y+4)*sizeof(float));

    extension_mirror<float>(h_IMAGE, h_IMAGE_extended, image_x, image_y);
    
/*
    for(int j=0; j<image_x; j++)
	{
		for(int i=0; i<image_y; i++)
		{
			printf("h_IMAGE[%d] = %f\n", i+(image_x)*(j), h_IMAGE[i+(image_x)*(j)]);
		}
	}
*/
    for(int j=0; j<image_x+4; j++)
	{
		for(int i=0; i<image_y+4; i++)
		{
			printf("h_IMAGE_extended[%d] = %f\n", i+(image_x+4)*(j), h_IMAGE_extended[i+(image_x+4)*(j)]);
		}
	}
    exit(0);


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

        unsigned long int size_i = (image_x+4)  * (image_y+4)  * sizeof(float);
        unsigned long int size_j = (image_x)  * (image_y)  * sizeof(float);
        unsigned long int size_k = kernel_x * kernel_y * sizeof(float);
    
        float* d_IMAGE;            
        cudaMalloc((void**)&d_IMAGE,    size_i);
        float* d_RESULTAT;        
        cudaMalloc((void**)&d_RESULTAT, size_i);
        float* d_RESULTAT_RESHAPED;        
        cudaMalloc((void**)&d_RESULTAT_RESHAPED, size_j);  

        cudaMemset ( d_RESULTAT, 0 , size_i) ;
        cudaMemset ( d_IMAGE, 0 , size_i) ;
                
     checkCudaErrors(cudaEventSynchronize(record_event[1]));    
     checkCudaErrors(cudaEventRecord(record_event[1], NULL));
    
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Transfering HOST data to GPU");
    
    cudaMemcpy(d_IMAGE, h_IMAGE_extended,   size_i, cudaMemcpyHostToDevice);
    
    //---------------------------------------------------------------------------------------------------------------------------        
//    printf("\n\t Transfering CONSTANTS to GPU");
    
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Running GPU 2D-Convolution algorithm for %lu iterations\n", nb_cycles);
        
        int    threads = BLOCK_SIZE;
        int blocks_x  = ceil(double(image_x+4) / double(BLOCK_SIZE));
        int blocks_y  = ceil(double(image_y+4) / double(BLOCK_SIZE));

        dim3 ThreadsParBlock(threads, threads);
        dim3 BlocksParGrille(blocks_x , blocks_y );

        for(int i = 0 ; i < (nb_cycles) ; i++)
            ConvKernel<<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT,  d_IMAGE, image_x+4, image_y+4);
//        convolve_global<<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT,  d_IMAGE, image_x+4, image_y+4);

        checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));    

    copy_gpu<<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_RESHAPED, d_RESULTAT, image_x, image_y);

    //---------------------------------------------------------------------------------------------------------------------------
//    printf("\n\t Transfering GPU Data back to HOST\n");
    
    cudaMemcpy(h_RESULTAT_GPU, d_RESULTAT_RESHAPED, size_j, cudaMemcpyDeviceToHost);
            
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

    cudaFree(d_IMAGE);
    cudaFree(d_RESULTAT);
    cudaFree(d_RESULTAT_RESHAPED);

    free(h_IMAGE_extended);
}


















































//beta_modif 
//
//
void conv2D_GPU_all(double* beta_modif, double *h_IMAGE, double* h_KERNEL, double* h_RESULTAT_GPU, long int image_x, long int image_y, int n_gauss, float temps_transfert, float temps_mirroirs)
{
    dummyInstantiator();
    double Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    int nb_cycles = 1;
    long int kernel_x = 3;
    long int kernel_y = 3;
    long int kernel_radius_x = 1;
    long int kernel_radius_y = 1;

    unsigned long int size_i = (image_x+4)  * (image_y+4)  * sizeof(double);
    unsigned long int size_j = (image_x)  * (image_y)  * sizeof(double);
    unsigned long int size_k = (kernel_x)  * (kernel_y)  * sizeof(double);

    double temps_temp = (double)(omp_get_wtime());
    double* h_IMAGE_extended = (double*)malloc((image_x+4)*(image_y+4)*sizeof(double));

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

    int threads = BLOCK_SIZE;
    int blocks_x  = ceil(double(image_x+4) / double(BLOCK_SIZE));
    int blocks_y  = ceil(double(image_y+4) / double(BLOCK_SIZE));
    int blocks_x_init = ceil(double(max(image_x+4, image_y)) / double(BLOCK_SIZE));
    int blocks_y_init = 2;//ceil(double(2) / double(BLOCK_SIZE));

    dim3 ThreadsParBlock(threads, threads);
    dim3 BlocksParGrille(blocks_x , blocks_y );

    dim3 ThreadsParBlock_init(threads, 2);
    dim3 BlocksParGrille_init(blocks_x_init , blocks_y_init);

    checkCudaErrors(cudaMemcpy(d_IMAGE_amp, h_IMAGE, size_j, cudaMemcpyHostToDevice));

//    parameter_maps_sliced_from_beta<double><<<BlocksParGrille, ThreadsParBlock>>>(beta_modif, d_IMAGE_amp, d_IMAGE_mu, d_IMAGE_sig, image_x, image_y, 0);

    prepare_for_convolution(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock);
/*
    prepare_for_convolution(d_IMAGE_mu, d_IMAGE_amp_mu, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock);
    prepare_for_convolution(d_IMAGE_sig, d_IMAGE_amp_sig, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock);
*/


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
    
    conv_twice_and_copy(d_IMAGE_amp, d_IMAGE_amp_ext, d_conv_amp, d_conv_conv_amp, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock);

/*
    ConvKernel<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_first_conv,  d_IMAGE_amp_ext, image_x+4, image_y+4);
    ConvKernel<double><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT_second_conv,  d_conv_amp, image_x+4, image_y+4);
    copy_gpu<double><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_amp, d_RESULTAT_first_conv, image_x, image_y);
    copy_gpu<double><<<BlocksParGrille, ThreadsParBlock>>>(d_conv_conv_amp, d_RESULTAT_second_conv, image_x, image_y);
*/



    checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));    

    cudaMemcpy(h_RESULTAT_GPU, d_conv_amp, size_j, cudaMemcpyDeviceToHost);

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

    cudaFree(d_IMAGE_amp);
    cudaFree(d_IMAGE_mu);
    cudaFree(d_IMAGE_sig);
    cudaFree(d_IMAGE_amp_ext);
    cudaFree(d_IMAGE_mu_ext);
    cudaFree(d_IMAGE_sig_ext);

    free(h_IMAGE_extended);
}

































































void conv2D_GPU_all(float *h_IMAGE, float* h_KERNEL, float* h_RESULTAT_GPU, long int image_x, long int image_y, int n_gauss,float temps_transfert, float temps_mirroirs)
{
    dummyInstantiator();
    float Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    int nb_cycles = 1;
    long int kernel_x = 3;
    long int kernel_y = 3;
    long int kernel_radius_x = 1;
    long int kernel_radius_y = 1;

    unsigned long int size_i = (image_x+4)  * (image_y+4)  * sizeof(float);
    unsigned long int size_j = (image_x)  * (image_y)  * sizeof(float);
    unsigned long int size_k = (kernel_x)  * (kernel_y)  * sizeof(float);

    float temps_temp = (float)(omp_get_wtime());
    float* h_IMAGE_extended = (float*)malloc((image_x+4)*(image_y+4)*sizeof(float));

    float* d_IMAGE_amp;
    cudaMalloc((void**)&d_IMAGE_amp, size_j);
    float* d_IMAGE_amp_ext;
    cudaMalloc((void**)&d_IMAGE_amp_ext, size_i);
    float* d_IMAGE_mu;
    cudaMalloc((void**)&d_IMAGE_mu, size_j);
    float* d_IMAGE_mu_ext;
    cudaMalloc((void**)&d_IMAGE_mu_ext, size_i);
    float* d_IMAGE_sig;
    cudaMalloc((void**)&d_IMAGE_sig, size_j);
    float* d_IMAGE_sig_ext;
    cudaMalloc((void**)&d_IMAGE_sig_ext, size_i);


    float* d_conv_amp;
    cudaMalloc((void**)&d_conv_amp, size_j);
    float* d_conv_mu;
    cudaMalloc((void**)&d_conv_mu, size_j);
    float* d_conv_sig;
    cudaMalloc((void**)&d_conv_sig, size_j);

    float* d_conv_conv_amp;
    cudaMalloc((void**)&d_conv_conv_amp, size_j);
    float* d_conv_conv_mu;
    cudaMalloc((void**)&d_conv_conv_mu, size_j);
    float* d_conv_conv_sig;
    cudaMalloc((void**)&d_conv_conv_sig, size_j);

    int threads = BLOCK_SIZE;
    int blocks_x  = ceil(double(image_x+4) / double(BLOCK_SIZE));
    int blocks_y  = ceil(double(image_y+4) / double(BLOCK_SIZE));
    int blocks_x_init = ceil(double(max(image_x+4, image_y)) / double(BLOCK_SIZE));
    int blocks_y_init = 2;//ceil(double(2) / double(BLOCK_SIZE));

    dim3 ThreadsParBlock(threads, threads);
    dim3 BlocksParGrille(blocks_x , blocks_y );

    dim3 ThreadsParBlock_init(threads, 2);
    dim3 BlocksParGrille_init(blocks_x_init , blocks_y_init);

    checkCudaErrors(cudaMemcpy(d_IMAGE_amp, h_IMAGE, size_j, cudaMemcpyHostToDevice));

    prepare_for_convolution(d_IMAGE_amp, d_IMAGE_amp_ext, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock);
/*
    prepare_for_convolution(d_IMAGE_mu, d_IMAGE_amp_mu, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock);
    prepare_for_convolution(d_IMAGE_sig, d_IMAGE_amp_sig, image_x, image_y, BlocksParGrille_init, ThreadsParBlock_init, BlocksParGrille, ThreadsParBlock);
*/


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

        float* d_RESULTAT;        
        cudaMalloc((void**)&d_RESULTAT, size_i);

        cudaMemset ( d_RESULTAT, 0 , size_i) ;
                
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
    
    for(int i = 0 ; i < (nb_cycles) ; i++)
        ConvKernel<float><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT,  d_IMAGE_amp_ext, image_x+4, image_y+4);

    checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));    

    copy_gpu<float><<<BlocksParGrille, ThreadsParBlock>>>(d_IMAGE_amp, d_RESULTAT, image_x, image_y);

    cudaMemcpy(h_RESULTAT_GPU, d_IMAGE_amp, size_j, cudaMemcpyDeviceToHost);
            
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

    cudaFree(d_IMAGE_amp);
    cudaFree(d_IMAGE_mu);
    cudaFree(d_IMAGE_sig);
    cudaFree(d_IMAGE_amp_ext);
    cudaFree(d_IMAGE_mu_ext);
    cudaFree(d_IMAGE_sig_ext);
    cudaFree(d_RESULTAT);

    free(h_IMAGE_extended);
}
