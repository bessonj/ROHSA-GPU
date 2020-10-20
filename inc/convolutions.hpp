#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdio.h>      /* printf */
#include <math.h>       /* ceil */
#include <omp.h>
//-------------------------------------------------------------------------------------------------------------------------------
    const int BLOCK_SIZE     = 16 ;

//POUR LA VERSION SEPARABLE AVEC MEMOIRE SHARED
    #define NUMBER_COMPUTED_BLOCK 8
    #define NUMBER_EDGE_HALO_BLOCK 1

    #define BLOCK_SIZE_ROW_X 4
    #define BLOCK_SIZE_ROW_Y 16

    #define BLOCK_SIZE_COL_X 8
    #define BLOCK_SIZE_COL_Y 16

//---------------------------------------------------------------------------------------------------------------------------

    void conv2D_GPU(double* h_IMAGE, double* h_KERNEL, double* h_RESULTAT_GPU, long int image_x, long int image_y);
    void conv2D_GPU(float* h_IMAGE, float* h_KERNEL, float* h_RESULTAT_GPU, long int image_x, long int image_y, float temps_transfert, float temps_mirroirs);

/*
template <typename T>
void conv2D_GPU(T *h_IMAGE, T* h_KERNEL, T* h_RESULTAT_GPU, long int image_x, long int image_y)
{
    T Kernel[9] = {0,-1,0,-1,4,-1,0,-1,0};
    long int kernel_x = 3;
    long int kernel_y = 3;
    long int kernel_radius_x = 1;
    long int kernel_radius_y = 1;

    int nb_cycles = 1;

    T* h_IMAGE_extended = (T*)malloc((image_x+4)*(image_y+4)*sizeof(T));
    T* h_RESULTAT_GPU_extended = (T*)malloc((image_x+4)*(image_y+4)*sizeof(T));

	for(int j(0); j<image_x; j++)
	{
		for(int i(0); i<image_y; i++)
		{
			h_IMAGE_extended[2+i+(image_x+4)*(2+j)]=h_IMAGE[i+image_x*j];
		}
	}

	for(int j(0); j<2; j++)
	{
		for(int i(0); i<image_y; i++)
		{
			h_IMAGE_extended[2+i+(image_x+4)*j] = h_IMAGE[i+image_x*j];
		}
	}

	for(int i(0); i<2; i++)
	{
		for(int j(0); j<image_x; j++)
		{
			h_IMAGE_extended[i+(image_x+4)*(2+j)] = h_IMAGE[i+image_x*j];
		}
	}

	for(int j=image_x; j<image_x+2; j++)
	{
		for(int i=0; i<image_y; i++)
		{
			h_IMAGE_extended[2+i+(image_x+4)*(2+j)]=h_IMAGE[i+image_x*(j-2)];
		}
	}

	for(int j(0); j<image_x; j++)
	{
		for(int i(image_y); i<image_y+2; i++)
		{
			h_IMAGE_extended[2+i+(image_x+4)*(2+j)]=h_IMAGE[i-2+image_x*j];
		}
	}
    
//    T c_Kernel[9] = {0,-1,0,-1,4,-1,0,-1,0};
//    long int c_image_x = 32;
//    long int c_image_y = 32;
//    long int c_kernel_x = 3;
//    long int c_kernel_y = 3;
//    long int c_kernel_radius_x = 1;
//    long int c_kernel_radius_y = 1;

    //---------------------------------------------------------------------------------------------------------------------------
    printf("\n\n GPU-Based 2D Image convolution");
    
    //---------------------------------------------------------------------------------------------------------------------------
    printf("\n\t Allocating and intializing memory");

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t record_event[5];
    float time_msec[4];
    for (int i=0;i<5;i++){
        checkCudaErrors(cudaEventCreate(record_event+i));   
    }

    // Record the start event
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventRecord(record_event[0], NULL));

        unsigned long int size_i = image_x  * image_y  * sizeof(T);
        unsigned long int size_k = kernel_x * kernel_y * sizeof(T);
    
        T* d_IMAGE;            cudaMalloc((void**)&d_IMAGE,    size_i);
        T* d_RESULTAT;        cudaMalloc((void**)&d_RESULTAT, size_i);  
        
        
        
        cudaMemset ( d_RESULTAT, 0 , size_i) ;
        cudaMemset ( d_IMAGE, 0 , size_i) ;
                
     checkCudaErrors(cudaEventSynchronize(record_event[1]));    
     checkCudaErrors(cudaEventRecord(record_event[1], NULL));
    
    //---------------------------------------------------------------------------------------------------------------------------
    printf("\n\t Transfering HOST data to GPU");
    
    cudaMemcpy(d_IMAGE,    h_IMAGE_extended,   size_i, cudaMemcpyHostToDevice);
    
    //---------------------------------------------------------------------------------------------------------------------------        
    printf("\n\t Transfering CONSTANTS to GPU");
    
    checkCudaErrors(cudaEventRecord(record_event[2], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[2]));
    //---------------------------------------------------------------------------------------------------------------------------
    printf("\n\t Running GPU 2D-Convolution algorithm for %lu iterations\n", nb_cycles);
        
        int    threads = BLOCK_SIZE;
        int blocks_x  = (image_x) / BLOCK_SIZE;
        int blocks_y  = (image_y) / BLOCK_SIZE;

        dim3 ThreadsParBlock(threads, threads);
        dim3 BlocksParGrille(blocks_x , blocks_y );

            for(int i = 0 ; i < (nb_cycles) ; i++)
                ConvKernel<T><<<BlocksParGrille, ThreadsParBlock>>>(d_RESULTAT,  d_IMAGE, image_x+4, image_y+4);
                    
            checkCudaErrors(cudaEventRecord(record_event[3], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[3]));    
        
    //---------------------------------------------------------------------------------------------------------------------------
    printf("\n\t Transfering GPU Data back to HOST");
    
    
            cudaMemcpy(h_RESULTAT_GPU_extended, d_RESULTAT, size_i, cudaMemcpyDeviceToHost);
            
        checkCudaErrors(cudaEventRecord(record_event[4], NULL));
    checkCudaErrors(cudaEventSynchronize(record_event[4]));
        
    //---------------------------------------------------------------------------------------------------------------------------    
checkCudaErrors(cudaEventElapsedTime(time_msec+0, record_event[0], record_event[1]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+1, record_event[1], record_event[2]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+2, record_event[2], record_event[3]));
    checkCudaErrors(cudaEventElapsedTime(time_msec+3, record_event[3], record_event[4]));
    time_msec[4]=time_msec[0]+time_msec[1]+time_msec[2]+time_msec[3];

    printf("\n\n\t Memory transfer speed :");
            
        float total_mem_time         = time_msec[0]+ time_msec[1]+time_msec[3];
        float total_time            = time_msec[4] ;
        float upload_speed           = float(((size_i )*1e-9) / ( time_msec[1] *1e-3));
        float download_speed         = float(( size_i*1e-9) / (time_msec[3]*1e-3));
        float average_speed          = (upload_speed + download_speed) / 2 ;
        float percentage_vs_memory  = ( total_mem_time / total_time )*100;
    
        printf("\n\t\t Time taken by memory transfer = %.2f ms", total_mem_time);
        printf("\n\t\t Upload  : %.2f GB/s  Download : %.2f GB/s",upload_speed ,download_speed );
        printf("\n\t\t Average : %.2f GB/s", average_speed);
        printf("\n\t\t memory transfer part: %.1f pct", percentage_vs_memory);
        
        printf("\n\n\t Total GPU (calc+memory) mean execution time   : %.2f ms", total_time);
        printf("\n\t Total GPU (calc+memory) mean processing speed : %.1f Mpx/s\n", 1e-6 * image_x * image_x / (total_time * 1e-3) );


    //---------------------------------------------------------------------------------------------------------------------------
    printf("\n\n\t Freeing GPU Memory \n");
    
         cudaFree(d_IMAGE);
        cudaFree(d_RESULTAT);
	for(int j(0);j<image_x;j++)
	{
		for(int i(0); i<image_y; i++)
		{
			h_RESULTAT_GPU[i+image_x*j] = h_RESULTAT_GPU_extended[2+i+(image_x+4)*(2+j)];
		}
	}
        free(h_RESULTAT_GPU_extended);
        free(h_IMAGE_extended);
}
*/