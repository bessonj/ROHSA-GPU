#include <stdio.h>

__global__
void convolve_global(float* Md, float* Rd, int width, int height){
    //(float* d_Data, float* d_Result, long int c_image_x, long int c_image_y)
    float Kdc[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};

    int kernel_size = 3;
    int tile_width = 1;
    int channels = 1;

    int row = blockIdx.y*tile_width + threadIdx.y;
    int col = blockIdx.x*tile_width + threadIdx.x;

    if(row < height  &&  col < width){
//printf("test !\n");
//printf("row = %d, col = %d \n", row, col);
        float sum = 0;
        int pixel;
        int local_pixel;
        int working_pixel;

        int row_offset = (kernel_size/2)*(width+kernel_size-1);
        int col_offset = kernel_size/2;

            pixel = row*width + col;
            local_pixel = row*(width+kernel_size-1) + col + row_offset + col_offset;
                for(int x=(-1)*kernel_size/2; x<=kernel_size/2; x++){
                    for(int y=(-1)*kernel_size/2; y<=kernel_size/2; y++){
                        working_pixel = local_pixel + x + y*(width+kernel_size-1);
                        sum += Md[working_pixel] * Kdc[x+kernel_size/2 + (y+kernel_size/2)*kernel_size];
//printf("sum = %f \n", sum);

                    }
                }
            Rd[pixel] = sum;
            sum = 0;
        }
//    printf("Rd[%d][%d] = %f \n", row, col, Rd[row + col*height]);
}

__global__ void ConvKernel(float* d_Result, float* d_Data, long int c_image_x, long int c_image_y)
	{
    float c_Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    long int c_kernel_x = 3;
    long int c_kernel_y = 3;
    long int c_kernel_radius_x = 1;
    long int c_kernel_radius_y = 1;

    float pixel_extrait = 0;
    float tmp_sum = 0;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_pos = pixel_y * c_image_y + pixel_x ;
if(pixel_y < c_image_y && pixel_x < c_image_x) {
	for (int y = - (c_kernel_radius_y) ; y <= c_kernel_radius_y; y++)
		{
		for (int x = - (c_kernel_radius_x) ; x <= c_kernel_radius_x; x++)
			{
            if ( ( (pixel_x + x) >= 0) && ( (pixel_y + y) >= 0) && ((pixel_x + x) <= c_image_x) && ((pixel_y + y) <= c_image_y) ){
			    pixel_extrait= d_Data[pixel_x+x + (pixel_y+y)*c_image_y ];
            }
            else{
                pixel_extrait = 0;
            }
//        	printf("pixel_extrait = %f \n", pixel_extrait);
			tmp_sum += pixel_extrait * c_Kernel[c_kernel_radius_x+x + (c_kernel_radius_y+y)*c_kernel_y];
			}
		}
	d_Result[pixel_pos] = tmp_sum;
//	printf("d_Result[%d] = %f \n", pixel_pos, d_Result[pixel_pos]);
    }
}
/*
__global__ void ConvKernel(float* d_Result, float* d_Data, long int c_image_x, long int c_image_y)
    {
    float c_Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    long int c_kernel_x = 3;
    long int c_kernel_y = 3;
    long int c_kernel_radius_x = 1;
    long int c_kernel_radius_y = 1;

    float tmp_sum = 0;
    float pixel_extrait = 0;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_pos = pixel_x * c_image_y + pixel_y ;

printf("--> pixel_extrait en 4,4 = %f \n", d_Data[4*c_image_x+4]);

    printf("-------------------INTÉRIEUR DU KERNEL CONVOLUTION-------------------\n");
if(pixel_y<c_image_y && pixel_x<c_image_x)
{
    for (int y = - (c_kernel_radius_y) ; y <= ((int)(c_kernel_radius_y)); y++)
        {
        for (int x = - (c_kernel_radius_x) ; x <= ((int)(c_kernel_radius_x)); x++)
            {
                if (( ( (pixel_x + x) < 0) || ( (pixel_y + y) < 0) || ((pixel_x + x - c_image_x) < blockDim.x ) || ( (pixel_y + y - c_image_y) < blockDim.y ))){
                pixel_extrait= d_Data[pixel_y+y + (pixel_x+x)*c_image_y];
//            pixel_extrait= ( ( (pixel_x + x)      < 0   ) || ( (pixel_y + y)< 0) || ( (pixel_x + x - c_image_x) < blockDim.x ) || ( (pixel_y + y - c_image_y) < blockDim.y )) ? 0 : d_Data[pixel_y+y + (pixel_x+x)*c_image_y ];
//            printf("--> tmp_sum = %f \n", tmp_sum);
            printf("--> pixel_extrait en %d = %f \n", pixel_y+y + (pixel_x+x)*c_image_y, pixel_extrait);
//            printf("--> c_Kernel[%d] = %f \n",c_kernel_radius_y-y + (c_kernel_radius_x-x)*c_kernel_y, c_Kernel[c_kernel_radius_y-y + (c_kernel_radius_x-x)*c_kernel_y]);
//            printf("d_Data[%d] = %f \n", pixel_y+y + (pixel_x+x)*c_image_y, d_Data[pixel_y+y + (pixel_x+x)*c_image_y]);
                }
                else{
                    pixel_extrait = 0.;
                }
            tmp_sum += pixel_extrait * c_Kernel[c_kernel_radius_y-y + (c_kernel_radius_x-x)*c_kernel_y];  
            }
        }

    d_Result[pixel_pos] = tmp_sum;
//    printf("tmp_sum = %f \n", tmp_sum);
}
    }
*/


__global__ void ConvKernel(double* d_Result, double* d_Data, long int c_image_x, long int c_image_y)
    {
    double c_Kernel[9] = {0,-1,0,-1,4,-1,0,-1,0};
    long int c_kernel_x = 3;
    long int c_kernel_y = 3;
    long int c_kernel_radius_x = 1;
    long int c_kernel_radius_y = 1;

    double tmp_sum = 0;
    double pixel_extrait = 0;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_pos = pixel_x * c_image_y + pixel_y ;
    //    printf("-------------------INTÉRIEUR DU KERNEL CONVOLUTION-------------------\n");
if(pixel_y<c_image_y && pixel_x<c_image_x)
{
    for (int y = - (c_kernel_radius_y) ; y <= ((int)(c_kernel_radius_y)); y++)
        {
        for (int x = - (c_kernel_radius_x) ; x <= ((int)(c_kernel_radius_x)); x++)
            {
                if (( ( (pixel_x + x) >= 0) || ( (pixel_y + y)>= 0) || ((pixel_x + x - c_image_x) >= blockDim.x ) || ( (pixel_y + y - c_image_y) >= blockDim.y ))){
                pixel_extrait= d_Data[pixel_y+y + (pixel_x+x)*c_image_y];
//            pixel_extrait= ( ( (pixel_x + x)      < 0   ) || ( (pixel_y + y)< 0) || ( (pixel_x + x - c_image_x) < blockDim.x ) || ( (pixel_y + y - c_image_y) < blockDim.y )) ? 0 : d_Data[pixel_y+y + (pixel_x+x)*c_image_y ];
//            printf("--> tmp_sum = %f \n", tmp_sum);
    //            printf("--> pixel_extrait en %d = %f \n", pixel_y+y + (pixel_x+x)*c_image_y, pixel_extrait);
//            printf("--> c_Kernel[%d] = %f \n",c_kernel_radius_y-y + (c_kernel_radius_x-x)*c_kernel_y, c_Kernel[c_kernel_radius_y-y + (c_kernel_radius_x-x)*c_kernel_y]);
//            printf("d_Data[%d] = %f \n", pixel_y+y + (pixel_x+x)*c_image_y, d_Data[pixel_y+y + (pixel_x+x)*c_image_y]);
                }
            tmp_sum += pixel_extrait * c_Kernel[c_kernel_radius_y-y + (c_kernel_radius_x-x)*c_kernel_y];  
            }
        }

    d_Result[pixel_pos] = tmp_sum;
//    printf("tmp_sum = %f \n", tmp_sum);
}
    }

__global__ void copy_gpu(double* d_out, double* d_in, long int length_x_in, long int length_y_in)
{
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixel_x < length_x_in && pixel_y < length_y_in){
        d_out[pixel_y + (length_y_in)*pixel_x] = d_in[pixel_y + (length_y_in+4)*pixel_x];
    }
/*
    for(int j(0); j<length_x_in; j++)
	{
		for(int i(0); i<length_y_in; i++)
		{
			printf("d_in[%d] = %f \n", i+(length_x_in)*(j), d_in[i+(length_x_in)*(j)]);
		}
	}
*/
}

__global__ void copy_gpu(float* d_out, float* d_in, long int length_x_in, long int length_y_in)
{
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixel_x < length_x_in && pixel_y < length_y_in){
        d_out[pixel_y + (length_y_in)*pixel_x] = d_in[(pixel_y+2) + (length_y_in+4)*(pixel_x+2)];
    }

}



/*
template <typename T>
__global__ void ConvKernel(T* d_Result, T* d_Data, long int c_image_x, long int c_image_y)
    {
    T c_Kernel[9] = {0,-1,0,-1,4,-1,0,-1,0};
    long int c_kernel_x = 3;
    long int c_kernel_y = 3;
    long int c_kernel_radius_x = 1;
    long int c_kernel_radius_y = 1;

    T tmp_sum = 0;
    T pixel_extrait = 0;
    int pixel_y = blockIdx.y * blockDim.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_pos = pixel_x * c_image_y + pixel_y ;

    for (int y = - (c_kernel_radius_y) ; y <= ((int)(c_kernel_radius_y)); y++)
        {
        for (int x = - (c_kernel_radius_x) ; x <= ((int)(c_kernel_radius_x)); x++)
            {
            pixel_extrait= ( ( (pixel_x + x)      < 0   )    ||
                             ( (pixel_y + y)      < 0    )     ||
                             ( (pixel_x + x - c_image_x) < blockDim.x ) ||
                             ( (pixel_y + y - c_image_y) < blockDim.y )        )  

                  ? 0 : d_Data[pixel_y+y + (pixel_x+x)*c_image_y ];

            tmp_sum += pixel_extrait * c_Kernel[c_kernel_radius_y-y + (c_kernel_radius_x-x)*c_kernel_y];  
            }
        }
    d_Result[pixel_pos] = tmp_sum;
    }
*/
//-------------------------------------------------------------------------------------------------------------------------------
//END OF FILE
