#include <stdio.h>
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]



template<typename T>
__global__ void ConvKernel(T* d_Result, T* d_Data, long int c_image_x, long int c_image_y)
	{
    T c_Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    long int c_kernel_x = 3;
    long int c_kernel_y = 3;
    long int c_kernel_radius_x = 1;
    long int c_kernel_radius_y = 1;

    T pixel_extrait = 0;
    T tmp_sum = 0;
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

template <typename T>
__global__ void copy_gpu(T* d_out, T* d_in, long int length_x_in, long int length_y_in)
{
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixel_x < length_x_in && pixel_y < length_y_in){
        d_out[pixel_y + (length_y_in)*pixel_x] = d_in[(pixel_y+2) + (length_y_in+4)*(pixel_x+2)];
    }
}


// beta = beta_modif !!!!!!!!!!
// beta is a float array
// id_par = 0,1,2
template <typename T>
__global__ void parameter_maps_sliced_from_beta(T* beta_modif, T* d_IMAGE_amp, T* d_IMAGE_mu, T* d_IMAGE_sig, int image_x, int image_y, int k)
{
    if(image_x == 0 || image_y == 0){
        return;
    }
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel_x < image_x && pixel_y < image_y){
		d_IMAGE_amp[pixel_y*image_x+pixel_x] = beta_modif[ (0+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];		
		d_IMAGE_mu[pixel_y*image_x+pixel_x] = beta_modif[ (1+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];		
		d_IMAGE_sig[pixel_y*image_x+pixel_x] = beta_modif[ (2+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];
	}

}
template <typename T>
__global__ void print_device_array(T* d_IMAGE, int image_x, int image_y){
	int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel_x < image_y && pixel_y < image_x){
	    printf("---> d_IMAGE[%d] = %f\n", pixel_x+image_y*pixel_y,d_IMAGE[pixel_x+image_y*pixel_y]);
	}
	__syncthreads();
}

//trouver le bon dim3d
// 0<=pixel<image+4
template <typename T>
__global__ void init_extended_array(T* d_IMAGE, T* d_IMAGE_extended, int image_x, int image_y){

    if(image_x == 0 || image_y == 0){
        return;
    }

	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;
	int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel_x < image_x && pixel_y<image_y){
		d_IMAGE_extended[2+pixel_y+image_x_ext*(2+pixel_x)]=d_IMAGE[pixel_y+image_x*pixel_x];
	}

	if (pixel_x < image_x_ext && pixel_y<image_y_ext){
		if (pixel_x >= image_x && pixel_y >= image_y){
			d_IMAGE_extended[pixel_y+image_y_ext*pixel_x]=0.;
		}
//    printf("d_IMAGE_extended[%d] = %f\n", pixel_y+image_y_ext*pixel_x,d_IMAGE_extended[pixel_y+image_y_ext*pixel_x]);
	}

}

//	int taille_image_conv[] = {indice_y, indice_x};
//trouver le bon dim3d
//nb_threads = max(dim_x_ext, dim_y)
//0 <= pixel_x < image_x_ext
//0 <= pixel_y < 2

template <typename T>
__global__ void extension_mirror_gpu(T* h_IMAGE, T* h_IMAGE_extended, int image_x, int image_y){
/*
    if(image_x == 0 || image_y == 0){
        return;
    }
*/
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;


	if(pixel_x < image_y && pixel_y < 2){
		h_IMAGE_extended[image_x_ext*pixel_y+(2+pixel_x)] = h_IMAGE[image_x*pixel_y+pixel_x];
	}
/*
	for(int j(0); j<2; j++)
	{
		for(int i(0); i<image_y; i++)
		{
			h_IMAGE_extended[(image_x+4)*j+2+i] = h_IMAGE[image_x*j+i];
		}
	}
*/
	if(pixel_x < image_x && pixel_y < 2){
		h_IMAGE_extended[(image_x_ext)*(pixel_x+2)+pixel_y] = h_IMAGE[image_x*pixel_x+pixel_y];
	}
/*
	for(int i(0); i<2; i++)
	{
		for(int j(0); j<image_x; j++)
		{
			h_IMAGE_extended[(image_x+4)*(2+j)+i] = h_IMAGE[image_x*j+i];
		}
	}
*/
	if(pixel_x < image_y && pixel_y < 2){
		h_IMAGE_extended[(image_x_ext)*(pixel_y+image_x+2)+2+pixel_x] = h_IMAGE[image_x*(pixel_y+image_x-2)+pixel_x];
	}
/*
	for(int j=image_x; j<image_x+2; j++)
	{
		for(int i=0; i<image_y; i++)
		{
			h_IMAGE_extended[(image_x+4)*(2+j)+2+i]=h_IMAGE[image_x*(j-2)+i];
		}
	}

*/
	if(pixel_x <image_x && pixel_y < 2){
		h_IMAGE_extended[(image_x_ext)*(pixel_x+2)+2+image_y+pixel_y] = h_IMAGE[image_x*pixel_x+pixel_y+image_y-2];
	}
/*
	for(int j(0); j<image_x; j++)
	{
		for(int i(image_y); i<image_y+2; i++)
		{
			h_IMAGE_extended[(image_x+4)*(2+j)+2+i]=h_IMAGE[image_x*j+i-2];
		}
	}
*/
}

template <typename T>
void extension_mirror(T* h_IMAGE, T* h_IMAGE_extended, int image_x, int image_y){
    if(image_x == 0 || image_y == 0){
        return;
    }
	for(int j(0); j<image_x+4; j++)
	{
		for(int i(0); i<image_y+4; i++)
		{
            h_IMAGE_extended[i+(image_y+4)*j]=0.;
        }
    }

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
}



