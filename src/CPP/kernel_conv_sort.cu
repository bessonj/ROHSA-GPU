#include <stdio.h>
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]


template<typename T>
__global__ void ConvKernel_sort(T* d_Result, T* d_Data, int c_image_x, int c_image_y)
	{
    T c_Kernel[9] = {0.,-0.25,0.,-0.25,1.,-0.25,0.,-0.25,0.};
    int c_kernel_radius_x = 1;
    int c_kernel_radius_y = 1;
	int c_kernel_x = 3;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	if(pixel_y < c_image_y && pixel_x < c_image_x) {
	    T tmp_sum = 0;
	    int pixel_pos = pixel_y * c_image_x + pixel_x ;
		for (int y = - (c_kernel_radius_y) ; y <= c_kernel_radius_y; y++)
			{
			for (int x = - (c_kernel_radius_x) ; x <= c_kernel_radius_x; x++)
				{
			    T pixel_extrait = 0.;
				if ( ( (pixel_x-c_kernel_radius_x) >= 0) && ((pixel_y-c_kernel_radius_y) >= 0) && ((pixel_x+c_kernel_radius_x) < c_image_x) && ((pixel_y+c_kernel_radius_x) < c_image_y) ){
				    pixel_extrait = d_Data[pixel_x+x + (pixel_y+y)*c_image_x];
				}
/*
				if ( ( (pixel_x+x) >= 0) && ((pixel_y+y) >= 0) && ((pixel_x+x) < c_image_x) && ((pixel_y+y) < c_image_y) ){
				    pixel_extrait = d_Data[pixel_x+x + (pixel_y+y)*c_image_x];
				}
*/
				tmp_sum += pixel_extrait * c_Kernel[c_kernel_radius_x+x + (c_kernel_radius_y+y)*c_kernel_x];
			}
		}
		d_Result[pixel_pos] = tmp_sum;
	}
}

template <typename T>
__global__ void copy_gpu_sort(T* d_out, T* d_in, int length_x_in, int length_y_in)
{
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(pixel_x < length_x_in && pixel_y < length_y_in){
        d_out[length_x_in*pixel_y + pixel_x] = d_in[(length_x_in+4)*(pixel_y+2) + (pixel_x+2)];
    }
}

template <typename T>
__global__ void fill_gpu_sort(T* d_out, T* d_in, int length_in)
{
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    if(pixel_x < length_in){
        d_out[pixel_x] = d_in[pixel_x];
    }
}


// beta = beta_modif !!!!!!!!!!
// beta is a float array
// id_par = 0,1,2
template <typename T>
__global__ void parameter_maps_sliced_from_beta_sort(T* beta_modif, T* d_IMAGE_amp, T* d_IMAGE_mu, T* d_IMAGE_sig, int image_x, int image_y, int k)
{
    if(image_x == 0 || image_y == 0){
        return;
    }
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel_x < image_x && pixel_y < image_y){
//	printf("index_x = %d , index_y = %d , k = %d\n",pixel_x, pixel_y, k);//, index_z = %d \n",index_x,index_y,index_z);
		d_IMAGE_amp[pixel_y*image_x+pixel_x] = beta_modif[ (0+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];		
		d_IMAGE_mu[pixel_y*image_x+pixel_x] = beta_modif[ (1+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];		
		d_IMAGE_sig[pixel_y*image_x+pixel_x] = beta_modif[ (2+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];
	}
}


template <typename T>
__global__ void print_device_array_sort(T* d_IMAGE, int image_x, int image_y){
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
__global__ void init_extended_array_sort(T* d_IMAGE, T* d_IMAGE_extended, int image_x, int image_y){

    if(image_x == 0 || image_y == 0){
        return;
    }

	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;
	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	
/*	if(image_x>64){
	    printf("pixel_x = %d, pixel_y = %d\n", pixel_x, pixel_y);
	}*/

/*	if (pixel_x < image_x_ext && pixel_y<image_y_ext){  
        d_IMAGE_extended[image_x_ext*pixel_y+pixel_x]=0.;
	}
	__syncthreads();
*/
	if (pixel_x < image_x && pixel_y<image_y){
		d_IMAGE_extended[image_x_ext*(pixel_y+2)+pixel_x+2]=d_IMAGE[image_x*pixel_y+pixel_x];
	}
}

//	int taille_image_conv[] = {indice_y, indice_x};
//trouver le bon dim3d
//nb_threads = max(dim_x_ext, dim_y)
//0 <= pixel_x < image_x_ext
//0 <= pixel_y < 2

template <typename T>
__global__ void extension_mirror_gpu_sort(T* h_IMAGE, T* h_IMAGE_extended, int image_x, int image_y){
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;

	if(pixel_x < image_x && pixel_y < 2){
//		printf("pixel_x = %d , pixel_y = %d \n",pixel_x,pixel_y);//,index_z);
		h_IMAGE_extended[image_x_ext*pixel_y+(2+pixel_x)] = h_IMAGE[image_x*pixel_y+pixel_x];
	}

	__syncthreads();

	if(pixel_x < image_y && pixel_y < 2){
//		printf("pixel_x = %d , pixel_y = %d \n",pixel_x,pixel_y);//,index_z);
		h_IMAGE_extended[(image_x_ext)*(pixel_x+2)+pixel_y] = h_IMAGE[image_x*pixel_x+pixel_y];
	}

	__syncthreads();

	if(pixel_x < image_x && pixel_y < 2){
//		printf("pixel_x = %d , pixel_y = %d \n",pixel_x,pixel_y);//,index_z);
		h_IMAGE_extended[(image_x_ext)*(pixel_y+image_y+2)+2+pixel_x] = h_IMAGE[image_x*(pixel_y+image_y-2)+pixel_x];
	}
	
		__syncthreads();

	if(pixel_x <image_y && pixel_y < 2){
//		printf("pixel_x = %d , pixel_y = %d \n",pixel_x,pixel_y);//,index_z);
		h_IMAGE_extended[(image_x_ext)*(pixel_x+2)+2+image_x+pixel_y] = h_IMAGE[image_x*pixel_x+pixel_y+image_x-2];
	}

/*
	if(pixel_x < 2 && pixel_y < image_y){
		h_IMAGE_extended[image_x_ext*(pixel_y+2)+(pixel_x)] = h_IMAGE[image_x*pixel_y+pixel_x];
	}
	if(pixel_x < 2 && pixel_y < image_y){
		h_IMAGE_extended[image_x_ext*(pixel_y)+(2+pixel_x)] = h_IMAGE[image_x*pixel_y+pixel_x];
	}
	if(pixel_x>=image_x && pixel_x < image_x+2 && pixel_y < image_y){
		h_IMAGE_extended[image_x_ext*(2+pixel_y)+(2+pixel_x)] = h_IMAGE[image_x*pixel_y+(pixel_x-2)];
	}
	if(pixel_x < image_x && pixel_y>=image_y && pixel_y < image_y+2){
		h_IMAGE_extended[image_x_ext*(2+pixel_y)+(2+pixel_x)] = h_IMAGE[image_x*(pixel_y-2)+pixel_x];
	}
*/

}

template <typename T>
__global__ void extension_mirror_gpu_sort_bis(T* h_IMAGE, T* h_IMAGE_extended, int image_x, int image_y){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;

	if(pixel_x < 2 && pixel_y < image_y){
		h_IMAGE_extended[image_x_ext*(pixel_y+2)+(pixel_x)] = h_IMAGE[image_x*pixel_y+pixel_x];
	}
	__syncthreads();
	if(pixel_x < image_x && pixel_y < 2){
		h_IMAGE_extended[image_x_ext*(pixel_y)+(2+pixel_x)] = h_IMAGE[image_x*pixel_y+pixel_x];
	}
	__syncthreads();
	if(pixel_x>=image_x && pixel_x < image_x+2 && pixel_y < image_y){
		h_IMAGE_extended[image_x_ext*(2+pixel_y)+(2+pixel_x)] = h_IMAGE[image_x*pixel_y+(pixel_x-2)];
	}
	__syncthreads();
	if(pixel_x < image_x && pixel_y>=image_y && pixel_y < image_y+2){
		h_IMAGE_extended[image_x_ext*(2+pixel_y)+(2+pixel_x)] = h_IMAGE[image_x*(pixel_y-2)+pixel_x];
	}
}

template <typename T>
__global__ void extension_mirror_gpu_sort_save(T* h_IMAGE, T* h_IMAGE_extended, int image_x, int image_y){

    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;

	if(pixel_x < image_x && pixel_y < 2){
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
	__syncthreads();
	if(pixel_x < image_y && pixel_y < 2){
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
	__syncthreads();
	if(pixel_x < image_x && pixel_y < 2){
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
	__syncthreads();
	if(pixel_x <image_y && pixel_y < 2){
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
void extension_mirror_sort(T* h_IMAGE, T* h_IMAGE_extended, int image_x, int image_y){
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


template <typename T>
__global__ void initialize_array(T* array_dev, int size, double value){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	if(pixel_x < size){
		array_dev[pixel_x] = value;
	}
}

template <typename T>
__global__ void initialize_array(T f, T value){
	f = 0.;
}

template <typename T>
__global__ void initialize_b_params(T* b_params_dev, T* beta_dev, int size, int n_beta, int n_gauss){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;

	if(pixel_x < size){
		b_params_dev[pixel_x]=beta_dev[n_beta-n_gauss+pixel_x];
	}
}

