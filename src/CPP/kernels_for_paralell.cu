#include "f_g_cube_gpu.hpp"
#include <stdio.h>
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]

template <typename T>
__global__ void kernel_conv_g_reduction_sort(int n_beta, T* d_g, T* result_reduction_sig, T lambda_var_sig, int n_gauss, T* b_params_dev, int k, int image_x, int image_y)
{ 
    d_g[n_beta - n_gauss + k-1] = lambda_var_sig * (image_x*image_y*b_params_dev[k] - result_reduction_sig[0]);
}

template <typename T>
__global__ void kernel_update_deriv_conv_conv_sort(T* deriv, T lambda_amp, T lambda_mu, T lambda_sig, T lambda_var_sig, T* conv_conv_amp, T* conv_conv_mu, T* conv_conv_sig, T* image_sig, T* b_params_dev, int indice_y, int indice_x, int k)
{ 
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
//						ROHSA world			dev world 
		//deriv      --> (ng,y,x)    --> (z,y,x)
		//conv_conv_*    --> (y,x)       --> (y,x)
	if(index_x<indice_x && index_y<indice_y)
	{
        deriv[(3*k+0)*indice_x*indice_y+index_y*indice_x+ index_x] += lambda_amp*conv_conv_amp[indice_x*index_y + index_x];
        deriv[(3*k+1)*indice_x*indice_y+index_y*indice_x+ index_x] += lambda_mu*conv_conv_mu[indice_x*index_y + index_x];
        deriv[(3*k+2)*indice_x*indice_y+index_y*indice_x+ index_x] += lambda_sig*conv_conv_sig[indice_x*index_y + index_x]+ lambda_var_sig*(image_sig[indice_x*index_y + index_x]-b_params_dev[k]);
	}
}


template <typename T>
__global__ void gradient_kernel_2_beta_with_INDEXING_sort(T* deriv, int* t_d, T* params, int* t_p, T* residual, int* t_r, T* std_map, int* t_std, int n_gauss)
{ 
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
	int index_z = blockIdx.z*blockDim.z +threadIdx.z;

	int dim_g = t_p[0];
	int dim_y = t_p[1];
	int dim_x = t_p[2];
	int dim_v = t_r[0];

	int residual_SHAPE0 = t_r[0];
	int residual_SHAPE1 = t_r[1];
	int residual_SHAPE2 = t_r[2];
	int std_map_SHAPE0 = t_std[0];
	int std_map_SHAPE1 = t_std[1];
	int deriv_SHAPE0 = t_d[0];
	int deriv_SHAPE1 = t_d[1];
	int deriv_SHAPE2 = t_d[2];
	int params_SHAPE0 = t_p[0];
	int params_SHAPE1 = t_p[1];
	int params_SHAPE2 = t_p[2];

	//taille tableau residual
	int tr0=t_r[0]; // v --> i
	int tr1=t_r[1]; // y --> index_y
    int tr2=t_r[2]; // x --> index_x

	//taille tableau std_map
	int ts0=t_std[0]; // y --> index_y
	int ts1=t_std[1]; // x --> index_x

	//taille tableau deriv
	int td0 = t_d[0]; // 3*ng --> index_z
	int td1 = t_d[1]; // y --> index_y
	int td2 = t_d[2]; // x --> index_x

	//taille params_flat
	int tp0 = t_p[0]; // 3*ng --> index_z
	int tp1 = t_p[1]; // y --> index_y
	int tp2 = t_p[2]; // x --> index_x

//						ROHSA world			dev world 
        //params     --> (ng,y,x)    --> (z,y,x)
		//residual   --> (z,y,x)     --> (i,y,x)
		//deriv      --> (ng,y,x)    --> (z,y,x)
		//std_map    --> (y,x)       --> (y,x)

	if(index_z<n_gauss && index_x<dim_x && index_y<dim_y && std_map[index_y*dim_x+index_x]>0.)
	{
		T par0 = params[(3*index_z+0)*dim_y*dim_x+index_y*dim_x+index_x];

		T par1_a = params[(3*index_z+1)*dim_y*dim_x+index_y*dim_x+index_x];
		T par2 = params[(3*index_z+2)*dim_y*dim_x+index_y*dim_x+index_x];
		T par2_pow = 1/(2*pow(par2,2.));
		T parstd = 1/pow(std_map[index_y*dim_x+index_x],2);

		T buffer_0 = 0.;        //dF_over_dB --> (v,y,x,3g)  --> (i,x,y,z)
		T buffer_1 = 0.;
		T buffer_2 = 0.;

		for(int i=0; i<residual_SHAPE0; i++){
			T par_res = residual[i*dim_y*dim_x+index_y*dim_x+index_x]* parstd;
			__syncthreads();
			T par1 = T(i+1) - par1_a;
			deriv[(3*index_z+0)*dim_y*dim_x+ index_y*dim_x+ index_x]+= exp( -pow( par1 ,2.)*par2_pow ) * par_res;
			deriv[(3*index_z+1)*dim_y*dim_x+ index_y*dim_x+ index_x]+= par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ) * par_res;
			deriv[(3*index_z+2)*dim_y*dim_x+ index_y*dim_x+ index_x]+= par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * par_res; 
		}
	}
}


template <typename T>
__global__ void kernel_norm_map_boucle_v_sort(T* map_norm_dev, T* residual, T* std_map, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;

  if(index_x<indice_x && index_y<indice_y && std_map[index_y*indice_x + index_x]>0.)
  {
	  T sum_temp = 0.;
	  for(int index_z = 0; index_z<indice_v ; index_z++) {
    	 sum_temp += pow(residual[index_z*indice_y*indice_x+index_y*indice_x+index_x],2);
	  }

	__syncthreads();

	map_norm_dev[index_y*indice_x+index_x] = 0.5*sum_temp/pow(std_map[index_y*indice_x + index_x],2);
  }
}

template <typename T>
__global__ void kernel_norm_map_boucle_v_sort_no_reduction(thrust::device_ptr<T> &map_norm_dev, T* residual, T* std_map, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;

  if(index_x<indice_x && index_y<indice_y && std_map[index_y*indice_x + index_x]>0.)
  {
	  T sum_temp = 0.;
	  for(int index_z = 0; index_z<indice_v ; index_z++) {
    	 sum_temp += pow(residual[index_z*indice_y*indice_x+index_y*indice_x+index_x],2);
	  }

	__syncthreads();

	map_norm_dev[index_y*indice_x+index_x] = 0.5*sum_temp/pow(std_map[index_y*indice_x + index_x],2);
  }
}

template <typename T>
__global__ void kernel_norm_map_simple_sort(T lambda, T* map_norm_dev, T* map_dev, int indice_x, int indice_y)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
    if(index_x<indice_x && index_y<indice_y)
    {
  	    map_norm_dev[index_y*indice_x+index_x] = 0.5*lambda*pow(map_dev[index_y*indice_x+index_x],2);
    }
}

template <typename T>
__global__ void kernel_norm_map_simple_sort(T lambda, T lambda_var, T* map_norm_dev, T* map_conv_dev, T* map_image_dev, int indice_x, int indice_y, int k, T* b_params)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;

  if(index_x<indice_x && index_y<indice_y)
  {
  	map_norm_dev[index_y*indice_x+index_x] = 0.5*lambda*pow(map_conv_dev[index_y*indice_x+index_x],2) + 0.5*lambda_var*pow(map_image_dev[index_y*indice_x+index_x]-b_params[k],2);
  }
}

template <typename T>
__global__ void add_first_elements_sort(T* array_in, T* array_out){
	array_out[0] = array_out[0] + array_in[0];
}

template <typename T>
__global__ void display_dev_sort(T* array_out){
    printf("array_out[0] = %.16f\n", array_out[0]);
}

template <typename T>
__global__ void display_dev_complete_sort(T* array_out, int size){
    for(int i = 0; i<size; i++){
        printf("array_out[%d] = %.16f\n",i,array_out[i]);
    }    
}

template <typename T>
__global__ void display_dev_complete_fin_sort(T* array_out, int size, int rang){
    for(int i = size-rang; i<size; i++){
        printf("array_out[%d] = %.16f\n",i,array_out[i]);
    }    
}

template <typename T>
__global__ void cpy_first_num_dev_sort(T* array_in, T* array_out){
	array_out[0] = array_in[0];
}

template <typename T>
__global__ void init_dev_sort(T* array_in, T value){
	array_in[0] = value;
}

template <typename T>
__global__ void copy_dev_sort(T* array_in, T* array_out, int size){
	int tid = threadIdx.x +blockIdx.x * blockDim.x;
	if (tid < size){
		array_out[tid] = array_in[tid];
		__syncthreads();
	}
}

template <typename T>
__global__ void reduce_last_in_one_thread_sort(T* array_in, T* array_out, int size){
	T sum = 0;
	for(int i = 0; i<size; i++){
		sum += array_in[i];
	}
	array_out[0]=sum;
}

template <typename T>
__global__ void kernel_residual_simple_difference(T* cube, T* cube_reconstructed, T* residual, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
	int index_z = blockIdx.z*blockDim.z +threadIdx.z;

	if(index_x<indice_x && index_y<indice_y && index_z<indice_v)
	{

	residual[index_z*indice_y*indice_x+index_y*indice_x+index_x]=cube_reconstructed[index_z*indice_y*indice_x+index_y*indice_x+index_x]-cube[index_z*indice_y*indice_x+index_y*indice_x+index_x];

	}
}

template <typename T>
__global__ void kernel_hypercube_reconstructed(T* beta, T* cube_tilde, int indice_x, int indice_y, int indice_v, int n_gauss)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
	int index_z = blockIdx.z*blockDim.z +threadIdx.z;

	if(index_x<indice_x && index_y<indice_y && index_z<indice_v)
	{
	  	T model_gaussienne = 0.;
		for(int g = 0; g<n_gauss; g++)
		{
			T par[3];
			par[0]=beta[(3*g+0)*indice_y*indice_x+index_y*indice_x+index_x];
			par[1]=beta[(3*g+1)*indice_y*indice_x+index_y*indice_x+index_x];
			par[2]=beta[(3*g+2)*indice_y*indice_x+index_y*indice_x+index_x];					
			model_gaussienne += par[0]*exp(-pow((T(index_z+1)-par[1]),2.) / (2.*pow(par[2],2.)));
		}
		cube_tilde[index_z*indice_y*indice_x+index_y*indice_x+index_x]=model_gaussienne;
	}
}

template <typename T>
__global__ void sum_reduction_sort(T* a, T* c, int N)
{
	__shared__ T cache[BLOCK_SIZE_REDUCTION];
	int tid = threadIdx.x +blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	T temp = 0.;
	while (tid<N) {
		temp += a[tid];
		tid += blockDim.x*gridDim.x;
	}

	cache[cacheIndex] = temp;

	__syncthreads();

	int i = blockDim.x/2;
	while (i != 0) {
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i]; 
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
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
__global__ void init_extended_array_sort(T* d_IMAGE, T* d_IMAGE_extended, int image_x, int image_y){

    if(image_x == 0 || image_y == 0){
        return;
    }

	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;
	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (pixel_x < image_x && pixel_y<image_y){
		d_IMAGE_extended[image_x_ext*(pixel_y+2)+pixel_x+2]=d_IMAGE[image_x*pixel_y+pixel_x];
	}
}


template <typename T>
__global__ void extension_mirror_gpu_sort(T* h_IMAGE, T* h_IMAGE_extended, int image_x, int image_y){
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;

	if(pixel_x < image_x && pixel_y < 2){
		h_IMAGE_extended[image_x_ext*pixel_y+(2+pixel_x)] = h_IMAGE[image_x*pixel_y+pixel_x];
	}

	__syncthreads();

	if(pixel_x < image_y && pixel_y < 2){
		h_IMAGE_extended[(image_x_ext)*(pixel_x+2)+pixel_y] = h_IMAGE[image_x*pixel_x+pixel_y];
	}

	__syncthreads();

	if(pixel_x < image_x && pixel_y < 2){
		h_IMAGE_extended[(image_x_ext)*(pixel_y+image_y+2)+2+pixel_x] = h_IMAGE[image_x*(pixel_y+image_y-2)+pixel_x];
	}
	
		__syncthreads();

	if(pixel_x <image_y && pixel_y < 2){
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
__global__ void initialize_b_params(T* b_params_dev, T* beta_dev, int size, int n_beta, int n_gauss){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;

	if(pixel_x < size){
		b_params_dev[pixel_x]=beta_dev[n_beta-n_gauss+pixel_x-1];
	}
}

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
__global__ void extension_mirror_gpu_sort_save(T* h_IMAGE, T* h_IMAGE_extended, int image_x, int image_y){

    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;

	if(pixel_x < image_x && pixel_y < 2){
		h_IMAGE_extended[image_x_ext*pixel_y+(2+pixel_x)] = h_IMAGE[image_x*pixel_y+pixel_x];
	}
	__syncthreads();
	if(pixel_x < image_y && pixel_y < 2){
		h_IMAGE_extended[(image_x_ext)*(pixel_x+2)+pixel_y] = h_IMAGE[image_x*pixel_x+pixel_y];
	}
	__syncthreads();
	if(pixel_x < image_x && pixel_y < 2){
		h_IMAGE_extended[(image_x_ext)*(pixel_y+image_x+2)+2+pixel_x] = h_IMAGE[image_x*(pixel_y+image_x-2)+pixel_x];
	}
	__syncthreads();
	if(pixel_x <image_y && pixel_y < 2){
		h_IMAGE_extended[(image_x_ext)*(pixel_x+2)+2+image_y+pixel_y] = h_IMAGE[image_x*pixel_x+pixel_y+image_y-2];
	}
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

















