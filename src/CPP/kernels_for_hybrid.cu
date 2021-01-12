#include "gradient.hpp"
#include <stdio.h>
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]


template <typename T> 
__global__ void gradient_kernel_2_beta_with_INDEXING(T* deriv, int* t_d, T* params, int* t_p, T* residual, int* t_r, T* std_map, int* t_std, int n_gauss)
{ 
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
	int index_z = blockIdx.z*blockDim.z +threadIdx.z;

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

	if(index_z<n_gauss && index_x<deriv_SHAPE2 && index_y<deriv_SHAPE1 && INDEXING_2D(std_map,index_y,index_x)>0.)
	{
		T par0 = INDEXING_3D(params,(3*index_z+0),index_y, index_x);

		T par1_a = INDEXING_3D(params, (3*index_z+1), index_y, index_x);
		T par2 = INDEXING_3D(params, (3*index_z+2), index_y, index_x);
		T par2_pow = 1/(2*pow(par2,2.));
		T parstd = 1/pow(INDEXING_2D(std_map, index_y, index_x),2);

		T buffer_0 = 0.;        
		T buffer_1 = 0.;
		T buffer_2 = 0.;

		for(int i=0; i<residual_SHAPE0; i++){
			T par_res = INDEXING_3D(residual,i,index_y,index_x)* parstd;
			T par1 = T(i+1) - par1_a;

			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * par_res;
			buffer_1 += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ) * par_res;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * par_res;
		}
//		__syncthreads();
		INDEXING_3D(deriv,(3*index_z+0), index_y, index_x)= buffer_0;
		INDEXING_3D(deriv,(3*index_z+1), index_y, index_x)= buffer_1;
		INDEXING_3D(deriv,(3*index_z+2), index_y, index_x)= buffer_2;
	}
}

template <typename T> 
__global__ void copy_dev(T* array_in, T* array_out, int size){
	int tid = threadIdx.x +blockIdx.x * blockDim.x;
	if (tid < size){
		array_out[tid] = array_in[tid];
	}
}

template <typename T> 
__global__ void sum_reduction(T* a, T* c, int N)
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
__global__ void cpy_first_num_dev(T* array_in, T* array_out){
	array_out[0] = array_in[0];
}

template <typename T> 
__global__ void kernel_residual(T* beta, T* cube, T* residual, int indice_x, int indice_y, int indice_v, int n_gauss)
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
		residual[index_z*indice_y*indice_x+index_y*indice_x+index_x]=model_gaussienne-cube[index_z*indice_y*indice_x+index_y*indice_x+index_x];
	}
}

template <typename T> 
__global__ void kernel_norm_map_boucle_v(T* map_norm_dev, T* residual, int* taille_residual, T* std_map, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;

    if(index_x<indice_x && index_y<indice_y)
    {
        if(std_map[index_y*indice_x + index_x]>0.){
            T sum = 0.;
            for(int index_z = 0; index_z<indice_v ; index_z++){
                sum += 0.5*pow(residual[index_z*indice_y*indice_x+index_y*indice_x+index_x],2);
            }
            map_norm_dev[index_y*indice_x+index_x] = sum*1/pow(std_map[index_y*indice_x + index_x],2);
        }
    }
}

template <typename T> 
__global__ void reduce_last_in_one_thread(T* array_in, T* array_out, int size){
	T sum = 0;
	for(int i = 0; i<size; i++){
		sum += array_in[i];
        printf("sum = %f\n",sum);
	}
	array_out[0]=sum;
    printf("array_out[0] = %f\n",array_out[0]);
}
