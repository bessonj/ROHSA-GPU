#include "gradient.hpp"
#include <stdio.h>
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]


template <typename T> 
__global__ void compute_nabla_Q(T* deriv, int* t_d, T* params, int* t_p, T* residual, int* t_r, T* std_map, int* t_std, int n_gauss)
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
/*
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
*/
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

/*
		T par2_pow = 1/(2*__powf(par2,2.));
		T parstd = 1/__powf(INDEXING_2D(std_map, index_y, index_x),2);
*/
		T buffer_0 = 0.;        
		T buffer_1 = 0.;
		T buffer_2 = 0.;

		for(int i=0; i<residual_SHAPE0; i++){
			T par_res = INDEXING_3D(residual,i,index_y,index_x)* parstd;
			T par_fin = INDEXING_3D(residual,i,index_y,index_x)/pow(INDEXING_2D(std_map, index_y, index_x),2);
//			T par_fin = INDEXING_3D(residual,i,index_y,index_x)/__powf(INDEXING_2D(std_map, index_y, index_x),2);
			T par1 = T(i+1) - par1_a;

/*
			buffer_0 += __expf( -__powf( par1 ,2.)*par2_pow ) * par_fin;
			buffer_1 += par0*(par1/__powf(par2,2.)) * __expf(-__powf( par1,2.)*par2_pow ) * par_fin;
			buffer_2 += par0*__powf( par1, 2.)/(__powf(par2,3.))*__expf(-__powf(par1 ,2.)*par2_pow ) * par_fin;
*/

			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * par_fin;
			buffer_1 += par0*(par1/pow(par2,2.)) * exp(-pow( par1,2.)*par2_pow ) * par_fin;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * par_fin;
		}
		INDEXING_3D(deriv,(3*index_z+0), index_y, index_x)= buffer_0;
		INDEXING_3D(deriv,(3*index_z+1), index_y, index_x)= buffer_1;
		INDEXING_3D(deriv,(3*index_z+2), index_y, index_x)= buffer_2;
	}
}

template <typename T> 
__global__ void gradient_kernel_2_beta_with_INDEXING_over_v(T* deriv, int* t_d, T* params, int* t_p, T* residual, int* t_r, T* std_map, int* t_std, int n_gauss)
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
/*
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
*/
//						ROHSA world			dev world 
        //params     --> (ng,y,x)    --> (z,y,x)
		//residual   --> (z,y,x)     --> (i,y,x)
		//deriv      --> (ng,ygradient_kernel_2_beta_with_INDEXING,x)    --> (z,y,x)
		//std_map    --> (y,x)       --> (y,x)

	if(index_z<residual_SHAPE0 && index_x<deriv_SHAPE2 && index_y<deriv_SHAPE1 && INDEXING_2D(std_map,index_y,index_x)>0.)
	{
		for(int i=0; i<n_gauss; i++){
			T par0 = INDEXING_3D(params,(3*i+0),index_y, index_x);

			T par1_a = INDEXING_3D(params, (3*i+1), index_y, index_x);
			T par2 = INDEXING_3D(params, (3*i+2), index_y, index_x);

			T par2_pow = 1/(2*pow(par2,2.));
			T parstd = 1/pow(INDEXING_2D(std_map, index_y, index_x),2);

	/*
			T par2_pow = 1/(2*__powf(par2,2.));
			T parstd = 1/__powf(INDEXING_2D(std_map, index_y, index_x),2);
	*/


			T par_res = INDEXING_3D(residual,index_z,index_y,index_x)* parstd;
			T par_fin = INDEXING_3D(residual,index_z,index_y,index_x)/pow(INDEXING_2D(std_map, index_y, index_x),2);
//			T par_fin = INDEXING_3D(residual,i,index_y,index_x)/__powf(INDEXING_2D(std_map, index_y, index_x),2);
			T par1 = T(i+1) - par1_a;

/*
			buffer_0 += __expf( -__powf( par1 ,2.)*par2_pow ) * par_fin;
			buffer_1 += par0*(par1/__powf(par2,2.)) * __expf(-__powf( par1,2.)*par2_pow ) * par_fin;
			buffer_2 += par0*__powf( par1, 2.)/(__powf(par2,3.))*__expf(-__powf(par1 ,2.)*par2_pow ) * par_fin;
*/

			INDEXING_3D(deriv,(3*i+0), index_y, index_x)+= exp( -pow( par1 ,2.)*par2_pow ) * par_fin;
			INDEXING_3D(deriv,(3*i+1), index_y, index_x)+= par0*(par1/pow(par2,2.)) * exp(-pow( par1,2.)*par2_pow ) * par_fin;
			INDEXING_3D(deriv,(3*i+2), index_y, index_x)+= par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * par_fin;
		}
	}
}

//Question is 3*n_gauss > dim_v
template <typename T> 
__global__ void gradient_kernel_2_beta_with_INDEXING_3D_noise(T* deriv, int* t_d, T* params, int* t_p, T* residual, int* t_r, T* std_map_reduced, int* t_std_reduced, T* std_map_3D, int* t_std_3D, int n_gauss)
{ 
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
	int index_z = blockIdx.z*blockDim.z +threadIdx.z;

	int residual_SHAPE0 = t_r[0];
	int residual_SHAPE1 = t_r[1];
	int residual_SHAPE2 = t_r[2];
	int std_map_reduced_SHAPE0 = t_std_reduced[0];
	int std_map_reduced_SHAPE1 = t_std_reduced[1];
	int std_map_3D_SHAPE0 = t_std_3D[0];
	int std_map_3D_SHAPE1 = t_std_3D[1];
	int std_map_3D_SHAPE2 = t_std_3D[2];
	int deriv_SHAPE0 = t_d[0];
	int deriv_SHAPE1 = t_d[1];
	int deriv_SHAPE2 = t_d[2];
	int params_SHAPE0 = t_p[0];
	int params_SHAPE1 = t_p[1];
	int params_SHAPE2 = t_p[2];
/*
	//taille tableau residual
	int tr0=t_r[0]; // v --> i
	int tr1=t_r[1]; // y --> index_y
    int tr2=t_r[2]; // x --> index_x

	//taille tableau std_map
	int ts0=t_std_3D[0]; // v --> index_v (<=> i)
	int ts1=t_std_3D[1]; // y --> index_y
	int ts2=t_std_3D[2]; // x --> index_x

	//taille tableau deriv
	int td0 = t_d[0]; // 3*ng --> index_z
	int td1 = t_d[1]; // y --> index_y
	int td2 = t_d[2]; // x --> index_x

	//taille params_flat
	int tp0 = t_p[0]; // 3*ng --> index_z
	int tp1 = t_p[1]; // y --> index_y
	int tp2 = t_p[2]; // x --> index_x
*/
//						ROHSA world			dev world 
        //params     --> (ng,y,x)    --> (z,y,x)
		//residual   --> (z,y,x)     --> (i,y,x)
		//deriv      --> (ng,y,x)    --> (z,y,x)
		//std_map    --> (y,x)       --> (y,x)

	if(index_z<n_gauss && index_x<deriv_SHAPE2 && index_y<deriv_SHAPE1 && INDEXING_2D(std_map_reduced,index_y,index_x)>0.)
	{
		T par0 = INDEXING_3D(params,(3*index_z+0),index_y, index_x);

		T par1_a = INDEXING_3D(params, (3*index_z+1), index_y, index_x);
		T par2 = INDEXING_3D(params, (3*index_z+2), index_y, index_x);
		T par2_pow = 1/(2*pow(par2,2.));

		T buffer_0 = 0.;        
		T buffer_1 = 0.;
		T buffer_2 = 0.;

		for(int i=0; i<residual_SHAPE0; i++){
			T par_fin = INDEXING_3D(residual,i,index_y,index_x)/pow(INDEXING_3D(std_map_3D, i, index_y, index_x),2);
			T par1 = T(i+1) - par1_a;

			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * par_fin;
			buffer_1 += par0*(par1/pow(par2,2.)) * exp(-pow( par1,2.)*par2_pow ) * par_fin;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * par_fin;
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
__global__ void compute_residual(T* beta, T* residual, int indice_x, int indice_y, int indice_v, int n_gauss)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
	int index_z = blockIdx.z*blockDim.z +threadIdx.z;

	if(index_x<indice_x && index_y<indice_y && index_z<indice_v)
	{
	  	T model_gaussienne = 0.;
		for(int g = 0; g<n_gauss; g++)
		{
/*
			T par_0=beta[(3*g+0)*indice_y*indice_x+index_y*indice_x+index_x];
			T par_1=beta[(3*g+1)*indice_y*indice_x+index_y*indice_x+index_x];
			T par_2=beta[(3*g+2)*indice_y*indice_x+index_y*indice_x+index_x];					
			model_gaussienne += par_0*exp(-pow((T(index_z+1)-par_1),2.) / (2.*pow(par_2,2.)));
*/
/*
			T par[3];
			par[0]=beta[(3*g+0)*indice_y*indice_x+index_y*indice_x+index_x];
			par[1]=beta[(3*g+1)*indice_y*indice_x+index_y*indice_x+index_x];
			par[2]=beta[(3*g+2)*indice_y*indice_x+index_y*indice_x+index_x];					
			model_gaussienne += par[0]*__expf(-__powf((T(index_z+1)-par[1]),2.) / (2.*__powf(par[2],2.)));
*/
			T par[3];
			par[0]=beta[(3*g+0)*indice_y*indice_x+index_y*indice_x+index_x];
			par[1]=beta[(3*g+1)*indice_y*indice_x+index_y*indice_x+index_x];
			par[2]=beta[(3*g+2)*indice_y*indice_x+index_y*indice_x+index_x];					
			model_gaussienne += par[0]*exp(-pow((T(index_z+1)-par[1]),2.) / (2.*pow(par[2],2.)));

		}
		residual[index_z*indice_y*indice_x+index_y*indice_x+index_x]=model_gaussienne-residual[index_z*indice_y*indice_x+index_y*indice_x+index_x];
	}
}

template <typename T> 
__global__ void kernel_residual_less_memory_shared(T* beta, T* residual, int indice_x, int indice_y, int indice_v, int n_gauss)
{
	__shared__ T model[BLOCK_SIZE_X_BIS*BLOCK_SIZE_Y_BIS*BLOCK_SIZE_Z_BIS];

	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
	int index_z = blockIdx.z*blockDim.z +threadIdx.z;

	if(index_x<indice_x && index_y<indice_y && index_z<indice_v)
	{
		model[BLOCK_SIZE_Y_BIS*BLOCK_SIZE_X_BIS*threadIdx.z+BLOCK_SIZE_X_BIS*threadIdx.y+threadIdx.x] = 0.;
		for(int g = 0; g<n_gauss; g++)
		{
			T par[3];
			par[0]=beta[(3*g+0)*indice_y*indice_x+index_y*indice_x+index_x];
			par[1]=beta[(3*g+1)*indice_y*indice_x+index_y*indice_x+index_x];
			par[2]=beta[(3*g+2)*indice_y*indice_x+index_y*indice_x+index_x];
			model[BLOCK_SIZE_Y_BIS*BLOCK_SIZE_X_BIS*threadIdx.z+BLOCK_SIZE_X_BIS*threadIdx.y+threadIdx.x] += par[0]*exp(-pow((T(index_z+1)-par[1]),2.) / (2.*pow(par[2],2.)));
		}
		residual[index_z*indice_y*indice_x+index_y*indice_x+index_x]=model[BLOCK_SIZE_Y_BIS*BLOCK_SIZE_X_BIS*threadIdx.z+BLOCK_SIZE_X_BIS*threadIdx.y+threadIdx.x]-residual[index_z*indice_y*indice_x+index_y*indice_x+index_x];
	}
}

template <typename T> 
__global__ void compute_Q_map(T* map_norm_dev, T* residual, T* std_map, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;

    if(index_x<indice_x && index_y<indice_y)
    {
        if(std_map[index_y*indice_x + index_x]>0.){
            T sum = 0.;
            for(int index_z = 0; index_z<indice_v ; index_z++){
//                sum += 0.5*pow(residual[index_z*indice_y*indice_x+index_y*indice_x+index_x],2);
                sum += 0.5*(residual[index_z*indice_y*indice_x+index_y*indice_x+index_x]*residual[index_z*indice_y*indice_x+index_y*indice_x+index_x]);
            }
//            map_norm_dev[index_y*indice_x+index_x] = sum*1/pow(std_map[index_y*indice_x + index_x],2);
            map_norm_dev[index_y*indice_x+index_x] = sum*1/(std_map[index_y*indice_x + index_x]*std_map[index_y*indice_x + index_x]);
        }
    }
}

// Question : Is dim_v<BLOCK_SIZE
template <typename T> 
__global__ void kernel_for_3D_noise(T* noise_map_reduced, T* noise_cube, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;
    if(index_x<indice_x && index_y<indice_y)
    {
        T sum = 0.;
		for(int index_z = 0; index_z<indice_v ; index_z++){
            sum += noise_cube[index_z*indice_y*indice_x+index_y*indice_x+index_x];
        }
        noise_map_reduced[index_y*indice_x+index_x] = sum;
    }
}

template <typename T> 
__global__ void reduce_last_in_one_thread(T* array_in, T* array_out, int size){
	T sum = 0;
	for(int i = 0; i<size; i++){
		sum += array_in[i];
	}
	array_out[0]=sum;
}

template <typename T> 
__global__ void print_diff(T* array_in_1, T* array_in_2){
	printf("difference = %.25f\n",abs((array_in_1[0] - array_in_2[0])/array_in_2[0]));
}

template <typename T> 
__global__ void display_size(T* array_out, int size){
	for(int i = 0; i<size; i++){
	    printf("array_out[%d] = %f\n",i, array_out[i]);
	}
}

template <typename T>
__global__ void get_gaussian_parameter_maps(T* beta_modif, T* d_IMAGE_amp, T* d_IMAGE_mu, T* d_IMAGE_sig, int image_x, int image_y, int k){
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (pixel_x < image_x && pixel_y < image_y){
		d_IMAGE_amp[pixel_y*image_x+pixel_x] = beta_modif[ (0+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];		
		d_IMAGE_mu[pixel_y*image_x+pixel_x] = beta_modif[ (1+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];		
		d_IMAGE_sig[pixel_y*image_x+pixel_x] = beta_modif[ (2+3*k)*image_x*image_y+ pixel_y*image_x+pixel_x];
	}
}

template <typename T>
__global__ void init_extended_array_sort(T* d_IMAGE, T* d_IMAGE_extended, int image_x, int image_y){

	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;
	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
	int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (pixel_x < image_x && pixel_y<image_y){
		d_IMAGE_extended[image_x_ext*(pixel_y+2)+pixel_x+2]=d_IMAGE[image_x*pixel_y+pixel_x];
	}
}

template <typename T>
__global__ void perform_mirror_effect_before_convolution(T* d_IMAGE, T* d_IMAGE_extended, int image_x, int image_y){
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
	int image_x_ext = image_x+4;
	int image_y_ext = image_y+4;

	if (pixel_x < image_x && pixel_y<image_y){
		d_IMAGE_extended[image_x_ext*(pixel_y+2)+pixel_x+2]=d_IMAGE[image_x*pixel_y+pixel_x];
	}
	__syncthreads();
	if(pixel_x < 2 && pixel_y < image_y){
		d_IMAGE_extended[image_x_ext*(pixel_y+2)+(pixel_x)] = d_IMAGE[image_x*pixel_y+pixel_x];
	}
	__syncthreads();
	if(pixel_x < image_x && pixel_y < 2){
		d_IMAGE_extended[image_x_ext*(pixel_y)+(2+pixel_x)] = d_IMAGE[image_x*pixel_y+pixel_x];
	}
	__syncthreads();
	if(pixel_x>=image_x && pixel_x < image_x+2 && pixel_y < image_y){
		d_IMAGE_extended[image_x_ext*(2+pixel_y)+(2+pixel_x)] = d_IMAGE[image_x*pixel_y+(pixel_x-2)];
	}
	__syncthreads();
	if(pixel_x < image_x && pixel_y>=image_y && pixel_y < image_y+2){
		d_IMAGE_extended[image_x_ext*(2+pixel_y)+(2+pixel_x)] = d_IMAGE[image_x*(pixel_y-2)+pixel_x];
	}
}

template<typename T>
__global__ void ConvKernel(T* d_Result, T* d_Data, int c_image_x, int c_image_y)
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
__global__ void copy_gpu(T* d_out, T* d_in, int length_x_in, int length_y_in)
{
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    if(pixel_x < length_x_in && pixel_y < length_y_in){
        d_out[length_x_in*pixel_y + pixel_x] = d_in[(length_x_in+4)*(pixel_y+2) + (pixel_x+2)];
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
__global__ void compute_nabla_R_wrt_m(int n_beta, T* d_g, T* result_reduction_sig, T lambda_var_sig, int n_gauss, T* b_params_dev, int k, int image_x, int image_y)
{ 
//	printf("n_beta - n_gauss + k = %d\n",n_beta - n_gauss + k);
//	printf("lambda_var_sig * (image_x*image_y*b_params_dev[k] - result_reduction_sig[0]) = %f\n",lambda_var_sig * (image_x*image_y*b_params_dev[k] - result_reduction_sig[0]));
    d_g[n_beta - n_gauss + k-1] = lambda_var_sig * (image_x*image_y*b_params_dev[k] - result_reduction_sig[0]);
}


template <typename T>
__global__ void compute_nabla_R_wrt_theta(T* deriv, T lambda_amp, T lambda_mu, T lambda_sig, T lambda_var_sig, T* conv_conv_amp, T* conv_conv_mu, T* conv_conv_sig, T* image_sig, T* b_params_dev, int indice_y, int indice_x, int k)
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
__global__ void compute_R_map(T lambda_amp, T lambda_mu, T lambda_sig, T lambda_var_sig, T* map_norm_dev, T* map_conv_sig_dev, T* map_conv_amp_dev, T* map_conv_mu_dev, T* map_image_sig_dev, int indice_x, int indice_y, int k, T* b_params)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;

  if(index_x<indice_x && index_y<indice_y)
  {
    map_norm_dev[index_y*indice_x+index_x] += 0.5*lambda_amp*pow(map_conv_amp_dev[index_y*indice_x+index_x],2);
    map_norm_dev[index_y*indice_x+index_x] += 0.5*lambda_mu*pow(map_conv_mu_dev[index_y*indice_x+index_x],2);
  	map_norm_dev[index_y*indice_x+index_x] += 0.5*lambda_sig*pow(map_conv_sig_dev[index_y*indice_x+index_x],2) + 0.5*lambda_var_sig*pow(map_image_sig_dev[index_y*indice_x+index_x]-b_params[k],2);
  }
}












