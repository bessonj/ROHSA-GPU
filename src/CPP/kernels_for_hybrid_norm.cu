#include "gradient_norm.hpp"
#include <stdio.h>
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]


template <typename T>
__global__ void compute_nabla_R_wrt_m_norm(int n_beta, T* d_g, T* result_reduction_sig, T* result_reduction_sig_square,T lambda_var_sig, int n_gauss, T* b_params_dev, int k, int image_x, int image_y)
{ 
//	printf("n_beta - n_gauss + k = %d\n",n_beta - n_gauss + k);
//	printf("lambda_var_sig * (image_x*image_y*b_params_dev[k] - result_reduction_sig[0]) = %f\n",lambda_var_sig * (image_x*image_y*b_params_dev[k] - result_reduction_sig[0]));
    d_g[n_beta - n_gauss + k-1] = lambda_var_sig * (result_reduction_sig[0]*b_params_dev[k] - result_reduction_sig_square[0])/pow(b_params_dev[k],2.);
}

template <typename T>
__global__ void compute_square(T* image_in, T* image_out, int indice_x, int indice_y)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;

  if(index_x<indice_x && index_y<indice_y)
  {
  	image_out[index_y*indice_x+index_x] = pow(image_in[index_y*indice_x+index_x],2.);
  }
}

template <typename T>
__global__ void compute_nabla_R_wrt_theta_norm(T* deriv, T lambda_amp, T lambda_mu, T lambda_sig, T lambda_var_sig, T* conv_conv_amp, T* conv_conv_mu, T* conv_conv_sig, T* image_sig, T* b_params_dev, int indice_y, int indice_x, int k)
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
        deriv[(3*k+2)*indice_x*indice_y+index_y*indice_x+ index_x] += lambda_sig*conv_conv_sig[indice_x*index_y + index_x] + lambda_var_sig*(image_sig[indice_x*index_y + index_x]-b_params_dev[k])/b_params_dev[k];
	}
}

template <typename T>
__global__ void compute_R_map_norm(T lambda_amp, T lambda_mu, T lambda_sig, T lambda_var_sig, T* map_norm_dev, T* map_conv_sig_dev, T* map_conv_amp_dev, T* map_conv_mu_dev, T* map_image_sig_dev, int indice_x, int indice_y, int k, T* b_params)
{
	int index_x = blockIdx.x*blockDim.x +threadIdx.x;
	int index_y = blockIdx.y*blockDim.y +threadIdx.y;

  if(index_x<indice_x && index_y<indice_y)
  {
    map_norm_dev[index_y*indice_x+index_x] += 0.5*lambda_amp*pow(map_conv_amp_dev[index_y*indice_x+index_x],2);
    map_norm_dev[index_y*indice_x+index_x] += 0.5*lambda_mu*pow(map_conv_mu_dev[index_y*indice_x+index_x],2);
  	map_norm_dev[index_y*indice_x+index_x] += 0.5*lambda_sig*pow(map_conv_sig_dev[index_y*indice_x+index_x],2) + 0.5*lambda_var_sig*pow((map_image_sig_dev[index_y*indice_x+index_x]-b_params[k])/b_params[k],2);
  }
}












