#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "gradient.hpp"
#include "model.hpp"

__device__ int flattened_index_4d(int* coordinates, int* taille_tab_4d)
{

int index = coordinates[3];
index+= taille_tab_4d[2]*coordinates[2];
index+= taille_tab_4d[1]*taille_tab_4d[2]*coordinates[1];
index+= taille_tab_4d[0]*taille_tab_4d[1]*taille_tab_4d[2]*coordinates[0];
/*
index+= taille_tab_4d[3]*coordinates[2];
index+= taille_tab_4d[3]*taille_tab_4d[2]*coordinates[1];
index+= taille_tab_4d[3]*taille_tab_4d[2]*taille_tab_4d[1]*coordinates[0];
*/	
return index;
}

__device__ void set_at_4d_index(double* flattened_4d_tab, int* coordinates, int* taille_tab_4d, double value)
{
	flattened_4d_tab[flattened_index_4d(coordinates, taille_tab_4d)] = value;
}

__device__ double value_at_4d_index(double* flattened_4d_tab, int* coordinates, int* taille_tab_4d)
{
	return flattened_4d_tab[flattened_index_4d(coordinates, taille_tab_4d)];
}


//on peut utiliser le fait que taille_dF_over_dB[2] et taille_dF_over_dB[3] sont des multiples de 2
__global__ void 
gradient_kernel_0(double* dF_over_dB, int* taille_dF_over_dB, double* params, int* taille_params, int n_gauss)
{ 
//int offset = threadIdx.x + blockIdx.x*(BLOCK_SIZE_X*N);
int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;

if(index_x<taille_dF_over_dB[3] && index_y<taille_dF_over_dB[2] && index_z<taille_dF_over_dB[1])
{

int coordinates[] = {0,index_z,index_y,index_x};
int coordinates_params_0[] = {0,index_y,index_x};
int coordinates_params_2[] = {0,index_y,index_x};
int coordinates_params_1[] = {0,index_y,index_x};

for(int i=0; i<n_gauss; i++){

     coordinates[0]=0+3*i;
     coordinates_params_0[0] = 0+3*i;
     coordinates_params_1[0] = 1+3*i;
     coordinates_params_2[0] = 2+3*i;

     set_at_4d_index(dF_over_dB, coordinates, taille_dF_over_dB, exp(-pow( double(index_z+1)-value_at_4d_index(params, coordinates_params_1, taille_params),2.)/(2*pow(value_at_4d_index(params, coordinates_params_2, taille_params),2.)) ) );

     coordinates[0]= 1+3*i;
     set_at_4d_index(dF_over_dB, coordinates, taille_dF_over_dB, value_at_4d_index(params, coordinates_params_0, taille_params)*( double(index_z+1) - value_at_4d_index(params, coordinates_params_1, taille_params))/pow(value_at_4d_index(params, coordinates_params_2, taille_params),2.) * 
                                     exp(-pow( double(index_z+1)-value_at_4d_index(params, coordinates_params_1, taille_params),2.)/(2*pow(value_at_4d_index(params, coordinates_params_2, taille_params),2.))));

     coordinates[0]=2+3*i;
     set_at_4d_index(dF_over_dB, coordinates, taille_dF_over_dB, value_at_4d_index(params, coordinates_params_0, taille_params)*pow( double(index_z+1) - value_at_4d_index(params, coordinates_params_1, taille_params), 2.)/(pow(value_at_4d_index(params, coordinates_params_2, taille_params),3.)) *
                                     exp(-pow( double(index_z+1)-value_at_4d_index(params, coordinates_params_1, taille_params),2.)/(2*pow(value_at_4d_index(params, coordinates_params_2, taille_params),2.)) ) );


}
}
     __syncthreads();
}

//taille_dF_over_dB = t
//taille_params = t_p
__global__ void 
gradient_kernel_1(double* dF_over_dB, int* t, double* params, int* t_p, int n_gauss)
{ 
//int offset = threadIdx.x + blockIdx.x*(BLOCK_SIZE_X*N);
int index = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;

if(index<t[3]*t[2])
{

for(int i=0; i<n_gauss; i++){
        for(int k=0; k<t[1]; k++){
		dF_over_dB[index + k*t[2]*t[1] + (3*i+0)*t[2]*t[1]*t[0]] = 8;//	exp(-pow( double(k+1)-params[index+(1+3*i)*t_p[1]*t_p[0]],2.)/(2*pow(params[index+(2+3*i)*t_p[1]*t_p[0]],2.)) );
		dF_over_dB[index + k*t[2]*t[1] + (3*i+1)*t[2]*t[1]*t[0]] = 8;
		dF_over_dB[index + k*t[2]*t[1] + (3*i+2)*t[2]*t[1]*t[0]] = 8;
	}
}

}
     __syncthreads();

}

/*
for(int i=0; i<M.n_gauss; i++){
        for(k=0; k<indice_v; k++){
                for(int j=0; j<indice_y; j++){
                        for(int l=0; l<indice_x; l++){
                                coordinates[0]=0+3*i;
                                coordinates[1]=k;
                                coordinates[2]=j;
                                coordinates[3]=l;

                                dF_over_dB.vec_add(coordinates, exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) ) );

                                coordinates[0]= 1+3*i;

                                dF_over_dB.vec_add(coordinates, params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * 
                                                                        exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) ) );
                                coordinates[0]=2+3*i;

                                dF_over_dB.vec_add(coordinates, params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) *
                                                                        exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) ) );

                        }
                }
        }
}


*/





void gradient(double* dF_over_dB, int* taille_dF_over_dB, int product_taille_dF_over_dB, double* params, int* taille_params, int product_taille_params, model &M)
{
   printf("test début gradient %d\n", dF_over_dB[0]);

   double* dF_over_dB_dev;
   double* params_dev;
   int* taille_dF_over_dB_dev;
   int* taille_params_dev;

   checkCudaErrors(cudaMalloc((void**)&dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double)));
   checkCudaErrors(cudaMalloc((void**)&params_dev, product_taille_params*sizeof(double)));
   checkCudaErrors(cudaMalloc((void**)&taille_dF_over_dB_dev, 4*sizeof(int)));
   checkCudaErrors(cudaMalloc((void**)&taille_params_dev, 4*sizeof(int)));

   checkCudaErrors(cudaMemcpy(dF_over_dB_dev, dF_over_dB, product_taille_dF_over_dB*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_dF_over_dB_dev, taille_dF_over_dB, 4*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 4*sizeof(int), cudaMemcpyHostToDevice));

//   cudaMalloc((void**)&dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double));

//   cudaMemcpy(dF_over_dB_dev, dF_over_dB, product_taille_dF_over_dB*sizeof(double), cudaMemcpyHostToDevice);


   dim3 Db, Dg;

/*
     int N = taille_dF_over_dB[3]*taille_dF_over_dB[2]*taille_dF_over_dB[1]*taille_dF_over_dB[0];

      Db.x = BLOCK_SIZE_X;
      Db.y = BLOCK_SIZE_Y;
      Db.z = BLOCK_SIZE_Z;

      if (N%BLOCK_SIZE_X == 0 && N%BLOCK_SIZE_Y == 0 && N%BLOCK_SIZE_Z == 0)
      {
          Dg.x = taille_dF_over_dB[3]/BLOCK_SIZE_X;
          Dg.y = taille_dF_over_dB[2]/BLOCK_SIZE_Y;
          Dg.z = taille_dF_over_dB[1]/BLOCK_SIZE_Z;
      }
      else
      {
          Dg.x = taille_dF_over_dB[3]/BLOCK_SIZE_X+1;
          Dg.y = taille_dF_over_dB[2]/BLOCK_SIZE_Y+1;
          Dg.z = taille_dF_over_dB[1]/BLOCK_SIZE_Z+1;
      }
      gradient_kernel_0<<< Dg , Db >>>(dF_over_dB_dev, taille_dF_over_dB_dev, params_dev, taille_params_dev, M.n_gauss);
*/

      int N = taille_dF_over_dB[3]*taille_dF_over_dB[2];
      Db.x = BLOCK_SIZE_X;
      Db.y = 1;
      Db.z = 1;

      if (N%BLOCK_SIZE_X == 0)
      {
          Dg.x = N/BLOCK_SIZE_X;
          Dg.y = 1;
          Dg.z = 1;
      }
      else
      {
          Dg.x = N/BLOCK_SIZE_X+1;
          Dg.y = 1;
          Dg.z = 1;
      }


//      gradient_kernel_test<<< Dg , Db >>>(dF_over_dB_dev, taille_dF_over_dB_dev, params_dev, taille_params_dev, M.n_gauss);
      gradient_kernel_1<<< Dg , Db >>>(dF_over_dB_dev, taille_dF_over_dB_dev, params_dev, taille_params_dev, M.n_gauss);
//      gradient_kernel_test<<< Dg , Db >>>(dF_over_dB_dev, taille_dF_over_dB_dev, params_dev, taille_params_dev, M.n_gauss);


  checkCudaErrors(cudaMemcpy(dF_over_dB, dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(params, params_dev, product_taille_params*sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(taille_dF_over_dB, taille_dF_over_dB_dev, 4*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(taille_params, taille_params_dev, 4*sizeof(int), cudaMemcpyDeviceToHost));
  // cudaMemcpy(dF_over_dB, dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double), cudaMemcpyDeviceToHost);

  checkCudaErrors(cudaFree(dF_over_dB_dev));
  checkCudaErrors(cudaFree(taille_dF_over_dB_dev));
  checkCudaErrors(cudaFree(params_dev));
  checkCudaErrors(cudaFree(taille_params_dev));

   for(int p; p<2000; p++)
     {
	std::cout<<"dF_over_dB["<<p<<"] = "<<dF_over_dB[p]<<std::endl;
//        printf("p =  %d et dF_over_dB = %f\n",p,dF_over_dB[p]);
     }
   for(int p; p<10; p++)
     {
        printf("p =  %d et taille_dF_over_dB = %d\n",p,taille_dF_over_dB[p]);
     }
   for(int p; p<10; p++)
     {
        printf("p =  %d et params = %f\n",p,params[p]);
     }
   for(int p; p<10; p++)
     {
        printf("p =  %d et taille_params = %d\n",p,taille_params[p]);
     }

     printf("SIZE = %d\n",Dg.x);

}

