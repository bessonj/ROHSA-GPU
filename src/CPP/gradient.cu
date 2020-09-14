#include "gradient.hpp"
#include "kernel_gradient.cuh"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

void test(){
  cuda_hello<<<1,1>>>();
}


void gradient_L(double* dF_over_dB, int* taille_dF_over_dB, int product_taille_dF_over_dB, double* params, int* taille_params, int product_taille_params, int n_gauss)
{
//    cuda_hello<<<1,1>>>();
//  test();

   double* params_dev;
   double* dF_over_dB_dev;
   int* taille_dF_over_dB_dev;
   int* taille_params_dev;


   checkCudaErrors(cudaMalloc(&dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double)));
   checkCudaErrors(cudaMalloc(&params_dev, product_taille_params*sizeof(double)));
   checkCudaErrors(cudaMalloc(&taille_dF_over_dB_dev, 4*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_params_dev, 4*sizeof(int)));

    checkCudaErrors(cudaMemcpy(dF_over_dB_dev, dF_over_dB, product_taille_dF_over_dB*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_dF_over_dB_dev, taille_dF_over_dB, 4*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 4*sizeof(int), cudaMemcpyHostToDevice));


//   cudaMalloc((void**)&dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double));

//   cudaMemcpy(dF_over_dB_dev, dF_over_dB, product_taille_dF_over_dB*sizeof(double), cudaMemcpyHostToDevice);

   dim3 Dg, Db;

    Db.x = BLOCK_SIZE_X; //gaussiennes
    Db.y = BLOCK_SIZE_Y; //x
    Db.z = BLOCK_SIZE_Z; //y
        //dF_over_dB --> (v,y,x,ng)  --> (i,z,y,x)
        //params     --> (y,x,ng)
    Dg.x = ceil(n_gauss/double(BLOCK_SIZE_X));
    Dg.y = ceil(taille_dF_over_dB[2]/double(BLOCK_SIZE_Y));
    Dg.z = ceil(taille_dF_over_dB[1]/double(BLOCK_SIZE_Z));

///    printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);
///    printf("taille_dF_over_dB = %d, %d, %d\n",taille_dF_over_dB[0],taille_dF_over_dB[1],taille_dF_over_dB[2]);

//    cuda_hello<<<Dg,Db>>>();
  gradient_kernel_1<<<Dg,Db>>>(dF_over_dB_dev, taille_dF_over_dB_dev, params_dev, taille_params_dev, n_gauss);


//  cudaMemcpy(dF_over_dB, dF_over_dB_dev, product_taille_dF_over_dB,cudaMemcpyDeviceToHost);

  checkCudaErrors(cudaMemcpy(dF_over_dB, dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double), cudaMemcpyDeviceToHost));
/*
  checkCudaErrors(cudaMemcpy(params, params_dev, product_taille_params*sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(taille_dF_over_dB, taille_dF_over_dB_dev, 4*sizeof(int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(taille_params, taille_params_dev, 4*sizeof(int), cudaMemcpyDeviceToHost));
*/

  // cudaMemcpy(dF_over_dB, dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double), cudaMemcpyDeviceToHost);

  checkCudaErrors(cudaFree(dF_over_dB_dev));
  checkCudaErrors(cudaFree(taille_dF_over_dB_dev));
  checkCudaErrors(cudaFree(params_dev));
  checkCudaErrors(cudaFree(taille_params_dev));
}

void gradient_L_2(double* deriv, int* taille_deriv, int product_taille_deriv, double* params, int* taille_params, int product_taille_params, double* residual, int* taille_residual, int product_residual, double* std_map, int* taille_std_map, int product_std_map, int n_gauss)
{
//    cuda_hello<<<1,1>>>();
//  test();

//   printf("test1");

   double* params_dev = NULL;
   double* deriv_dev = NULL;
   double* residual_dev = NULL;
   double* std_map_dev = NULL;
   int* taille_params_dev = NULL;
   int* taille_deriv_dev = NULL;
   int* taille_residual_dev = NULL;
   int* taille_std_map_dev = NULL;
printf("test1.5 ");
printf("taille = %d \n",product_taille_deriv);
   checkCudaErrors(cudaMalloc(&deriv_dev, product_taille_deriv*sizeof(double)));
printf("test1.6 ");
printf("taille = %d \n",product_residual);
   checkCudaErrors(cudaMalloc(&residual_dev, product_residual*sizeof(double)));
printf("test1.7 ");
printf("taille = %d \n",product_taille_params);
   checkCudaErrors(cudaMalloc(&params_dev, product_taille_params*sizeof(double)));
printf("test1.8 ");
printf("taille = %d \n",product_std_map); //problème juste après
   checkCudaErrors(cudaMalloc(&std_map_dev, product_std_map*sizeof(double)));
printf("test1.9 \n");
   checkCudaErrors(cudaMalloc(&taille_deriv_dev, 3*sizeof(int)));
printf("test1.10 \n");
   checkCudaErrors(cudaMalloc(&taille_params_dev, 3*sizeof(int)));
printf("test1.11 \n");
   checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
printf("test1.12 \n");
   checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
printf("test2 \n");
    checkCudaErrors(cudaMemcpy(deriv_dev, deriv, product_taille_deriv*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(residual_dev, residual, product_residual*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_std_map*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_deriv_dev, taille_deriv, 3*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 3*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map,2*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

printf("test3 \n");
//   cudaMalloc((void**)&dF_over_dB_dev, product_taille_dF_over_dB*sizeof(double));

//   cudaMemcpy(dF_over_dB_dev, dF_over_dB, product_taille_dF_over_dB*sizeof(double), cudaMemcpyHostToDevice);

   dim3 Dg, Db;

    Db.x = BLOCK_SIZE_X; //gaussiennes
    Db.y = BLOCK_SIZE_Y; //x
    Db.z = BLOCK_SIZE_Z; //y
        //dF_over_dB --> (v,y,x,ng)  --> (i,z,y,x)
        //params     --> (y,x,ng)
    Dg.x = ceil(n_gauss/double(BLOCK_SIZE_X));
    Dg.y = ceil(taille_deriv[2]/double(BLOCK_SIZE_Y));
    Dg.z = ceil(taille_deriv[1]/double(BLOCK_SIZE_Z));

//printf("test4");
//    printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);
///    printf("taille_dF_over_dB = %d, %d, %d\n",taille_dF_over_dB[0],taille_dF_over_dB[1],taille_dF_over_dB[2]);

  gradient_kernel_2<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, params_dev, taille_params_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);

//printf("test5");
  checkCudaErrors(cudaMemcpy(deriv, deriv_dev, product_taille_deriv*sizeof(double), cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(deriv_dev));
  checkCudaErrors(cudaFree(taille_deriv_dev));
  checkCudaErrors(cudaFree(params_dev));
  checkCudaErrors(cudaFree(taille_params_dev));
  checkCudaErrors(cudaFree(std_map_dev));
  checkCudaErrors(cudaFree(taille_std_map_dev));
  checkCudaErrors(cudaFree(residual_dev));
  checkCudaErrors(cudaFree(taille_residual_dev));
 
   }