#include <iostream> 
#include <stdio.h>
#include <stdlib.h>
//#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "gradient.hpp"
#include "model.hpp"

#include "kernel_gradient.cuh"





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

