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
   checkCudaErrors(cudaMalloc(&taille_params_dev, 3*sizeof(int)));

    checkCudaErrors(cudaMemcpy(dF_over_dB_dev, dF_over_dB, product_taille_dF_over_dB*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_dF_over_dB_dev, taille_dF_over_dB, 4*sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 3*sizeof(int), cudaMemcpyHostToDevice));


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

void gradient_L_2_beta(double* deriv, int* taille_deriv, int product_taille_deriv, double* params, int* taille_params, int product_taille_params, double* residual, int* taille_residual, int product_residual, double* std_map, int* taille_std_map, int product_std_map, int n_gauss)
{
   double* params_dev = NULL;
   double* deriv_dev = NULL;
   double* residual_dev = NULL;
   double* std_map_dev = NULL;

   int* taille_params_dev = NULL;
   int* taille_deriv_dev = NULL;
   int* taille_residual_dev = NULL;
   int* taille_std_map_dev = NULL;

   checkCudaErrors(cudaMalloc(&deriv_dev, product_taille_deriv*sizeof(double)));
   checkCudaErrors(cudaMalloc(&residual_dev, product_residual*sizeof(double)));
   checkCudaErrors(cudaMalloc(&params_dev, product_taille_params*sizeof(double)));
   checkCudaErrors(cudaMalloc(&std_map_dev, product_std_map*sizeof(double)));

   checkCudaErrors(cudaMalloc(&taille_deriv_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_params_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
   
   checkCudaErrors(cudaMemcpy(deriv_dev, deriv, product_taille_deriv*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(residual_dev, residual, product_residual*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_std_map*sizeof(double), cudaMemcpyHostToDevice));
   
   checkCudaErrors(cudaMemcpy(taille_deriv_dev, taille_deriv, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map,2*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

   dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X; //x
    Db.y = BLOCK_SIZE_Y; //y
    Db.z = BLOCK_SIZE_Z; //gaussiennes
        //deriv      --> (3g,y,x)  --> (z,y,x)
        //params     --> (3g,y,x)  --> (z,y,x)
    Dg.x = ceil(taille_deriv[2]/double(BLOCK_SIZE_X));
    Dg.y = ceil(taille_deriv[1]/double(BLOCK_SIZE_Y));
    Dg.z = ceil(n_gauss/double(BLOCK_SIZE_Z));

//  printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);
  gradient_kernel_2_beta<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, params_dev, taille_params_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);
//  printf("DEBUG !!\n");
//  printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);

//printf("DEBUG !!\n");

  checkCudaErrors(cudaMemcpy(deriv, deriv_dev, product_taille_deriv*sizeof(double), cudaMemcpyDeviceToHost));

//printf("DEBUG !!\n");

  checkCudaErrors(cudaFree(deriv_dev));
  checkCudaErrors(cudaFree(taille_deriv_dev));
  checkCudaErrors(cudaFree(params_dev));
  checkCudaErrors(cudaFree(taille_params_dev));
  checkCudaErrors(cudaFree(std_map_dev));
  checkCudaErrors(cudaFree(taille_std_map_dev));
  checkCudaErrors(cudaFree(residual_dev));
  checkCudaErrors(cudaFree(taille_residual_dev));
// printf("DEBUG !!\n");
   }


void gradient_L_2(double* deriv, int* taille_deriv, int product_taille_deriv, double* params, int* taille_params, int product_taille_params, double* residual, int* taille_residual, int product_residual, double* std_map, int* taille_std_map, int product_std_map, int n_gauss)
{
//    cuda_hello<<<1,1>>>();
//  test();

   double* params_dev = NULL;
   double* deriv_dev = NULL;
   double* residual_dev = NULL;
   double* std_map_dev = NULL;

   int* taille_params_dev = NULL;
   int* taille_deriv_dev = NULL;
   int* taille_residual_dev = NULL;
   int* taille_std_map_dev = NULL;

   checkCudaErrors(cudaMalloc(&deriv_dev, product_taille_deriv*sizeof(double)));
   checkCudaErrors(cudaMalloc(&residual_dev, product_residual*sizeof(double)));
   checkCudaErrors(cudaMalloc(&params_dev, product_taille_params*sizeof(double)));
   checkCudaErrors(cudaMalloc(&std_map_dev, product_std_map*sizeof(double)));

   checkCudaErrors(cudaMalloc(&taille_deriv_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_params_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
   
   checkCudaErrors(cudaMemcpy(deriv_dev, deriv, product_taille_deriv*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(residual_dev, residual, product_residual*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_std_map*sizeof(double), cudaMemcpyHostToDevice));
   
   checkCudaErrors(cudaMemcpy(taille_deriv_dev, taille_deriv, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map,2*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

   dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X; //gaussiennes
    Db.y = BLOCK_SIZE_Y; //y
    Db.z = BLOCK_SIZE_Z; //x
        //deriv      --> (3g,y,x)  --> (x,y,z)
        //params     --> (y,x,3g)  --> (y,z,x)
    Dg.x = ceil(n_gauss/double(BLOCK_SIZE_X));
    Dg.y = ceil(taille_deriv[1]/double(BLOCK_SIZE_Y));
    Dg.z = ceil(taille_deriv[2]/double(BLOCK_SIZE_Z));

//    printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);
///    printf("taille_dF_over_dB = %d, %d, %d\n",taille_dF_over_dB[0],taille_dF_over_dB[1],taille_dF_over_dB[2]);

  gradient_kernel_2<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, params_dev, taille_params_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);

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

   void gradient_L_3(double* deriv, int* taille_deriv, int product_taille_deriv, double* params, int* taille_params, int product_taille_params, double* residual, int* taille_residual, int product_residual, double* std_map, int* taille_std_map, int product_std_map, int n_gauss)
{
   double* params_dev = NULL;
   double* deriv_dev = NULL;
   double* residual_dev = NULL;
   double* std_map_dev = NULL;

   int* taille_params_dev = NULL;
   int* taille_deriv_dev = NULL;
   int* taille_residual_dev = NULL;
   int* taille_std_map_dev = NULL;

   checkCudaErrors(cudaMalloc(&deriv_dev, product_taille_deriv*sizeof(double)));
   checkCudaErrors(cudaMalloc(&residual_dev, product_residual*sizeof(double)));
   checkCudaErrors(cudaMalloc(&params_dev, product_taille_params*sizeof(double)));
   checkCudaErrors(cudaMalloc(&std_map_dev, product_std_map*sizeof(double)));

   checkCudaErrors(cudaMalloc(&taille_deriv_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_params_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
   
   checkCudaErrors(cudaMemcpy(deriv_dev, deriv, product_taille_deriv*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(residual_dev, residual, product_residual*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(params_dev, params, product_taille_params*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_std_map*sizeof(double), cudaMemcpyHostToDevice));
   
   checkCudaErrors(cudaMemcpy(taille_deriv_dev, taille_deriv, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_params_dev, taille_params, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map,2*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

   dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X; //gaussiennes
    Db.y = BLOCK_SIZE_Y; //y

    
    Db.z = BLOCK_SIZE_Z; //x
        //deriv      --> (y,x,3g)  --> (y,z,x)
        //params     --> (y,x,3g)  --> (y,z,x)
    Dg.x = ceil(n_gauss/double(BLOCK_SIZE_X));
    Dg.y = ceil(taille_deriv[1]/double(BLOCK_SIZE_Y));
    Dg.z = ceil(taille_deriv[2]/double(BLOCK_SIZE_Z));

//    printf("Dg = %d, %d, %d ; Db = %d, %d, %d\n",Dg.x,Dg.y,Dg.z,Db.x,Db.y,Db.z);
///    printf("taille_dF_over_dB = %d, %d, %d\n",taille_dF_over_dB[0],taille_dF_over_dB[1],taille_dF_over_dB[2]);

  gradient_kernel_3<<<Dg,Db>>>(deriv_dev, taille_deriv_dev, params_dev, taille_params_dev, residual_dev, taille_residual_dev, std_map_dev, taille_std_map_dev, n_gauss);

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

  double compute_residual_and_f(double* beta, int* taille_beta, int product_taille_beta, double* cube, int* taille_cube, int product_taille_cube, double* residual, int* taille_residual, int product_taille_residual, double* std_map, int* taille_std_map, int product_taille_std_map, int indice_x, int indice_y, int indice_v, int n_gauss)
  {
   double* beta_dev = NULL;
   double* cube_dev = NULL;
   double* residual_dev = NULL;
   double* std_map_dev = NULL;

   int* taille_beta_dev = NULL;
   int* taille_cube_dev = NULL;
   int* taille_residual_dev = NULL;
   int* taille_std_map_dev = NULL;

   checkCudaErrors(cudaMalloc(&beta_dev, product_taille_beta*sizeof(double)));
   checkCudaErrors(cudaMalloc(&residual_dev, product_taille_residual*sizeof(double)));
   checkCudaErrors(cudaMalloc(&cube_dev, product_taille_cube*sizeof(double)));
   checkCudaErrors(cudaMalloc(&std_map_dev, product_taille_std_map*sizeof(double)));

   checkCudaErrors(cudaMalloc(&taille_cube_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_beta_dev, 3*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_std_map_dev, 2*sizeof(int)));
   checkCudaErrors(cudaMalloc(&taille_residual_dev, 3*sizeof(int)));
   
   checkCudaErrors(cudaMemcpy(beta_dev, beta, product_taille_beta*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(residual_dev, residual, product_taille_residual*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(cube_dev, cube, product_taille_cube*sizeof(double), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(std_map_dev, std_map, product_taille_std_map*sizeof(double), cudaMemcpyHostToDevice));
   
   checkCudaErrors(cudaMemcpy(taille_cube_dev, taille_cube, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_beta_dev, taille_beta, 3*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_std_map_dev, taille_std_map,2*sizeof(int), cudaMemcpyHostToDevice));
   checkCudaErrors(cudaMemcpy(taille_residual_dev, taille_residual, 3*sizeof(int), cudaMemcpyHostToDevice));

   dim3 Dg, Db;
    Db.x = BLOCK_SIZE_X; //
    Db.y = BLOCK_SIZE_Y; //
    Db.z = BLOCK_SIZE_Z; //

    Dg.x = ceil(indice_x/double(BLOCK_SIZE_X));
    Dg.y = ceil(indice_y/double(BLOCK_SIZE_Y));
    Dg.z = ceil(indice_v/double(BLOCK_SIZE_Z));

// index_x -> indice_x
// index_y -> indice_y
// index_z -> indice_v

//    printf("beta[0] = %f \n", beta[0]);

    kernel_residual<<<Dg,Db>>>(beta_dev, cube_dev, residual_dev,indice_x, indice_y, indice_v, n_gauss);

//    printf("residual[0] = %f \n",residual[0]);



    dim3 Dg_L2, Db_L2;
    Db_L2.x = BLOCK_SIZE_L2_X;
    Db_L2.y = BLOCK_SIZE_L2_Y;
    Db_L2.z = 1;//BLOCK_SIZE_L2_Z;
    Dg_L2.x = ceil(indice_x/double(BLOCK_SIZE_L2_X));
    Dg_L2.y = ceil(indice_y/double(BLOCK_SIZE_L2_Y));
    Dg_L2.z = 1;//ceil(indice_x/double(BLOCK_SIZE_L2_Z));

    double* map_norm_dev = NULL;
    checkCudaErrors(cudaMalloc(&map_norm_dev, indice_x*indice_y*sizeof(double)));

    kernel_norm_map_boucle_v<<<Dg_L2, Db_L2>>>(map_norm_dev, residual_dev, taille_residual_dev, std_map_dev, indice_x, indice_y, indice_v);

/*
//vérification kernel_norm_map_boucle_v
    double* map_norm_host = NULL;
    map_norm_host = (double*)malloc(indice_x*indice_y*sizeof(double));
    checkCudaErrors(cudaMemcpy(map_norm_host, map_norm_dev, indice_x*indice_y*sizeof(double), cudaMemcpyDeviceToHost));
    for(int p = 0; p<indice_x*indice_y; p++)
    {
      f+= map_norm_host[p];
    }
    printf("f = %f \n", f);
exit(0);
  
    printf("indice_x = %d , indice_y = %d , indice_v = %d , BLOCK_SIZE_REDUCTION = %d \n", indice_x, indice_y, indice_v, BLOCK_SIZE_REDUCTION);
    printf("int(ceil(double(indice_x*indice_y)/double(BLOCK_SIZE_REDUCTION))) = %d \n", int(ceil(double(indice_x*indice_y)/double(BLOCK_SIZE_REDUCTION))));
    printf("Dg = %d , Db = %d\n",int(ceil(double(indice_x*indice_y)/double(BLOCK_SIZE_REDUCTION))), BLOCK_SIZE_REDUCTION);
*/

    int GRID_SIZE_REDUCTION = int(ceil(double(indice_x*indice_y)/double(BLOCK_SIZE_REDUCTION)));
    double* tab_cpy_cpu = NULL;
    tab_cpy_cpu = (double*)malloc(GRID_SIZE_REDUCTION*sizeof(double));

    double* tab_test_dev_out=NULL;
    checkCudaErrors(cudaMalloc(&tab_test_dev_out, GRID_SIZE_REDUCTION*sizeof(double)));

    sum_reduction<<< GRID_SIZE_REDUCTION, BLOCK_SIZE_REDUCTION >>>(map_norm_dev, tab_test_dev_out, indice_x*indice_y);

    checkCudaErrors(cudaMemcpy(tab_cpy_cpu, tab_test_dev_out, GRID_SIZE_REDUCTION*sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.;
    for(int o = 0; o<GRID_SIZE_REDUCTION ; o++)
    {
      sum+=tab_cpy_cpu[o];
    }
    
    checkCudaErrors(cudaMemcpy(residual, residual_dev, product_taille_residual*sizeof(double), cudaMemcpyDeviceToHost));

/*
    printf("beta[0] = %f \n",beta[0]);
    printf("residual[0] = %f \n",residual[0]);
    printf("residual[1] = %f \n",residual[1]);
    printf("residual[2] = %f \n",residual[2]);
    printf("residual[3] = %f \n",residual[3]);
*/

    checkCudaErrors(cudaFree(tab_test_dev_out));

    checkCudaErrors(cudaFree(map_norm_dev));
    checkCudaErrors(cudaFree(beta_dev));
    checkCudaErrors(cudaFree(taille_beta_dev));
    checkCudaErrors(cudaFree(cube_dev));
    checkCudaErrors(cudaFree(taille_cube_dev));
    checkCudaErrors(cudaFree(std_map_dev));
    checkCudaErrors(cudaFree(taille_std_map_dev));
    checkCudaErrors(cudaFree(residual_dev));
    checkCudaErrors(cudaFree(taille_residual_dev));

  return sum;
  }