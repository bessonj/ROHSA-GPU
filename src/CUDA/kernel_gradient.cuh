#ifndef _GRADIENT_KERNEL_H_
#define _GRADIENT_KERNEL_H_

#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]

#include <stdio.h>

__global__ void cuda_hello(){

	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;
	int A = threadIdx.x + (blockDim.y*blockIdx.y);
	int B = blockDim.y*blockIdx.y;

//printf("threadIdx.x = %d , threadIdx.y = %d , threadIdx.z = %d \n",threadIdx.x,threadIdx.y,threadIdx.z);
//printf(" blockIdx.x = %d , blockIdx.y = %d , blockIdx.z = %d \n",blockIdx.x,blockIdx.y,blockIdx.z);
//printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);

if(index_x<2 && index_y<2 && index_z<9)
{
//printf("threadIdx.x = %d , threadIdx.y = %d , threadIdx.z = %d \n",threadIdx.x,threadIdx.y,threadIdx.z);

printf("index_x = %d , index_y = %d , index_z = %d , A = %d , B = %d\n",index_x,index_y,index_z, A , B);
}
//printf("blockIdx.x = %d , index_y = %d , index_z = %d , A = %d , B = %d\n",blockIdx.x,index_y,index_z, A , B);
//	printf("OK\n");
	__syncthreads();
}

__global__ void gradient_kernel_0(double* dF_over_dB_dev, int* t, double* params, int* t_p, int n_gauss)
{ 
	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;

	int t_1 = t[1];
	int t_2 = t[2];
	int t_3 = t[3];
	int t_p_1 = t_p[1];
	int t_p_2 = t_p[2];

printf("threadIdx.x = %d , threadIdx.y = %d , threadIdx.z = %d \n",threadIdx.x,threadIdx.y,threadIdx.z);


//printf("lim_x = %d , lim_y = %d , lim_z = %d\n",t[2],t[1],t[0]);

//printf("index_x = %d , index_y = %d , index_z = %d , A = %d , B = %d\n",index_x,index_y,index_z, A , B);

//printf("blockIdx.x = %d , index_y = %d , index_z = %d , A = %d , B = %d\n",blockIdx.x,index_y,index_z, A , B);

/*
if(index_x<t[2] && index_y<t[1] && index_z<t[0])
{
	printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
}
*/

if(index_x<t[2] && index_y<t[1] && index_z<t[0])
{
	printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
for(int i=0; i<n_gauss; i++){
//      double par0 = params[index_y][index_x][3*i+0];
    double par0 = params[index_y*t_p_1*t_p_2+index_x*t_p_2+(3*i+0)];
//	double par1 = double(k+1) - params[index_y][index_x][3*i+1];
	double par1 = double(index_z+1) - params[index_y*t_p_1*t_p_2+index_x*t_p_2+3*i+1];
//	double par2 = params[index_y][index_x][3*i+2];
	double par2 = params[index_y*t_p_1*t_p_2+index_x*t_p_2+3*i+2];
	double par2_pow = 1/(2*pow(par2,2.));
//printf("par2_pow = %d",par2_pow);

//dF_over_dB[v][y][x][M.n__gauss]
//	dF_over_dB[k][l][j][3*i+0] += exp( -pow( par1 ,2.)*par2_pow );//( -pow( par1_k ,2.)*par2_pow );
	dF_over_dB_dev[index_z*t_3*t_2*t_1+index_y*t_3*t_2+index_x*t_3+3*i+0] = exp( -pow( par1 ,2.)*par2_pow );//( -pow( par1_k ,2.)*par2_pow );
//	dF_over_dB[k][l][j][3*i+1] += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ); //par0*par1_k*par2_pow*2 * exp(-pow( par1_k,2.)*par2_pow );
	dF_over_dB_dev[index_z*t_3*t_2*t_1+index_y*t_3*t_2+index_x*t_3+3*i+1] = par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ); //par0*par1_k*par2_pow*2 * exp(-pow( par1_k,2.)*par2_pow );
//	dF_over_dB[k][l][j][3*i+2] += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow );//par0*pow( par1_k, 2.)/(pow(par2,3.)) * exp(-pow( par1_k,2.)*par2_pow );
	dF_over_dB_dev[index_z*t_3*t_2*t_1+index_y*t_3*t_2+index_x*t_3+3*i+2] = par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow );//par0*pow( par1_k, 2.)/(pow(par2,3.)) * exp(-pow( par1_k,2.)*par2_pow );
}



}

//printf("dF_over_dB_dev[%d] = %f\n",index_z*t_3*t_2*t_1+index_y*t_3*t_2+index_x*t_3+2,dF_over_dB_dev[index_z*t_3*t_2*t_1+index_y*t_3*t_2+index_x*t_3+2]);
__syncthreads();

}

/*
for(int k=0; k<indice_v; k++){
	for(int l=0; l<indice_y; l++){
		for(int j=0; j<indice_x; j++){
			for(int i=0; i<M.n_gauss; i++){
				double par0 = params_T[l][j][3*i+0];
				double par1 = double(k+1) - params_T[j][l][3*i+1];
				double par2 = params_T[l][j][3*i+2];

//				double par1_k = double(k+1) - par1;
				double par2_pow = 1/(2*pow(par2,2.));
//dF_over_dB[v][y][x][M.n__gauss]
				dF_over_dB[k][l][j][3*i+0] += exp( -pow( par1 ,2.)*par2_pow );//( -pow( par1_k ,2.)*par2_pow );
				dF_over_dB[k][l][j][3*i+1] += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ); //par0*par1_k*par2_pow*2 * exp(-pow( par1_k,2.)*par2_pow );
				dF_over_dB[k][l][j][3*i+2] += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow );//par0*pow( par1_k, 2.)/(pow(par2,3.)) * exp(-pow( par1_k,2.)*par2_pow );
			}
		}
	}
}
*/

__global__ void gradient_kernel_1(double* dF_over_dB_dev, int* t, double* params, int* t_p, int n_gauss)
{ 
	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;

	int t_1 = t[1]; // y
	int t_2 = t[2]; // x
	int t_3 = 3*n_gauss;//t[3]; // ng
	int t_p_1 = t_p[1];
	int t_p_2 = 3*n_gauss;//t_p[2];

    //dF_over_dB --> (v,y,x,ng)  --> (i,z,y,x)
    //params     --> (y,x,ng)    --> (z,y,x)

if(index_x<n_gauss && index_y<t[2] && index_z<t[1])
{
  //dF_over_dB --> (v,y,x,ng)
for(int i=0; i<t[0]; i++){
//	printf("i = %d , index_z = %d , index_y = %d , index_x = %d\n",i,index_z,index_y,index_x);

//  double par0 = params[index_z][index_y][3*index_x+0];
    double par0 = params[index_y*t_p_1*t_p_2+index_z*t_p_2+(3*index_x+0)];

//	double par1 = double(k+1) - params[index_z][index_y][3*index_x+1];
	double par1 = double(i+1) - params[index_z*t_p_1*t_p_2+index_y*t_p_2+3*index_x+1];

//	double par2 = params[index_z][index_y][3*index_x+2];
	double par2 = params[index_z*t_p_1*t_p_2+index_y*t_p_2+3*index_x+2];

	double par2_pow = 1/(2*pow(par2,2.));
//printf("par2_pow = %d",par2_pow);

//dF_over_dB[v][y][x][M.n__gauss]
//	dF_over_dB[k][l][j][3*ng+0] += exp( -pow( par1 ,2.)*par2_pow );
	dF_over_dB_dev[i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+0] = exp( -pow( par1 ,2.)*par2_pow );

//	dF_over_dB[k][l][j][3*ng+1] += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow );
	dF_over_dB_dev[i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+1] = par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow );

//	dF_over_dB[k][l][j][3*ng+2] += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow );
	dF_over_dB_dev[i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+2] = par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow );
}
__syncthreads();
}

/*
for(int i=0; i<t[0]; i++){
	printf("dF_over_dB_dev[%d] = %f\n",i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+0,dF_over_dB_dev[i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+0]);
	printf("dF_over_dB_dev[%d] = %f\n",i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+1,dF_over_dB_dev[i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+1]);
	printf("dF_over_dB_dev[%d] = %f\n",i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+2,dF_over_dB_dev[i*t_3*t_2*t_1+index_z*t_3*t_2+index_y*t_3+3*index_x+2]);
}
*/
__syncthreads();

}

__global__ void gradient_kernel_2(double* deriv, int* t_d, double* params, int* t_p, double* residual, int* t_r, double* std_map, int* t_std, int n_gauss)
{ 

	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;
__syncthreads();

	//taille tableau residual
	int tr0=t_r[0]; // x --> index_z
	int tr1=t_r[1]; // y --> index_y
    int tr2=t_r[2]; // v --> i

	//taille tableau std_map
	int ts0=t_std[0]; // y --> index_y
	int ts1=t_std[1]; // x --> index_z

	//taille tableau deriv
	int td0 = t_d[0]; // 3*ng --> index_x
	int td1 = t_d[1]; // y --> index_y
	int td2 = t_d[2]; // x --> index_z

	//taille params_flat
	int tp0 = t_p[0]; // y --> index_y
	int tp1 = t_p[1]; // x --> index_z
	int tp2 = t_p[2]; // 3*ng --> index_x

//						ROHSA world			dev world 
        //params     --> (y,x,ng)    --> (y,z,x)
		//residual   --> (x,y,z)     --> (z,y,i)
		//deriv      --> (ng,y,x)    --> (x,y,z)
		//std_map    --> (y,x)       --> (y,z)

	
	if(index_x<n_gauss && index_z<td2 && index_y<td1 && std_map[index_y*ts1+index_z]>0.)
	{
		double par0 = params[index_y*tp1*tp2+index_z*tp2+(3*index_x+0)];

//	printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
//	printf(" par0 = %d \n", par0);// = %d , index_z = %d\n",index_x,index_y,index_z);

		double par1_a = params[index_y*tp1*tp2+index_z*tp2+(3*index_x+1)];
		double par2 = params[index_y*tp1*tp2+index_z*tp2+(3*index_x+2)];
		double par2_pow = 1/(2*pow(par2,2.));
		double parstd = 1/pow(std_map[index_y*ts1+index_z],2);

		double buffer_0 = 0.;        //dF_over_dB --> (v,y,x,3g)  --> (i,z,y,x)
		double buffer_1 = 0.;
		double buffer_2 = 0.;

		for(int i=0; i<tr2; i++){
//			printf("i = %d , index_z = %d , index_y = %d , index_x = %d\n",i,index_z,index_y,index_x);
			double par1 = double(i) - par1_a;

			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
			buffer_1 += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
		}

//		printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
//		printf(" buffer_0 = %f ,  buffer_1 = %f ,  buffer_2 = %f \n", buffer_0, buffer_1, buffer_2);// = %d , index_z = %d\n",index_x,index_y,index_z);

		__syncthreads();
		deriv[(3*index_x+0)*td1*td2+index_y*td2+index_z]=buffer_0;
		deriv[(3*index_x+1)*td1*td2+index_y*td2+index_z]=buffer_1;
		deriv[(3*index_x+2)*td1*td2+index_y*td2+index_z]=buffer_2;
	}

	__syncthreads();
}

__global__ void gradient_kernel_2_beta(double* deriv, int* t_d, double* params, int* t_p, double* residual, int* t_r, double* std_map, int* t_std, int n_gauss)
{ 
	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;
	__syncthreads();

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

	
	if(index_z<n_gauss && index_x<td2 && index_y<td1 && std_map[index_y*ts1+index_x]>0.)
	{
		double par0 = params[(3*index_z+0)*tp1*tp2+index_y*tp2+index_x];

//	printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
//	printf(" par0 = %d \n", par0);// = %d , index_z = %d\n",index_x,index_y,index_z);

		double par1_a = params[(3*index_z+1)*tp1*tp2+index_y*tp2+index_x];
		double par2 = params[(3*index_z+2)*tp1*tp2+index_y*tp2+index_x];
		double par2_pow = 1/(2*pow(par2,2.));
		double parstd = 1/pow(std_map[index_y*ts1+index_x],2);

		double buffer_0 = 0.;        //dF_over_dB --> (v,y,x,3g)  --> (i,x,y,z)
		double buffer_1 = 0.;
		double buffer_2 = 0.;

		for(int i=0; i<tr0; i++){
//			printf("i = %d , index_z = %d , index_y = %d , index_x = %d\n",i,index_z,index_y,index_x);
			double par1 = double(i) - par1_a;

			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * residual[i*tr1*tr2+index_y*tr2+index_x]*parstd;
			buffer_1 += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ) * residual[i*tr1*tr2+index_y*tr2+index_x]*parstd;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * residual[i*tr1*tr2+index_y*tr2+index_x]*parstd;
		}

//		printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
//		printf(" buffer_0 = %f ,  buffer_1 = %f ,  buffer_2 = %f \n", buffer_0, buffer_1, buffer_2);// = %d , index_z = %d\n",index_x,index_y,index_z);

		__syncthreads();
		deriv[(3*index_z+0)*td1*td2+index_y*td2+index_x]=buffer_0;
		deriv[(3*index_z+1)*td1*td2+index_y*td2+index_x]=buffer_1;
		deriv[(3*index_z+2)*td1*td2+index_y*td2+index_x]=buffer_2;
	}

	__syncthreads();
}

__global__ void gradient_kernel_2_beta_with_INDEXING(double* deriv, int* t_d, double* params, int* t_p, double* residual, int* t_r, double* std_map, int* t_std, int n_gauss)
{ 
	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;

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
		double par0 = INDEXING_3D(params,(3*index_z+0),index_y, index_x);

		double par1_a = INDEXING_3D(params, (3*index_z+1), index_y, index_x);
		double par2 = INDEXING_3D(params, (3*index_z+2), index_y, index_x);
		double par2_pow = 1/(2*pow(par2,2.));
		double parstd = 1/pow(INDEXING_2D(std_map, index_y, index_x),2);

		double buffer_0 = 0.;        //dF_over_dB --> (v,y,x,3g)  --> (i,x,y,z)
		double buffer_1 = 0.;
		double buffer_2 = 0.;

		for(int i=0; i<residual_SHAPE0; i++){
			double par_res = INDEXING_3D(residual,i,index_y,index_x)* parstd;
//			printf("i = %d , index_z = %d , index_y = %d , index_x = %d\n",i,index_z,index_y,index_x);
			double par1 = double(i) - par1_a;

			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * par_res;
			buffer_1 += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ) * par_res;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * par_res;
		}

//		printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
//		printf(" buffer_0 = %f ,  buffer_1 = %f ,  buffer_2 = %f \n", buffer_0, buffer_1, buffer_2);// = %d , index_z = %d\n",index_x,index_y,index_z);
		__syncthreads();
		INDEXING_3D(deriv,(3*index_z+0), index_y, index_x)=buffer_0;
		INDEXING_3D(deriv,(3*index_z+1), index_y, index_x)=buffer_1;
		INDEXING_3D(deriv,(3*index_z+2), index_y, index_x)=buffer_2;
	}

	__syncthreads();
}

__global__ void gradient_kernel_2_beta_working(double* deriv, int* t_d, double* params, int* t_p, double* residual, int* t_r, double* std_map, int* t_std, int n_gauss)
{ 

	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;
	__syncthreads();

	//taille tableau residual
	int tr0=t_r[0]; // x --> index_z
	int tr1=t_r[1]; // y --> index_y
    int tr2=t_r[2]; // v --> i

	//taille tableau std_map
	int ts0=t_std[0]; // y --> index_y
	int ts1=t_std[1]; // x --> index_z

	//taille tableau deriv
	int td0 = t_d[0]; // 3*ng --> index_x
	int td1 = t_d[1]; // y --> index_y
	int td2 = t_d[2]; // x --> index_z

	//taille params_flat
	int tp0 = t_p[0]; // x --> index_z
	int tp1 = t_p[1]; // y --> index_y
	int tp2 = t_p[2]; // 3*ng --> index_x

//						ROHSA world			dev world 
        //params     --> (x,y,ng)    --> (z,y,x)
		//residual   --> (x,y,z)     --> (z,y,i)
		//deriv      --> (ng,y,x)    --> (x,y,z)
		//std_map    --> (y,x)       --> (y,z)

	
	if(index_x<n_gauss && index_z<td2 && index_y<td1 && std_map[index_y*ts1+index_z]>0.)
	{
		double par0 = params[index_z*tp1*tp2+index_y*tp2+(3*index_x+0)];

//	printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
//	printf(" par0 = %d \n", par0);// = %d , index_z = %d\n",index_x,index_y,index_z);

		double par1_a = params[index_z*tp1*tp2+index_y*tp2+(3*index_x+1)];
		double par2 = params[index_z*tp1*tp2+index_y*tp2+(3*index_x+2)];
		double par2_pow = 1/(2*pow(par2,2.));
		double parstd = 1/pow(std_map[index_y*ts1+index_z],2);

		double buffer_0 = 0.;        //dF_over_dB --> (v,y,x,3g)  --> (i,z,y,x)
		double buffer_1 = 0.;
		double buffer_2 = 0.;

		for(int i=0; i<tr2; i++){
//			printf("i = %d , index_z = %d , index_y = %d , index_x = %d\n",i,index_z,index_y,index_x);
			double par1 = double(i) - par1_a;

			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
			buffer_1 += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
		}

//		printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
//		printf(" buffer_0 = %f ,  buffer_1 = %f ,  buffer_2 = %f \n", buffer_0, buffer_1, buffer_2);// = %d , index_z = %d\n",index_x,index_y,index_z);

		__syncthreads();
		deriv[(3*index_x+0)*td1*td2+index_y*td2+index_z]=buffer_0;
		deriv[(3*index_x+1)*td1*td2+index_y*td2+index_z]=buffer_1;
		deriv[(3*index_x+2)*td1*td2+index_y*td2+index_z]=buffer_2;
	}

	__syncthreads();
}

__global__ void gradient_kernel_3(double* deriv, int* t_d, double* params, int* t_p, double* residual, int* t_r, double* std_map, int* t_std, int n_gauss)
{ 
	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;
	__syncthreads();

	//taille tableau residual
	int tr0=t_r[0]; // x --> index_z
	int tr1=t_r[1]; // y --> index_y
    int tr2=t_r[2]; // v --> i

	//taille tableau std_map
	int ts0=t_std[0]; // y --> index_y
	int ts1=t_std[1]; // x --> index_z

	//taille tableau deriv
	int td0 = t_d[0]; // y --> index_y
	int td1 = t_d[1]; // x --> index_z
	int td2 = t_d[2]; // 3*ng --> index_x

	//taille params_flat
	int tp0 = t_p[0]; // y --> index_y
	int tp1 = t_p[1]; // x --> index_z
	int tp2 = t_p[2]; // 3*ng --> index_x

//						ROHSA world			dev world 
        //params     --> (y,x,ng)    --> (y,z,x)
		//residual   --> (x,y,z)     --> (z,y,i)
		//deriv      --> (ng,y,x)    --> (y,z,x)
		//std_map    --> (y,x)       --> (y,z)

	
	if(index_x<n_gauss && index_y<td2 && index_z<td1 && std_map[index_y*ts1+index_z]>0.)
	{
		double par0 = params[index_y*tp1*tp2+index_z*tp2+(3*index_x+0)];

		double par1_a = params[index_y*tp1*tp2+index_z*tp2+(3*index_x+1)];
		double par2 = params[index_y*tp1*tp2+index_z*tp2+(3*index_x+2)];
		double par2_pow = 1/(2*pow(par2,2.));
		double parstd = 1/pow(std_map[index_y*ts1+index_z],2);

		double buffer_0 = 0.;        //dF_over_dB --> (v,y,x,3g)  --> (i,z,y,x)
		double buffer_1 = 0.;
		double buffer_2 = 0.;

		for(int i=0; i<tr2; i++){
			double par1 = double(i) - par1_a;
			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
			buffer_1 += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
		}

		__syncthreads();
		deriv[index_y*td1*td2 + index_z*td2  +  (3*index_x+0)]=buffer_0;
		deriv[index_y*td1*td2 + index_z*td2  +  (3*index_x+1)]=buffer_1;
		deriv[index_y*td1*td2 + index_z*td2  +  (3*index_x+2)]=buffer_2;
	}
	__syncthreads();
}


__global__ void kernel_residual(double* beta, double* cube, double* residual, int indice_x, int indice_y, int indice_v, int n_gauss)
{
	int index_x = blockIdx.x*BLOCK_SIZE_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_Z +threadIdx.z;

	if(index_x<indice_x && index_y<indice_y && index_z<indice_v)
	{
  	double model_gaussienne = 0.;
		for(int g = 0; g<n_gauss; g++)
		{
			double par[3];
			par[0]=beta[(3*g+0)*indice_y*indice_x+index_y*indice_x+index_x];
			par[1]=beta[(3*g+1)*indice_y*indice_x+index_y*indice_x+index_x];
			par[2]=beta[(3*g+2)*indice_y*indice_x+index_y*indice_x+index_x];					
			model_gaussienne += par[0]*exp(-pow((double(index_z)-par[1]),2.) / (2.*pow(par[2],2.)));
		}
		residual[index_z*indice_y*indice_x+index_y*indice_x+index_x]=model_gaussienne-cube[index_z*indice_y*indice_x+index_y*indice_x+index_x];
	}
}

__global__ void kernel_norm_map(double* map_norm_dev, double* residual, int* taille_residual, double* std_map, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*BLOCK_SIZE_L2_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_L2_Y +threadIdx.y;
	int index_z = blockIdx.z*BLOCK_SIZE_L2_Z +threadIdx.z;

  if(index_z<indice_x && index_y<indice_y && index_x<indice_v && std_map[index_y*indice_x + index_z]>0.)
  {
    map_norm_dev[index_y*indice_x+index_z] += 0.5*pow(residual[index_z*indice_y*indice_v+index_y*indice_v+index_x],2)/pow(std_map[index_y*indice_x + index_z],2);
	__syncthreads();

//  printf("index_x = %d , index_y = %d , index_z = %d \n", index_x, index_y, index_z);

//	map_norm_dev[index_y*indice_x + index_z] = 0.5*map_norm_dev[index_y*indice_x + index_z]/pow(std_map[index_y*indice_x + index_z],2);
//	printf(" --> %f \n", map_norm_dev[index_y*indice_x + index_z]);
  }
}

__global__ void kernel_norm_map_boucle_v(double* map_norm_dev, double* residual, int* taille_residual, double* std_map, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*BLOCK_SIZE_L2_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_L2_Y +threadIdx.y;
//	int index_z = blockIdx.z*BLOCK_SIZE_L2_Z +threadIdx.z;

  if(index_x<indice_x && index_y<indice_y && std_map[index_y*indice_x + index_x]>0.)
  {
	  for(int index_z = 0; index_z<indice_v ; index_z++) {
    	map_norm_dev[index_y*indice_x+index_x] += 0.5*pow(residual[index_z*indice_y*indice_x+index_y*indice_x+index_x],2)/pow(std_map[index_y*indice_x + index_x],2);
	  }
	__syncthreads();

//  printf("index_x = %d , index_y = %d , index_z = %d \n", index_x, index_y, index_z);

//	map_norm_dev[index_y*indice_x + index_z] = 0.5*map_norm_dev[index_y*indice_x + index_z]/pow(std_map[index_y*indice_x + index_x],2);
//	printf(" --> %f \n", map_norm_dev[index_y*indice_x + index_z]);
  }
}

__global__ void kernel_norm_map_boucle_v(double* map_norm_dev, double* residual, double* std_map, int indice_x, int indice_y, int indice_v)
{
	int index_x = blockIdx.x*BLOCK_SIZE_L2_X +threadIdx.x;
	int index_y = blockIdx.y*BLOCK_SIZE_L2_Y +threadIdx.y;
//	int index_z = blockIdx.z*BLOCK_SIZE_L2_Z +threadIdx.z;

  if(index_x<indice_x && index_y<indice_y && std_map[index_y*indice_x + index_x]>0.)
  {
	  for(int index_z = 0; index_z<indice_v ; index_z++) {
    	map_norm_dev[index_y*indice_x+index_x] += 0.5*pow(residual[index_z*indice_y*indice_x+index_y*indice_x+index_x],2)/pow(std_map[index_y*indice_x + index_x],2);
	  }
	__syncthreads();

//  printf("index_x = %d , index_y = %d , index_z = %d \n", index_x, index_y, index_z);

//	map_norm_dev[index_y*indice_x + index_z] = 0.5*map_norm_dev[index_y*indice_x + index_z]/pow(std_map[index_y*indice_x + index_x],2);
//	printf(" --> %f \n", map_norm_dev[index_y*indice_x + index_z]);
  }
}

__global__ void dot(double* a, double* b, double* c, int N)
{
	__shared__ double cache[BLOCK_SIZE_REDUCTION];
	int tid = threadIdx.x +blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	double temp = 0.;
	while (tid<N) {
		temp += a[tid]*b[tid];
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

__global__ void sum_reduction(double* a, double* c, int N)
{
	__shared__ double cache[BLOCK_SIZE_REDUCTION];
	int tid = threadIdx.x +blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	double temp = 0.;
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




__global__ void cpy_first_num_dev(double* array_in, double* array_out){
	array_out[0] = array_in[0];
}

__global__ void copy_dev(double* array_in, double* array_out, int size){
	int tid = threadIdx.x +blockIdx.x * blockDim.x;
	if (tid < size){
		array_out[tid] = array_in[tid];
	}
}

__global__ void reduce_last_in_one_thread(double* array_in, double* array_out, int size){
//	int tid = threadIdx.x +blockIdx.x * blockDim.x;
	double sum = 0;
	for(int i = 0; i<size; i++){
		sum += array_in[i];
	}
	array_out[0]=sum;
}

__global__ void display_dev(double* array_out){
    printf("?\n");
    printf("array_out[0] = %f\n", array_out[0]);
    printf("?\n");
}











/*
__global__ void kernel_norm_map(double* residual, int* taille_residual, double* std_map, int indice_x, int indice_y, int indice_v)
{
  int index_x = blockIdx.x*BLOCK_SIZE_L2_X +threadIdx.x;
  int index_y = blockIdx.y*BLOCK_SIZE_L2_Y +threadIdx.y;
  if(index_x<indice_x && index_y<indice_y && std_map[index_y*indice_x + index_x]>0.)
  {

	int sum = thrust::reduce(D.begin(), D.end())

  }
}
*/

/*
__global__ void kernel_f(double* f, double* residual, int* taille_residual, double* std_map, int indice_x, int indice_y, int indice_v)
{
  int index_x = blockIdx.x*BLOCK_SIZE_L2_X +threadIdx.x;
  int index_y = blockIdx.y*BLOCK_SIZE_L2_Y +threadIdx.y;

  if(index_x<indice_x && index_y<indice_y && std_map[index_y*indice_x + index_x]>0.)
  {
//	printf("index_x = %d , index_y = %d \n", index_x, index_y);
    double S=0.;
		for(int k = 0; k<indice_v; k++){
			S+= pow(residual[index_x*indice_y*indice_v+index_y*indice_v+k],2);
		}
	printf("S*... = %f\n",0.5*S/pow(std_map[index_y*indice_x + index_x],2.));

	f[0] += 0.5*S/pow(std_map[index_y*indice_x + index_x],2.);
	printf("std_map[%d] = %f\n",index_y*indice_x + index_x,std_map[index_y*indice_x + index_x]);
  }
//	printf("f = %f\n",f[0]);
}
*/
#endif
