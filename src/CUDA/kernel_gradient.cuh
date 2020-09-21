#ifndef _GRADIENT_KERNEL_H_
#define _GRADIENT_KERNEL_H_

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
//	printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);

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
			double par1 = double(i+1) - par1_a;

			buffer_0 += exp( -pow( par1 ,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
			buffer_1 += par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
			buffer_2 += par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow ) * residual[index_z*tr1*tr2+index_y*tr2+i]*parstd;
		}

//		printf("index_x = %d , index_y = %d , index_z = %d\n",index_x,index_y,index_z);
//		printf(" buffer_0 = %f \n", buffer_0);

//		printf(" buffer_0 = %f ,  buffer_1 = %f ,  buffer_2 = %f \n", buffer_0, buffer_1, buffer_2);// = %d , index_z = %d\n",index_x,index_y,index_z);

		__syncthreads();
		deriv[(3*index_x+0)*td1*td2+index_y*td2+index_z]=buffer_0;
		deriv[(3*index_x+1)*td1*td2+index_y*td2+index_z]=buffer_1;
		deriv[(3*index_x+2)*td1*td2+index_y*td2+index_z]=buffer_2;
	}

//  printf("tr2 = %d\n",tr2);


////  printf("ERROR 3 - ");

	//printf("dF_over_dB_dev[%d] = %f\n",index_z*t_3*t_2*t_1+index_y*t_3*t_2+index_x*t_3+2,dF_over_dB_dev[index_z*t_3*t_2*t_1+index_y*t_3*t_2+index_x*t_3+2]);
	__syncthreads();
}

/*
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for(k=0; k<indice_v; k++){
				for(l=0; l<M.n_gauss; l++){
				if(std_map[i][j]>0.){
//					coordinates[0] = l;
//					coordinates[1] = k;
//					coordinates[2] = i;
//					coordinates[3] = j;
					deriv[3*l+0][i][j]+=  dF_over_dB[k*t2*t1*t3+i*t2*t3+j*t3+3*l]*residual[j][i][k]/pow(std_map[i][j],2);
					deriv[3*l+1][i][j]+=  dF_over_dB[k*t2*t1*t3+i*t2*t3+j*t3+3*l+1]*residual[j][i][k]/pow(std_map[i][j],2);
					deriv[3*l+2][i][j]+=  dF_over_dB[k*t2*t1*t3+i*t2*t3+j*t3+3*l+2]*residual[j][i][k]/pow(std_map[i][j],2);
				}
			}
		}
	}
}
*/


#endif
