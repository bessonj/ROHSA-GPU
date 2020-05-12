#ifndef _GRADIENT_KERNEL_H_
#define _GRADIENT_KERNEL_H_



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

#endif


