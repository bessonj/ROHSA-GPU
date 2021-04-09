#ifndef DEF_HYPERCUBE
#define DEF_HYPERCUBE

#include "matplotlibcpp.h"
#include "parameters.hpp"
#include <iostream>
#include <stdio.h>
#include <cmath>
//#include <math.h>
#include <string>
#include <fstream>
#include <valarray>
#include <CCfits/CCfits>
#include <vector>
#include <omp.h>
#include <iterator>

/**
 * @brief This class is about reading the fits file, transforming the data array and displaying the result in various ways.
 *
 *
 *
 *
 *  We can read a FITS or a DAT file, the result is then stored into a output file (binary or fits). The hypercube extracted is put into a larger hypercube of dimensions \f$ dim\_\nu \times 2^{n\_side} \times 2^{n\_side} \f$.
 * We can then rebuild the hypercube using the results, print the mean values of the gaussian parameters or print the gaussian parameters map. 
 * 
 *
 *
 * We read and write the FITS file using the library CCFits based on CFitsio.
 *
 *
 *
 * 
 *
 */
template<typename T>
class hypercube
{
	public:

	hypercube();
	hypercube(int dim_x, int dim_y, int dim_v); //dummy constructor for initialization of an hypercube object
	hypercube(parameters<T> &M);
	hypercube(parameters<T> &M,int indice_debut, int indice_fin); // assuming whole_data_in_cube = false (faster and better provided the dimensions are close)
	hypercube(parameters<T> &M,int indice_debut, int indice_fin, bool one_level);
	hypercube(parameters<T> &M, int indice_debut, int indice_fin, int pos_x, int pos_y, int size_side_square);
	hypercube(parameters<T> &M, int indice_debut, int indice_fin, int pos_x, int pos_y, int size_side_square, bool select_spectral_range, bool get_a_square_of_a_given_size, bool larger_power_of_two);
	
	int writeImage();
	int save_noise_map_in_fits(parameters<T> &M, std::vector <std::vector<T>>& noise_map);
	int save_grid_in_fits(parameters<T> &M, std::vector <std::vector<std::vector<T>>>& grid);
	void get_noise_map_from_fits(parameters<T> &M, std::vector <std::vector<T>>& noise_data, std::vector <std::vector<T>>& noise_cube);
	void get_noise_map_from_DHIGLS(parameters<T> &M, std::vector <std::vector<T>>& noise_data, std::vector <std::vector<T>>& noise_cube);
	void get_noise_map_from_GHIGLS(parameters<T> &M, std::vector <std::vector<T>>& noise_map);
	void reshape_noise_up(std::vector<std::vector<T>>& std_map_in, std::vector<std::vector<T>>& std_map_out);
	
	void reshape_noise_up_data(std::vector<std::vector<T>>& std_data_raw, std::vector<std::vector<T>>& std_data_out, int dim1, int dim0);//!<Given std_data_raw (whole fits noise map), it returns std_data (adapted to the spatial positions of this->data). dim1, dim0 = spatial dimensions of whole fits noise map.
	void reshape_noise_up_cube(std::vector<std::vector<T>>& std_data, std::vector<std::vector<T>>& std_cube_out); //!<When given std_data (noise map related to the spatial position of this->data), it returns std_cube (the noise map used through multiresolution)

	void mean_parameters(std::vector<std::vector<std::vector<T>>> &params, int num_gauss);

	T model_function(int x, T a, T m, T s);

	int dim2nside(); //obtenir les dimensions 2^n
	void multiresolution(int nside); 
	int get_binary_from_fits();

	std::vector <std::vector<std::vector<T>>> get_array_from_fits(parameters<T> &M);
//	void get_array_from_fits(parameters<T> &M);
	void get_vector_from_binary(std::vector<std::vector<std::vector<T>>> &z);
	std::vector<int> get_dim_data();
	std::vector<int> get_dim_cube();
	int get_nside() const;
	std::vector<std::vector<std::vector<T>>> use_dat_file(parameters<T> &M);
	std::vector<std::vector<std::vector<T>>> reshape_up();
	std::vector<std::vector<std::vector<T>>> reshape_up(int borne_inf, int borne_sup);
	std::vector<std::vector<std::vector<T>>> reshape_up_for_last_level(int borne_inf, int borne_sup);

	void write_into_binary(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params);
	void get_from_file(std::vector<std::vector<std::vector<T>>> &file_out, int dim_0, int dim_1, int dim_2);
	void write_in_file(std::vector<std::vector<std::vector<T>>> &file_in);
	void write_in_file(std::vector<std::vector<std::vector<T>>> &file_in, std::string filename);

	void write_vector_to_file(const std::vector<T>& myVector, std::string filename);
	std::vector<T> read_vector_from_file(std::string filename);

	void save_result(std::vector<std::vector<std::vector<T>>>&, parameters<T>&);
	void save_result_multires(std::vector<std::vector<std::vector<T>>>&, parameters<T>&, int);


	int indice_debut, indice_fin; //!< Only some spectral ranges of the hypercube are exploitable. We cut the hypercube, this will introduce an offset on the result values.
	std::vector<std::vector<std::vector<T>>> cube; //!< The hypercube "data" is centered into a larger hypercube "cube" for the purpose of multiresolution this hypercube is useful and its spatial dimensions are \f$  2^{n\_side} \times 2^{n\_side} \f$. Where \f$n\_side\f$ is computed by dim2nside(), it is the smallest power of 2 greater than the spatial dimensions. 
//	std::vector<std::vector<std::vector<T>>> data_not_reshaped;
	std::vector<std::vector<std::vector<T>>> data; //!< Hypercube array extracted from the fits file, its spectral range is changed according to indice_debut and indice_fin. 
	std::vector <std::vector<T>> noise_map;

	int dim_data[3];
	int dim_cube[3];
	std::vector<int> dim_data_v;
	std::vector<int> dim_cube_v;
	int nside;

	std::string filename; 

	bool select_spectral_range_option;
	bool get_a_square_of_a_given_size_option;
	bool larger_power_of_two_option;

	int size_side_square_option;
	int position_x;
	int position_y;
	bool position_given;
};



namespace plt = matplotlibcpp;
//using namespace std;
using namespace CCfits;

template<typename T>
hypercube<T>::hypercube(parameters<T> &M, int indice_debut, int indice_fin, int pos_x, int pos_y, int size_side_square, bool select_spectral_range, bool get_a_square_of_a_given_size, bool larger_power_of_two)
{
	this->position_given = true;
	this->select_spectral_range_option = select_spectral_range;
	this->get_a_square_of_a_given_size_option = get_a_square_of_a_given_size;
	this->larger_power_of_two_option = larger_power_of_two;
	this->size_side_square_option = size_side_square;

	this->indice_debut= indice_debut;
	this->indice_fin = indice_fin;

	this->position_x = pos_x;
	this->position_y = pos_y;

	//First we get this->data which is the data array
	if(M.input_format_fits){
//		std::cout<<"indice_debut-indice_fin+1 = "<<indice_fin-indice_debut+1<<std::endl;
//		exit(0);

		this->data = get_array_from_fits(M);
		if(larger_power_of_two){
			this->nside = dim2nside()-1;
		}else{
			this->nside = dim2nside();
		}

		std::cout<<"larger_power_of_two = "<<larger_power_of_two<<std::endl;

		std::cout<<"this->data[0][0][0] = "<<this->data[0][0][0]<<std::endl;
		std::cout<<"this->data[0][0][1] = "<<this->data[0][0][1]<<std::endl;
		std::cout<<"this->data[0][0][2] = "<<this->data[0][0][2]<<std::endl;
		std::cout<<"this->data[0][0][3] = "<<this->data[0][0][3]<<std::endl;

		std::cout<<" "<<std::endl;
		std::cout<<" "<<std::endl;

//		this->dim_cube[2] = dim_data[2];
		this->dim_cube[2] = indice_fin-indice_debut+1;
		int shift = this->size_side_square_option/2;

		std::cout<<"shift = "<<shift<<std::endl;
		std::cout<<"pos_x = "<<pos_x<<std::endl;
		std::cout<<"pos_y = "<<pos_y<<std::endl;
		std::cout<<"size_side_square = "<<size_side_square<<std::endl;
		std::cout<<" "<<std::endl;
		std::cout<<" "<<std::endl;
		std::cout<<" "<<std::endl;
		std::cout<<" "<<std::endl;
		std::cout<<"this->data["<<0+pos_x-shift<<"]["<<0+pos_y-shift<<"][0] ="<<this->data[0+pos_x-shift][0+pos_y-shift][0]<<std::endl;


		std::cout<<" "<<std::endl;
		std::cout<<" "<<std::endl;
		std::cout<<" "<<std::endl;

		std::vector<std::vector<std::vector<T>>> data_reshaped_local(size_side_square, std::vector<std::vector<T>>(size_side_square,std::vector<T>(dim_cube[2],0.)));
/*
		for(int i=0; i< this->size_side_square_option; i++)
		{
			for(int j=0; j< this->size_side_square_option; j++)
			{
				for(int k= 0; k< dim_data[2]; k++)
				{
					data_reshaped_local[i][j][k]= this->data[i+pos_x-shift][j+pos_y-shift][k];
				}
			}
		}
*/
//AJOUT :
		this->dim_data[2] = this->dim_cube[2];
		for(int i=0; i< this->size_side_square_option; i++)
		{
			for(int j=0; j< this->size_side_square_option; j++)
			{
				for(int k= 0; k< indice_fin-indice_debut+1; k++)
				{
					data_reshaped_local[i][j][k]= this->data[i+pos_x-shift][j+pos_y-shift][k+indice_debut];
				}
			}
		}

		std::cout<<"data_reshaped_local[0][0][0] = "<<data_reshaped_local[0][0][0]<<std::endl;
		std::cout<<"data_reshaped_local[0][0][1] = "<<data_reshaped_local[0][0][1]<<std::endl;
		std::cout<<"data_reshaped_local[0][0][2] = "<<data_reshaped_local[0][0][2]<<std::endl;
		std::cout<<"data_reshaped_local[0][0][3] = "<<data_reshaped_local[0][0][3]<<std::endl;

		this->data = data_reshaped_local;
		data_reshaped_local.clear();// = std::vector<std::vector<std::vector<T>>>();

		this->dim_data[0] = this->size_side_square_option;
		this->dim_data[1] = this->size_side_square_option;
		this->nside = dim2nside();

		std::cout<<"this->data[0][0][0] = "<<this->data[0][0][0]<<std::endl;
		std::cout<<"this->data[0][0][1] = "<<this->data[0][0][1]<<std::endl;
		std::cout<<"this->data[0][0][2] = "<<this->data[0][0][2]<<std::endl;
		std::cout<<"this->data[0][0][3] = "<<this->data[0][0][3]<<std::endl;
		std::cout<<"this->nside = "<<this->nside<<std::endl;

			this->dim_cube[0] =pow(2.0,this->nside);
			this->dim_cube[1] =pow(2.0,this->nside);
			std::cout<<"dim_cube[0] = "<<this->dim_cube[0]<<std::endl;
			std::cout<<"dim_cube[1] = "<<this->dim_cube[1]<<std::endl;

			int offset_x = abs((-dim_cube[0]+dim_data[0])/2);
			int offset_y = abs((-dim_cube[1]+dim_data[1])/2);

			std::cout<<"offset_x = "<<offset_x<<std::endl;
			std::cout<<"offset_y = "<<offset_y<<std::endl;

			std::vector<std::vector<std::vector<T>>> cube_reshaped_local(this->dim_cube[0], std::vector<std::vector<T>>(this->dim_cube[1],std::vector<T>(this->dim_cube[2],0.)));
			for(int i=0; i< this->dim_cube[0]; i++)
			{
				for(int j=0; j< this->dim_cube[1]; j++)
				{
					for(int k=0; k<this->dim_cube[2]; k++)
					{
					cube_reshaped_local[i][j][k]= this->data[i+offset_x][j+offset_y][k];
					}
				}
			}
			std::cout<<"END 1"<<std::endl;

			this->cube = cube_reshaped_local;
			cube_reshaped_local.clear();// = std::vector<std::vector<std::vector<T>>>();

			std::cout<<"END 2"<<std::endl;
	/*
		if(select_spectral_range){
			dim_cube[0] =pow(2.0,this->nside);
			dim_cube[1] =pow(2.0,this->nside);
			dim_cube[2] =indice_fin-indice_debut+1;

			int offset_x = (-dim_cube[0]+dim_data[0])/2;
			int offset_y = (-dim_cube[1]+dim_data[1])/2;
			std::vector<std::vector<std::vector<T>>> data_reshaped_local(dim_data[0], std::vector<std::vector<T>>(dim_data[1],std::vector<T>(indice_fin-indice_debut+1,0.)));
			for(int i=0; i< dim_data[0]; i++)
			{
				for(int j=0; j< dim_data[1]; j++)
				{
					for(int k= indice_debut; k<= indice_fin; k++)
					{
					data_reshaped_local[i][j][k-indice_debut]= this->data[i][j][k];
					}
				}
			}
			this->data = data_reshaped_local;
			dim_data[2] =indice_fin-indice_debut+1;

			std::vector<std::vector<std::vector<T>>> cube_reshaped_local(dim_cube[0], std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(indice_fin-indice_debut+1,0.)));
			for(int i=0; i< dim_cube[0]; i++)
			{
				for(int j=0; j< dim_cube[1]; j++)
				{
					for(int k=0; k<indice_fin-indice_debut+1; k++)
					{
					cube_reshaped_local[i][j][k]= this->data[i-offset_x][j-offset_y][k];
					}
				}
			}
			this->cube = cube_reshaped_local;
		}else if(select_spectral_range && larger_power_of_two){
			dim_cube[0] =pow(2.0,this->nside);
			dim_cube[1] =pow(2.0,this->nside);
			dim_cube[2] =indice_fin-indice_debut+1;
			std::vector<std::vector<std::vector<T>>> data_reshaped_local(dim_data[0], std::vector<std::vector<T>>(dim_data[1],std::vector<T>(indice_fin-indice_debut+1,0.)));
			for(int i=0; i< dim_data[0]; i++)
			{
				for(int j=0; j< dim_data[1]; j++)
				{
					for(int k= indice_debut; k<= indice_fin; k++)
					{
					data_reshaped_local[i][j][k-indice_debut]= this->data[i][j][k];
					}
				}
			}
			this->data = data_reshaped_local;
			dim_data[2] =indice_fin-indice_debut+1;

			int offset_x = (-dim_cube[0]+dim_data[0])/2;
			int offset_y = (-dim_cube[1]+dim_data[1])/2;
			std::vector<std::vector<std::vector<T>>> cube_reshaped_local(dim_cube[0], std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(indice_fin-indice_debut+1,0.)));
			for(int i=0; i< dim_cube[0]; i++)
			{
				for(int j=0; j< dim_cube[1]; j++)
				{
					for(int k=0; k<indice_fin-indice_debut+1; k++)
					{
					cube_reshaped_local[i][j][k]= this->data[i+offset_x][j+offset_y][k];
					}
				}
			}
			this->cube = cube_reshaped_local;
			this->data = cube_reshaped_local;
		}else if(larger_power_of_two){
			dim_cube[0] =pow(2.0,this->nside);
			dim_cube[1] =pow(2.0,this->nside);
			dim_cube[2] =dim_data[2];

			int offset_x = (-dim_cube[0]+dim_data[0])/2;
			int offset_y = (-dim_cube[1]+dim_data[1])/2;
			std::vector<std::vector<std::vector<T>>> cube_reshaped_local(dim_cube[0], std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(dim_cube[2],0.)));
			for(int i=0; i< dim_cube[0]; i++)
			{
				for(int j=0; j< dim_cube[1]; j++)
				{
					for(int k=0; k<dim_cube[2]; k++)
					{
					cube_reshaped_local[i][j][k]= this->data[i+offset_x][j+offset_y][k];
					}
				}
			}
			this->cube = cube_reshaped_local;
			this->data = cube_reshaped_local;
		}else if(select_spectral_range && get_a_square_of_a_given_size){
			dim_cube[2] =indice_fin-indice_debut+1;
			int shift = size_side_square/2;
			std::vector<std::vector<std::vector<T>>> data_reshaped_local(size_side_square, std::vector<std::vector<T>>(size_side_square,std::vector<T>(dim_cube[2],0.)));
			for(int i=0; i< size_side_square; i++)
			{
				for(int j=0; j< size_side_square; j++)
				{
					for(int k= indice_debut; k<= indice_fin; k++)
					{
						data_reshaped_local[i][j][k-indice_debut]= this->data[i+pos_x-shift][j+pos_y-shift][k];
					}
				}
			}

			this->data = data_reshaped_local;
			this->nside = dim2nside();

			dim_cube[0] =pow(2.0,this->nside);
			dim_cube[1] =pow(2.0,this->nside);
			dim_data[2] =indice_fin-indice_debut+1;

			int offset_x = (-dim_cube[0]+dim_data[0])/2;
			int offset_y = (-dim_cube[1]+dim_data[1])/2;
			std::vector<std::vector<std::vector<T>>> cube_reshaped_local(dim_cube[0], std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(indice_fin-indice_debut+1,0.)));
			for(int i=0; i< dim_cube[0]; i++)
			{
				for(int j=0; j< dim_cube[1]; j++)
				{
					for(int k=0; k<indice_fin-indice_debut+1; k++)
					{
					cube_reshaped_local[i][j][k]= this->data[i-offset_x][j-offset_y][k];
					}
				}
			}
			this->cube = cube_reshaped_local;
		}else if(get_a_square_of_a_given_size){
			this->dim_cube[2] = dim_data[2];
			int shift = size_side_square/2;
			std::vector<std::vector<std::vector<T>>> data_reshaped_local(size_side_square, std::vector<std::vector<T>>(size_side_square,std::vector<T>(dim_cube[2],0.)));
			for(int i=0; i< size_side_square; i++)
			{
				for(int j=0; j< size_side_square; j++)
				{
					for(int k= 0; k< dim_data[2]; k++)
					{
						data_reshaped_local[i][j][k]= this->data[i+pos_x-shift][j+pos_y-shift][k];
					}
				}
			}

			this->data = data_reshaped_local;
			this->dim_data[0] = size_side_square;
			this->dim_data[1] = size_side_square;
			this->nside = dim2nside();

	std::cout<<"this->data[0][0][0] = "<<this->data[0][0][0]<<std::endl;
	std::cout<<"this->data[0][0][1] = "<<this->data[0][0][1]<<std::endl;
	std::cout<<"this->data[0][0][2] = "<<this->data[0][0][2]<<std::endl;
	std::cout<<"this->nside = "<<this->nside<<std::endl;

			this->dim_cube[0] =pow(2.0,this->nside);
			this->dim_cube[1] =pow(2.0,this->nside);
	std::cout<<"dim_cube[0] = "<<this->dim_cube[0]<<std::endl;
	std::cout<<"dim_cube[1] = "<<this->dim_cube[1]<<std::endl;

			int offset_x = abs((-dim_cube[0]+dim_data[0])/2);
			int offset_y = abs((-dim_cube[1]+dim_data[1])/2);
	std::cout<<"offset_x = "<<offset_x<<std::endl;
	std::cout<<"offset_y = "<<offset_y<<std::endl;
			std::vector<std::vector<std::vector<T>>> cube_reshaped_local(this->dim_cube[0], std::vector<std::vector<T>>(this->dim_cube[1],std::vector<T>(this->dim_cube[2],0.)));
			for(int i=0; i< this->dim_cube[0]; i++)
			{
				for(int j=0; j< this->dim_cube[1]; j++)
				{
					for(int k=0; k<this->dim_cube[2]; k++)
					{
					cube_reshaped_local[i][j][k]= this->data[i+offset_x][j+offset_y][k];
					}
				}
			}
	std::cout<<"END 1"<<std::endl;

			this->cube = cube_reshaped_local;
	std::cout<<"END 2"<<std::endl;
		}else{
			dim_cube[0] =pow(2.0,this->nside);
			dim_cube[1] =pow(2.0,this->nside);
			dim_cube[2] =dim_data[2];

			int offset_x = (-dim_cube[0]+dim_data[0])/2;
			int offset_y = (-dim_cube[1]+dim_data[1])/2;


			std::vector<std::vector<std::vector<T>>> cube_reshaped_local(dim_cube[0], std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(dim_cube[2],0.)));
			for(int i=0; i< dim_cube[0]; i++)
			{
				for(int j=0; j< dim_cube[1]; j++)
				{
					for(int k=0; k<dim_cube[2]; k++)
					{
					cube_reshaped_local[i][j][k]= this->data[i-offset_x][j-offset_y][k];
					}
				}
			}
			this->cube = cube_reshaped_local;
		}
		*/
	}else{
		if(larger_power_of_two){
			this->nside = dim2nside()-1;
		}else{
			this->nside = dim2nside();
		}
		this->data = use_dat_file(M);
		this->dim_cube[0] = this->dim_data[0];
		this->dim_cube[1] = this->dim_data[1];
		this->dim_cube[2] = this->dim_data[2];
		this->cube = this->data;

	}
	std::cout<<"dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"dim_data[2] = "<<dim_data[2]<<std::endl;

	std::cout<<"dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"dim_cube[2] = "<<dim_cube[2]<<std::endl;

	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	for(int i = 0; i<this->dim_data[2]; i++) 	std::cout<<"this->data[1023][1023]["<<i<<"] = "<<this->data[1023][1023][i]<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;
	for(int i = 0; i<this->dim_data[2]; i++) 	std::cout<<"this->data[0][0]["<<i<<"] = "<<this->data[0][0][i]<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	printf("this->data[0][0][0] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][1] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][2] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][3] = %.26f\n",this->data[0][0][0]);
	std::cout<<" "<<std::endl;
	printf("this->data[0][0][56] = %.26f\n",this->data[0][0][56]);
	printf("this->data[0][0][57] = %.26f\n",this->data[0][0][57]);
	printf("this->data[0][0][58] = %.26f\n",this->data[0][0][58]);
	printf("this->data[0][0][59] = %.26f\n",this->data[0][0][59]);

	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	printf("this->cube[0][0][0] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][1] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][2] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][3] = %.26f\n",this->cube[0][0][0]);
	std::cout<<" "<<std::endl;
	printf("this->cube[0][0][56] = %.26f\n",this->cube[0][0][56]);
	printf("this->cube[0][0][57] = %.26f\n",this->cube[0][0][57]);
	printf("this->cube[0][0][58] = %.26f\n",this->cube[0][0][58]);
	printf("this->cube[0][0][59] = %.26f\n",this->cube[0][0][59]);

}

template<typename T>
hypercube<T>::hypercube(parameters<T> &M, int indice_debut, int indice_fin, bool last_level_power_of_two)
{
	this->indice_debut= indice_debut;
	this->indice_fin = indice_fin;
	if(M.file_type_fits){
		this->data = get_array_from_fits(M);
	}
	if(M.file_type_dat){
		this->data = use_dat_file(M);
	}
	if(last_level_power_of_two){
		this->nside = dim2nside()-1;
		std::cout<<"this->nside = "<<this->nside<<std::endl;
	}else{
		this->nside = dim2nside();
	}
	
	dim_cube[0] =pow(2.0,nside);
	dim_cube[1] =pow(2.0,nside);
	if(M.file_type_fits){
		dim_cube[2] = indice_fin-indice_debut+1;
	}
	else{
		dim_cube[2] = dim_data[2];
	}

	std::cout<<"BEFORE SCALING"<<std::endl;
	std::cout<<"dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"dim_data[2] = "<<dim_data[2]<<std::endl;

	std::cout<<"dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"dim_cube[2] = "<<dim_cube[2]<<std::endl;

	if(M.file_type_fits){
		if(last_level_power_of_two){
			std::vector<std::vector<std::vector<T>>> data_reshaped_local(dim_cube[0], std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(dim_cube[2],0.)));
			data_reshaped_local = reshape_up_for_last_level(indice_debut, indice_fin);
			this->dim_data[0]=this->dim_cube[0];
			this->dim_data[1]=this->dim_cube[1];
			dim_cube[0] =pow(2.0,nside-1);
			dim_cube[1] =pow(2.0,nside-1);
			cube = reshape_up_for_last_level(indice_debut, indice_fin);
			this->data = data_reshaped_local;
		}else{
			cube = reshape_up(indice_debut, indice_fin);
			std::vector<std::vector<std::vector<T>>> data_reshaped_local(dim_data[0], std::vector<std::vector<T>>(dim_data[1],std::vector<T>(dim_cube[2],0.)));

			for(int i=0; i< this->dim_data[0]; i++)
				{
					for(int j=0; j< this->dim_data[1]; j++)
					{
						for(int k= indice_debut; k<= indice_fin; k++)
						{
							data_reshaped_local[i][j][k-indice_debut]= this->data[i][j][k];
						}
					}
				}
			this->data = data_reshaped_local;
		}
	}
	else{
		cube = data;
	}
	std::cout<<"AFTER SCALING"<<std::endl;
	std::cout<<"dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"dim_data[2] = "<<dim_data[2]<<std::endl;

	std::cout<<"dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"dim_cube[2] = "<<dim_cube[2]<<std::endl;

	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	printf("this->data[0][0][0] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][1] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][2] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][3] = %.26f\n",this->data[0][0][0]);
	std::cout<<" "<<std::endl;
	printf("this->data[0][0][56] = %.26f\n",this->data[0][0][56]);
	printf("this->data[0][0][57] = %.26f\n",this->data[0][0][57]);
	printf("this->data[0][0][58] = %.26f\n",this->data[0][0][58]);
	printf("this->data[0][0][59] = %.26f\n",this->data[0][0][59]);

	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	printf("this->cube[0][0][0] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][1] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][2] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][3] = %.26f\n",this->cube[0][0][0]);
	std::cout<<" "<<std::endl;
	printf("this->cube[0][0][56] = %.26f\n",this->cube[0][0][56]);
	printf("this->cube[0][0][57] = %.26f\n",this->cube[0][0][57]);
	printf("this->cube[0][0][58] = %.26f\n",this->cube[0][0][58]);
	printf("this->cube[0][0][59] = %.26f\n",this->cube[0][0][59]);

}

template<typename T>
hypercube<T>::hypercube(parameters<T> &M, int indice_debut, int indice_fin)
{
	this->indice_debut= indice_debut;
	this->indice_fin = indice_fin;
	if(M.input_format_fits){
		this->data = get_array_from_fits(M);
	}else{
		this->data = use_dat_file(M);
	}
	this->nside = dim2nside();
	std::cout<<"nside = "<<nside<<std::endl;
	
	dim_cube[0] =pow(2,nside);
	dim_cube[1] =pow(2,nside);
	if(M.input_format_fits){
		dim_cube[2] = indice_fin-indice_debut+1;
	}
	else{
		dim_cube[2] = dim_data[2];
	}

	std::cout<<"dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"dim_data[2] = "<<dim_data[2]<<std::endl;

	std::cout<<"dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"dim_cube[2] = "<<dim_cube[2]<<std::endl;

	if(M.input_format_fits){
		cube = reshape_up(indice_debut, indice_fin);
		std::vector<std::vector<std::vector<T>>> data_reshaped_local(dim_data[0], std::vector<std::vector<T>>(dim_data[1],std::vector<T>(dim_cube[2],0.)));
		for(int i=0; i< this->dim_data[0]; i++)
			{
				for(int j=0; j< this->dim_data[1]; j++)
				{
					for(int k= indice_debut; k<= indice_fin; k++)
					{
						data_reshaped_local[i][j][k-indice_debut]= this->data[i][j][k];
					}
				}
			}
		this->data = data_reshaped_local;
	} else{
		cube = data;
	}
	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	printf("this->data[0][0][0] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][1] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][2] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][3] = %.26f\n",this->data[0][0][0]);
	std::cout<<" "<<std::endl;
	printf("this->data[0][0][56] = %.26f\n",this->data[0][0][56]);
	printf("this->data[0][0][57] = %.26f\n",this->data[0][0][57]);
	printf("this->data[0][0][58] = %.26f\n",this->data[0][0][58]);
	printf("this->data[0][0][59] = %.26f\n",this->data[0][0][59]);

	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	printf("this->cube[0][0][0] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][1] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][2] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][3] = %.26f\n",this->cube[0][0][0]);
	std::cout<<" "<<std::endl;
	printf("this->cube[0][0][56] = %.26f\n",this->cube[0][0][56]);
	printf("this->cube[0][0][57] = %.26f\n",this->cube[0][0][57]);
	printf("this->cube[0][0][58] = %.26f\n",this->cube[0][0][58]);
	printf("this->cube[0][0][59] = %.26f\n",this->cube[0][0][59]);

}

//constructor for square centered on a pixel position
//for fits
template<typename T>
hypercube<T>::hypercube(parameters<T> &M, int indice_debut, int indice_fin, int pos_x, int pos_y, int size_side_square)
{
	this->indice_debut= indice_debut;
	this->indice_fin = indice_fin;
	if(M.input_format_fits){
		this->data = get_array_from_fits(M);
	}else{
		this->data = use_dat_file(M);
	}
	this->nside = ceil(log(size_side_square)/log(2));

	std::cout<<"nside = "<<nside<<std::endl;
	
	this->dim_cube[0] = size_side_square;
	this->dim_cube[1] = size_side_square;
	if(M.input_format_fits){
		dim_cube[2] = indice_fin-indice_debut+1;
	}
	else{
		dim_cube[2] = dim_data[2];
	}

	std::cout<<"dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"dim_data[2] = "<<dim_data[2]<<std::endl;

	std::cout<<"dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"dim_cube[2] = "<<dim_cube[2]<<std::endl;

	int shift = size_side_square/2;

	this->dim_data[0] = size_side_square;
	this->dim_data[1] = size_side_square;
	this->dim_data[2] = this->dim_cube[2];

	if(M.input_format_fits){
		std::vector<std::vector<std::vector<T>>> data_reshaped_local(size_side_square, std::vector<std::vector<T>>(size_side_square,std::vector<T>(dim_cube[2],0.)));
		for(int i=0; i< size_side_square; i++)
			{
				for(int j=0; j< size_side_square; j++)
				{
					for(int k= indice_debut; k<= indice_fin; k++)
					{
						data_reshaped_local[i][j][k-indice_debut]= this->data[i+pos_x-shift][j+pos_y-shift][k];
					}
				}
			}
		this->data = data_reshaped_local;
		cube = reshape_up(indice_debut, indice_fin);
	} else{
		cube = data;
	}
	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	printf("this->data[0][0][0] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][1] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][2] = %.26f\n",this->data[0][0][0]);
	printf("this->data[0][0][3] = %.26f\n",this->data[0][0][0]);
	std::cout<<" "<<std::endl;
	printf("this->data[0][0][56] = %.26f\n",this->data[0][0][56]);
	printf("this->data[0][0][57] = %.26f\n",this->data[0][0][57]);
	printf("this->data[0][0][58] = %.26f\n",this->data[0][0][58]);
	printf("this->data[0][0][59] = %.26f\n",this->data[0][0][59]);

	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;

	printf("this->cube[0][0][0] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][1] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][2] = %.26f\n",this->cube[0][0][0]);
	printf("this->cube[0][0][3] = %.26f\n",this->cube[0][0][0]);
	std::cout<<" "<<std::endl;
	printf("this->cube[0][0][56] = %.26f\n",this->cube[0][0][56]);
	printf("this->cube[0][0][57] = %.26f\n",this->cube[0][0][57]);
	printf("this->cube[0][0][58] = %.26f\n",this->cube[0][0][58]);
	printf("this->cube[0][0][59] = %.26f\n",this->cube[0][0][59]);


}

template<typename T>
hypercube<T>::hypercube(parameters<T> &M)
{
	if(M.input_format_fits){
		this->data = get_array_from_fits(M);
//		get_dimensions_from_fits();
//		get_binary_from_fits(); // WARNING 
//		get_vector_from_binary(this->data);
		this->nside = dim2nside()-1;
	}else{
		this->data = use_dat_file(M);
		this->nside = dim2nside();
	}
//	std::cout << "	DEBUG " << std::endl;
//	std::cout<<dim_data[2]<<std::endl;

	
	dim_cube[0] =pow(2.0,nside);
	dim_cube[1] =pow(2.0,nside);
	dim_cube[2] = dim_data[2];
	
	std::cout<<"dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"dim_cube[2] = "<<dim_cube[2]<<std::endl;

	cube = reshape_up();
/*
	for(int k(0); k<dim_cube[2]; k++)
	{
		for(int j(0); j<dim_cube[1]; j++)
		{
			for(int i(0); i< dim_cube[0]; i++)
			{
				std::cout<<"cube["<<i<<"]["<<j<<"]["<<k<<"] = "<<cube[i][j][k]<<std::endl;
			}
		}
	}
*/
}

template<typename T>
hypercube<T>::hypercube(int dim_x, int dim_y, int dim_v) //dummy constructor for initialization of an hypercube object
{
	std::vector<std::vector<std::vector<T>>> data_reshaped_local(dim_v, std::vector<std::vector<T>>(dim_y,std::vector<T>(dim_x,0.)));
	this->data = data_reshaped_local;
}

template<typename T>
hypercube<T>::hypercube() //dummy constructor for initialization of an hypercube object
{

}

template<typename T>
std::vector<std::vector<std::vector<T>>> hypercube<T>::use_dat_file(parameters<T> &M)
{
	printf("DEBUG 1\n");
   	int x,y,z;
	T res;

	std::ifstream fichier(M.filename_dat);

	fichier >> z >> x >> y;

	this->dim_data[2]=z;
	this->dim_data[1]=y;
	this->dim_data[0]=x;

//	printf("dim_data[0] = %d\n",this->dim_data[0]);
//	printf("dim_data[1] = %d\n",this->dim_data[1]);
//	printf("dim_data[2] = %d\n",this->dim_data[2]);

	std::vector<std::vector<std::vector<T>>> data_(this->dim_data[0],std::vector<std::vector<T>>(this->dim_data[1],std::vector<T>(this->dim_data[2], 0.)));

	printf("DEBUG 2\n");
	int p = 0;
	while(!fichier.std::ios::eof())
	{
   		fichier >> z >> y >> x >> res;
		data_[x][y][z] = res;
//		printf("DEBUG %d\n",p);
//		p++;
   	}
	return data_;
}

template<typename T>
std::vector<int> hypercube<T>::get_dim_data()
{
	dim_data_v.vector::push_back(dim_data[0]);
	dim_data_v.vector::push_back(dim_data[1]);
	dim_data_v.vector::push_back(dim_data[2]);
	return dim_data_v;
}

template<typename T>
int hypercube<T>::get_nside() const
{
	return nside;
}

template<typename T>
std::vector<int> hypercube<T>::get_dim_cube()
{
	dim_cube_v.vector::push_back(dim_cube[0]);
	dim_cube_v.vector::push_back(dim_cube[1]);
	dim_cube_v.vector::push_back(dim_cube[2]);
	return dim_cube_v;
}

/*
Parse::~Parse()
{
//	faire des .clear() comme Cube.clear();
}
*/

// Compute nside value from \(dim_y\) and \(dim_x\) 
template<typename T>
int hypercube<T>::dim2nside()
{
	return std::max( 0, std::min(int(ceil( log(double(this->dim_data[0]))/log(2.))), int(ceil( log(double(this->dim_data[1]))/log(2.))))  ) ;  
}

//assuming the cube has been reshaped before through the python tool
template<typename T>
std::vector<std::vector<std::vector<T>>> hypercube<T>::reshape_up()
{
	std::vector<std::vector<std::vector<T>>> cube_(dim_cube[0],std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(dim_cube[2])));
	for(int i(0); i< dim_cube[0]; i++)
	{
		for(int j(0); j<dim_cube[1]; j++)
		{
			for(int k(0); k<dim_cube[2]; k++)
			{
				cube_[i][j][k]= data[i][j][k];
			}
		}
	}

	return cube_;
}


template<typename T>
std::vector<std::vector<std::vector<T>>> hypercube<T>::reshape_up_for_last_level(int borne_inf, int borne_sup)
{
	//compute the offset so that the data file lies in the center of a cube
	int offset_x = (-dim_cube[0]+dim_data[0])/2;
	int offset_y = (-dim_cube[1]+dim_data[1])/2;

	std::vector<std::vector<std::vector<T>>> cube_(dim_cube[0],std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(dim_cube[2],0.)));

	for(int i=offset_x; i< this->dim_cube[0]+offset_x; i++)
	{
		for(int j=offset_y; j<this->dim_cube[1]+offset_y; j++)
		{
			for(int k=0; k<this->dim_cube[2]; k++)
			{
				cube_[i-offset_x][j-offset_y][k]= this->data[i][j][borne_inf+k];
			}
		}
	}

	return cube_;
}


template<typename T>
std::vector<std::vector<std::vector<T>>> hypercube<T>::reshape_up(int borne_inf, int borne_sup)
{
	//compute the offset so that the data file lies in the center of a cube
	int offset_x = (dim_cube[0]-dim_data[0])/2;
	int offset_y = (dim_cube[1]-dim_data[1])/2;

	std::vector<std::vector<std::vector<T>>> cube_(dim_cube[0],std::vector<std::vector<T>>(dim_cube[1],std::vector<T>(dim_cube[2],0.)));

	for(int i=offset_x; i< this->dim_data[0]+offset_x; i++)
	{
		for(int j=offset_y; j<this->dim_data[1]+offset_y; j++)
		{
			for(int k=0; k<this->dim_cube[2]; k++)
			{
				cube_[i][j][k]= this->data[i-offset_x][j-offset_y][borne_inf+k];
			}
		}
	}
	return cube_;
}


// grid_params is written into a binary after the process in algo_rohsa
template<typename T>
void hypercube<T>::write_into_binary(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params){

	std::ofstream objetfichier;

 	objetfichier.open(M.fileout, std::ios::out | std::ofstream::binary); //on ouvre le fichier en ecriture
	if (objetfichier.bad()) //permet de tester si le fichier s'est ouvert sans probleme
		std::cout<<"ERREUR À L'OUVERTURE DU FICHIER RAW AVANT ÉCRITURE"<< std::endl;

	int n(sizeof(T) * grid_params.size());

	objetfichier.write((char*)&(grid_params)[0], n);

	objetfichier.close();
//	free(n);
}

template<typename T>
int hypercube<T>::get_binary_from_fits(){

	std::auto_ptr<FITS> pInfile(new FITS("./GHIGLS_DFN_Tb.fits",Read,true));

        PHDU& image = pInfile->pHDU();
	std::valarray<T> contents;
        image.readAllKeys();

        image.read(contents);

        // this doesn't print the data, just header info. The Late Show with Stephen Colbert

        // std::cout << image << std::endl;

        long ax1(image.axis(0));
        long ax2(image.axis(1));
        long ax3(image.axis(2));
	long ax4(image.axis(3));

	this->dim_data[0]=ax1;
	this->dim_data[1]=ax2;
	this->dim_data[2]=ax3;

//	std::vector <T> x;
//	std::vector <std::vector<T>> y;
//	std::vector <std::vector<std::vector<T>>> z;

	std::ofstream objetfichier;
 	objetfichier.open("./data.raw", std::ios::out | std::ofstream::binary ); //on ouvre le fichier en ecriture
	if (objetfichier.bad()) //permet de tester si le fichier s'est ouvert sans probleme
		std::cout<<"ERREUR À L'OUVERTURE DU FICHIER RAW AVANT ÉCRITURE"<< std::endl;

	int n(sizeof(T) * contents.size());

	objetfichier.write((char*)&contents[0], n);

	objetfichier.close();

	return n;
}

template<typename T>
void hypercube<T>::write_in_file(std::vector<std::vector<std::vector<T>>> &file_in, std::string filename){
	int dim_0 = file_in.size();
	int dim_1 = file_in[0].size();
	int dim_2 = file_in[0][0].size();

	std::vector<T> file_in_flat(dim_0*dim_1*dim_2,0.);

	for(int k=0; k<dim_0; k++)
	{
		for(int j=0; j<dim_1; j++)
		{
			for(int i=0; i<dim_2; i++)
			{
				file_in_flat[k*dim_2*dim_1+j*dim_2+i] = file_in[k][j][i];
			}
		}
	}

	write_vector_to_file(file_in_flat, filename);
}

template<typename T>
void hypercube<T>::write_in_file(std::vector<std::vector<std::vector<T>>> &file_in){
	int dim_0 = file_in.size();
	int dim_1 = file_in[0].size();
	int dim_2 = file_in[0][0].size();

	std::vector<T> file_in_flat(dim_0*dim_1*dim_2,0.);

	for(int k=0; k<dim_0; k++)
	{
		for(int j=0; j<dim_1; j++)
		{
			for(int i=0; i<dim_2; i++)
			{
				file_in_flat[k*dim_2*dim_1+j*dim_2+i] = file_in[k][j][i];
			}
		}
	}

	write_vector_to_file(file_in_flat, "./right_before_last_level.raw");
}

template<typename T>
void hypercube<T>::get_from_file(std::vector<std::vector<std::vector<T>>> &file_out, int dim_0, int dim_1, int dim_2){

   	int n = dim_0*dim_1*dim_2;
	std::ifstream is("./right_before_last_level.raw", std::ios::in | std::ifstream::binary);
	std::vector<T> file_out_flat = read_vector_from_file("./right_before_last_level.raw");

	for(int k=0; k<dim_0; k++)
	{
		for(int j=0; j<dim_1; j++)
		{
			for(int i=0; i<dim_2; i++)
			{
//				printf("file_out_flat[%d][%d][%d]=%f\n",k,j,i,file_out_flat[k*dim_2*dim_1+j*dim_2+i]);
				file_out[k][j][i] = file_out_flat[k*dim_2*dim_1+j*dim_2+i];
			}
		}
	}
}



template<typename T>
void hypercube<T>::write_vector_to_file(const std::vector<T>& myVector, std::string filename)
{
    std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
    std::ostream_iterator<T> osi{ofs," "};
    std::copy(myVector.begin(), myVector.end(), osi);
}

template<typename T>
std::vector<T> hypercube<T>::read_vector_from_file(std::string filename)
{
    std::vector<T> newVector{};
    std::ifstream ifs(filename, std::ios::in | std::ifstream::binary);
    std::istream_iterator<T> iter{ifs};
    std::istream_iterator<T> end{};
    std::copy(iter, end, std::back_inserter(newVector));
    return newVector;
}

template<typename T>
std::vector <std::vector<std::vector<T>>> hypercube<T>::get_array_from_fits(parameters<T> &M){
//	std::cout<<" On entre la boucle de lecture !!!!!! "<<std::endl;
//	std::cin.ignore();

	std::auto_ptr<FITS> pInfile(new FITS(M.filename_fits,Read,true));

    PHDU& image = pInfile->pHDU();
	std::valarray<T> contents;
    image.readAllKeys();

    image.read(contents);

        // this doesn't print the data, just header info.
	std::cout << image << std::endl;

    long ax1(image.axis(0));
    long ax2(image.axis(1));
    long ax3(image.axis(2));
//	long ax4(image.axis(3));

	this->dim_data[0]=ax1;
	this->dim_data[1]=ax2;
	this->dim_data[2]=ax3;

	std::cout<<"--> dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"--> dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"--> dim_data[2] = "<<dim_data[2]<<std::endl;

	std::cout<<"  "<<std::endl;
	std::cout<<"  "<<std::endl;

	std::cout<<"--> contents[0] = "<<contents[0]<<std::endl;
	std::cout<<"--> contents[1] = "<<contents[1]<<std::endl;
	std::cout<<"--> contents[2] = "<<contents[2]<<std::endl;
	std::cout<<"--> contents[3] = "<<contents[3]<<std::endl;
	std::cout<<"--> contents[4] = "<<contents[4]<<std::endl;
	std::cout<<"--> contents[5] = "<<contents[5]<<std::endl;

	std::cout<<"  "<<std::endl;
	std::cout<<"  "<<std::endl;

	std::cout<<"--> contents[0][0][0] = "<<contents[0]<<std::endl;
	std::cout<<"--> contents[1][1][0] = "<<contents[1*ax3*ax2+1*ax3]<<std::endl;
	std::cout<<"--> contents[1][2][0] = "<<contents[1*ax1*ax2+2*ax1]<<std::endl;
	std::cout<<"--> contents[2][1][0] = "<<contents[2*ax1*ax2+1*ax1]<<std::endl;
	std::cout<<"--> contents[2][2][0] = "<<contents[2*ax1*ax2+2*ax1]<<std::endl;
	std::cout<<"--> contents[3][3][0] = "<<contents[3*ax1*ax2+3*ax1]<<std::endl;
	std::cout<<"--> contents[4][4][0] = "<<contents[4*ax1*ax2+4*ax1]<<std::endl;
	std::cout<<"--> contents[5][5][0] = "<<contents[5*ax1*ax2+5*ax1]<<std::endl;
//	exit(0);
	std::vector <std::vector<std::vector<T>>> z_transpose(ax3, std::vector<std::vector<T>>(ax2, std::vector<T>(ax1,0.)));

//	std::cout<<" On entre la boucle de copie !!!!!! "<<std::endl;
//	std::cin.ignore();

	for(int k=0; k<ax3; k++)
	{
		for(int j=0; j<ax2; j++)
		{
			for(int i=0; i<ax1; i++)
			{
				z_transpose[k][j][i] = contents[k*ax1*ax2+j*ax1+i];
//				z[k][j][i] = contents[k*ax3*ax2+j*ax3+i];
			}
		}
	}
//	std::cout<<" Ligne suivante !!!!!! "<<std::endl;

//	std::cin.ignore();

	std::vector <std::vector<std::vector<T>>> z(ax1, std::vector<std::vector<T>>(ax2, std::vector<T>(ax3,0.)));
	for(int k=0; k<ax3; k++)
	{
		for(int j=0; j<ax2; j++)
		{
			for(int i=0; i<ax1; i++)
			{
				z[i][j][k] = z_transpose[k][j][i];
//				z[k][j][i] = contents[k*ax3*ax2+j*ax3+i];
			}
		}
	}
//	std::cout<<" Ligne suivante !!!!!! "<<std::endl;
	z_transpose.clear();//=std::vector<std::vector<std::vector<T>>>();


//	std::cout<<" On a fini ! "<<std::endl;

	return z;
}

template<typename T>
void hypercube<T>::get_noise_map_from_DHIGLS(parameters<T> &M, std::vector <std::vector<T>>& noise_data, std::vector <std::vector<T>>& noise_cube){

	std::cout << "M.filename_fits = " <<M.filename_fits<< std::endl;

	std::auto_ptr<FITS> pInfile(new FITS(M.filename_fits,Read,true));

	PHDU& image = pInfile->pHDU();

    ExtHDU& extension = pInfile->extension(3);

	std::valarray<T> contents;
    extension.readAllKeys();

	int dim[] = {0,0};

    extension.read(contents);

	std::cout << extension << std::endl;

    long ax1(extension.axis(0));
    long ax2(extension.axis(1));

	dim[0]=ax1;
	dim[1]=ax2;

	std::vector <std::vector<T>> z(dim[1], std::vector<T>(dim[0],0.));//, std::vector<T>(dim_data[2],0.)));

	int i__=0;
	for(int j=0; j<dim[1]; j++)
	{
		for(int k=0; k<dim[0]; k++)
		{
			z[j][k] = contents[dim[0]*j+k];
		}
	}


	std::vector <std::vector<T>> z_data(this->dim_data[0], std::vector<T>(this->dim_data[1],0.));//, std::vector<T>(dim_data[2],0.)));
	reshape_noise_up_data(z, z_data, dim[1], dim[0]);//!<Given std_data_raw (whole fits noise map), it returns std_data (adapted to the spatial positions of this->data). dim1, dim0 = spatial dimensions of whole fits noise map.
	z=std::vector<std::vector<T>>();

	noise_data = z_data;
	z_data=std::vector<std::vector<T>>();

	std::vector <std::vector<T>> z_cube(this->dim_cube[0], std::vector<T>(this->dim_cube[1],0.));//, std::vector<T>(dim_data[2],0.)));
	reshape_noise_up_cube(noise_data, z_cube); //!<When given std_data (noise map related to the spatial position of this->data), it returns std_cube (the noise map used through multiresolution)

	noise_cube = z_cube;
	z_cube=std::vector<std::vector<T>>();

	std::cout<<"noise_data[0][0] = "<<noise_data[0][0]<<std::endl;
	std::cout<<"noise_data[1][0] = "<<noise_data[1][0]<<std::endl;
	std::cout<<"noise_data[0][1] = "<<noise_data[0][1]<<std::endl;
	std::cout<<"noise_data[1][1] = "<<noise_data[1][1]<<std::endl;

	std::cout<<"noise_cube[0][0] = "<<noise_cube[0][0]<<std::endl;
	std::cout<<"noise_cube[1][0] = "<<noise_cube[1][0]<<std::endl;
	std::cout<<"noise_cube[0][1] = "<<noise_cube[0][1]<<std::endl;
	std::cout<<"noise_cube[1][1] = "<<noise_cube[1][1]<<std::endl;

}

template<typename T>
void hypercube<T>::get_noise_map_from_fits(parameters<T> &M, std::vector <std::vector<T>>& noise_data, std::vector <std::vector<T>>& noise_cube){
	std::auto_ptr<FITS> pInfile(new FITS(M.filename_noise,Read,true));

    PHDU& image = pInfile->pHDU();
	std::valarray<T> contents;
    image.readAllKeys();
	image.read(contents);

 	// this doesn't print the data, just header info.
    // std::cout << image << std::endl;
	long ax1(image.axis(0));
    long ax2(image.axis(1));
	printf("ax1 = %d , ax2 = %d\n", ax1, ax2);
	int dim[] = {0,0};
	dim[0]=ax1;
	dim[1]=ax2;

	std::vector <std::vector<T>> z(dim[0], std::vector<T>(dim[1],0.));//, std::vector<T>(dim_data[2],0.)));

	for(int j=0; j<dim[0]; j++)
	{
		for(int k=0; k<dim[1]; k++)
		{
			z[j][k] = contents[dim[1]*j+k];
		}
	}

	std::vector <std::vector<T>> z_data(this->dim_data[0], std::vector<T>(this->dim_data[1],0.));//, std::vector<T>(dim_data[2],0.)));
	reshape_noise_up_data(z, z_data, dim[1], dim[0]);//!<Given std_data_raw (whole fits noise map), it returns std_data (adapted to the spatial positions of this->data). dim1, dim0 = spatial dimensions of whole fits noise map.
	z=std::vector<std::vector<T>>();

	noise_data = z_data;
	z_data=std::vector<std::vector<T>>();

	std::vector <std::vector<T>> z_cube(this->dim_cube[0], std::vector<T>(this->dim_cube[1],0.));//, std::vector<T>(dim_data[2],0.)));
	reshape_noise_up_cube(noise_data, z_cube); //!<When given std_data (noise map related to the spatial position of this->data), it returns std_cube (the noise map used through multiresolution)

	noise_cube = z_cube;
	z_cube=std::vector<std::vector<T>>();
}

template<typename T>
void hypercube<T>::get_vector_from_binary(std::vector<std::vector<std::vector<T>>> &z)
{
   	int filesize = dim_data[0]*dim_data[1]*dim_data[2];

   	std::ifstream is("./data.raw", std::ifstream::binary);

   	std::cout<<"taille :"<<filesize<<std::endl;

   	const size_t count = filesize;
   	std::vector<T> vec(count);
   	is.read(reinterpret_cast<char*>(&vec[0]), count*sizeof(T));
   	is.close();

	std::vector <std::vector<T>> y;
	std::vector <T> x;
	int compteur(0);
/*
	for (int j(0); j<dim_data[2]; j++)
	{
		for (int k(j*dim_data[1]); k<(j+1)*dim_data[1]; k++)
		{
			for (int i(k*dim_data[0]); i<(k+1)*dim_data[0]; i++)
			{
				x.vector::push_back(vec[i]);
			}
			y.vector::push_back(x);
			x.clear();
		}
		z.vector::push_back(y);
		y.clear();
	}
*/
	int i__(0);
	int k,j,i;

	for(k=0; k<dim_data[0]; k++)
	{
		for(j=0; j<dim_data[1]; j++)
		{
			for(i=0; i<dim_data[2]; i++)
			{
				z[k][j][i] = vec[i__];
				i__++;
			}
		}
	}

/*
	for(int k(0); k<dim_data[0]; k++)
	{
		for(int j(0); j<dim_data[1]; j++)
		{
			for(int i(0); i< dim_data[2]; i++)
			{
				std::cout<<"cube["<<i<<"]["<<j<<"]["<<k<<"] = "<<z[k][j][i]<<std::endl;
			}
		}
	}
*/
}

template<typename T>
int hypercube<T>::writeImage()
{
    long naxis    =   2;      
    long naxes[2] = { 300, 200 };   
    
    // declare auto-pointer to FITS at function scope. Ensures no resources
    // leaked if something fails in dynamic allocation.
    std::auto_ptr<FITS> pFits(0);
      
    try
    {                
        // overwrite existing file if the file already exists.
        const std::string fileName("!atestfil.fit");            
        
        // Create a new FITS object, specifying the data type and axes for the primary
        // image. Simultaneously create the corresponding file.
        
        // this image is unsigned short data, demonstrating the cfitsio extension
        // to the FITS standard.
        
        pFits.reset( new FITS(fileName , USHORT_IMG , naxis , naxes ) );
    }
    catch (FITS::CantCreate)
    {
          // ... or not, as the case may be.
          return -1;       
    }
    
    // references for clarity.
    
    long& vectorLength = naxes[0];
    long& numberOfRows = naxes[1];
    long nelements(1); 
    
    
    // Find the total size of the array. 
    // this is a little fancier than necessary ( It's only
    // calculating naxes[0]*naxes[1]) but it demonstrates  use of the 
    // C++ standard library accumulate algorithm.
    
    nelements = std::accumulate(&naxes[0],&naxes[naxis],1,std::multiplies<long>());
           
    // create a new image extension with a 300x300 array containing float data.
    
    std::vector<long> extAx(2,300);
    string newName ("NEW-EXTENSION");
    ExtHDU* imageExt = pFits->addImage(newName,FLOAT_IMG,extAx);
    
    // create a dummy row with a ramp. Create an array and copy the row to 
    // row-sized slices. [also demonstrates the use of valarray slices].   
    // also demonstrate implicit type conversion when writing to the image:
    // input array will be of type float.
    
    std::valarray<int> row(vectorLength);
    for (long j = 0; j < vectorLength; ++j) row[j] = j;
    std::valarray<int> array(nelements);
    for (int i = 0; i < numberOfRows; ++i)
    {
        array[std::slice(vectorLength*static_cast<int>(i),vectorLength,1)] = row + i;     
    }
    
    // create some data for the image extension.
    long extElements = std::accumulate(extAx.begin(),extAx.end(),1,std::multiplies<long>()); 
    std::valarray<float> ranData(extElements);
    const float PIBY (M_PI/150.);
    for ( int jj = 0 ; jj < extElements ; ++jj) 
    {
            float arg = PIBY*jj;
            ranData[jj] = std::cos(arg);
    }
 
    long  fpixel(1);
    
    // write the image extension data: also demonstrates switching between
    // HDUs.
    imageExt->write(fpixel,extElements,ranData);
    
    //add two keys to the primary header, one long, one complex.
    
    long exposure(1500);
    std::complex<float> omega(std::cos(2*M_PI/3.),std::sin(2*M_PI/3));
    pFits->pHDU().addKey("EXPOSURE", exposure,"Total Exposure Time"); 
    pFits->pHDU().addKey("OMEGA",omega," Complex cube root of 1 ");  

    
    // The function PHDU& FITS::pHDU() returns a reference to the object representing 
    // the primary HDU; PHDU::write( <args> ) is then used to write the data.
    pFits->pHDU().write(fpixel,nelements,array);
    
    // PHDU's friend ostream operator. Doesn't print the entire array, just the
    // required & user keywords, and is provided largely for testing purposes [see 
    // readImage() for an example of how to output the image array to a stream].
    
    std::cout << pFits->pHDU() << std::endl;

    return 0;
}

template<typename T>
int hypercube<T>::save_noise_map_in_fits(parameters<T> &M, std::vector <std::vector<T>>& noise_map){
	long ax1 = long(noise_map.size());
	long ax2 = long(noise_map[0].size());
    long naxis    =   2;      
    long naxes[2] = { ax1, ax2 };   	
    
    // declare auto-pointer to FITS at function scope. Ensures no resources
    // leaked if something fails in dynamic allocation.
    std::auto_ptr<FITS> pFits(0);

	//adding a "!" to the filename, it allows overwritting
	const std::string fileName = "!" + M.filename_noise;
	std::cout<<"fileName = "<<fileName<<std::endl;

    try
    {                
        // this image is unsigned short data, demonstrating the cfitsio extension
        // to the FITS standard.
		if(M.float_mode){
		    pFits.reset( new FITS(fileName , FLOAT_IMG , naxis , naxes ) );
		}else if(M.double_mode){
		    pFits.reset( new FITS(fileName , DOUBLE_IMG , naxis , naxes ) );
		}
    }
    catch (FITS::CantCreate)
    {
          // ... or not, as the case may be.
          return -1;       
    }
    
    // references for clarity.
    long& vectorLength = naxes[0];
    long& numberOfRows = naxes[1];
    long nelements(1); 
    
    // Find the total size of the array. 
    // this is a little fancier than necessary ( It's only
    // calculating naxes[0]*naxes[1]) but it demonstrates  use of the 
    // C++ standard library accumulate algorithm.    
    nelements = std::accumulate(&naxes[0],&naxes[naxis],1,std::multiplies<long>());
    std::valarray<T> array_T(nelements);
    for (long j = 0; j < vectorLength; ++j){
    	for (int i = 0; i < numberOfRows; ++i)
    	{
        	array_T[i+j*numberOfRows] = T(noise_map[j][i]);     
    	}
	}
   // create some data for the image extension.
    long fpixel(1);
    
    //add two keys to the primary header, one long, one complex.
    long exposure(1500);
    std::complex<float> omega(std::cos(2*M_PI/3.),std::sin(2*M_PI/3));
    pFits->pHDU().addKey("EXPOSURE", exposure,"Total Exposure Time"); 
    pFits->pHDU().addKey("OMEGA",omega," Complex cube root of 1 ");  

    // The function PHDU& FITS::pHDU() returns a reference to the object representing 
    // the primary HDU; PHDU::write( <args> ) is then used to write the data.
//    pFits->pHDU().write(fpixel,nelements,array_T);
    pFits->pHDU().write(fpixel,nelements,array_T);
        
    // PHDU's friend ostream operator. Doesn't print the entire array, just the
    // required & user keywords, and is provided largely for testing purposes [see 
    // readImage() for an example of how to output the image array to a stream].
    
    std::cout << pFits->pHDU() << std::endl;

	//	pFits->pHDU().write(fpixel, nelements, noise_map_valarray, nullPtr);
    return 0;
}

template<typename T>
int hypercube<T>::save_grid_in_fits(parameters<T> &M, std::vector <std::vector<std::vector<T>>>& grid){
	long ax1 = long(grid.size()); //3*n_gauss
	long ax2 = long(grid[0].size()); //dim_y
	long ax3 = long(grid[0][0].size()); //dim_x
    long naxis = 3;      

//	std::cout<<"naxes : ax1,ax2,ax3 = "<<ax1<<","<<ax2<<","<<ax3<<std::endl;
//	std::cout<<"grid.size() : " << grid.size() << " , " << grid[0].size() << " , " << grid[0][0].size() <<  std::endl;

    long naxes[3] = { ax1, ax2, ax3 };   
    long naxes_bis[3] = { ax3, ax2, ax1 };   
   
//	std::cout<<"ax1 = "<<ax1<<std::endl;
//	std::cout<<"ax2 = "<<ax2<<std::endl;
//	std::cout<<"ax3 = "<<ax3<<std::endl;

	  // declare auto-pointer to FITS at function scope. Ensures no resources
    // leaked if something fails in dynamic allocation.
    std::auto_ptr<FITS> pFits(0);

//	std::cout<<"fileout = "<<M.fileout<<std::endl;

	//adding a "!" to the filename, it allows overwritting
	const std::string fileName = "!" + M.name_without_extension+".fits";
//	std::cout<<"fileName = "<<fileName<<std::endl;

    try
    {                
        // this image is unsigned short data, demonstrating the cfitsio extension
        // to the FITS standard.
		if(M.float_mode){
		    pFits.reset( new FITS(fileName , FLOAT_IMG , naxis , naxes_bis ) );
		}else if(M.double_mode){
		    pFits.reset( new FITS(fileName , DOUBLE_IMG , naxis , naxes_bis ) );
		}
    }
    catch (FITS::CantCreate)
    {
          // ... or not, as the case may be.
          return -1;       
    }
    
    // references for clarity.
    long& vectorLength = naxes[0];
    long& numberOfRows = naxes[1];
    long& depth = naxes[2];
    
    long nelements(1); 

    // C++ standard library accumulate algorithm.    
    nelements = std::accumulate(&naxes[0],&naxes[naxis],1,std::multiplies<long>());
    std::valarray<T> array_T(nelements);
	/*
    for (long j = 0; j < depth; ++j){
    	for (int i = 0; i < numberOfRows; ++i){
			for (int k = 0; k < vectorLength; ++k){
        		array_T[k+i*vectorLength + j*numberOfRows*vectorLength] = T(grid[k][i][j]);     
			}
    	}
	}
	*/

//	std::cout<<"naxes : vectorLength,numberOfRows,depth = "<<vectorLength<<","<<numberOfRows<<","<<depth<<std::endl;

    for (long j = 0; j < depth; ++j){
    	for (int i = 0; i < numberOfRows; ++i){
			for (int k = 0; k < vectorLength; ++k){
        		array_T[j+i*depth+k*depth*numberOfRows] = T(grid[k][i][j]);     
			}
    	}
	}
    // create some data for the image extension.
    long fpixel(1);
    
    //add two keys to the primary header, one long, one complex.
	int n_y = grid[0].size();
	int n_x = grid[0][0].size();
 
    pFits->pHDU().addKey("n_gauss", M.n_gauss,"Number of gaussians."); 
    pFits->pHDU().addKey("dim_y", n_y,"Spatial dimension on the Y axis."); 
    pFits->pHDU().addKey("dim_x", n_x,"Spatial dimension on the X axis."); 
    pFits->pHDU().addKey("lambda_amp",M.lambda_amp,"Hyperparameter to be tuned (smoothness).");  
    pFits->pHDU().addKey("lambda_mu",M.lambda_mu,"Hyperparameter to be tuned (smoothness).");  
    pFits->pHDU().addKey("lambda_sig",M.lambda_sig,"Hyperparameter to be tuned (smoothness).");  
    pFits->pHDU().addKey("lambda_var_sig",M.lambda_var_sig,"Hyperparameter to be tuned (smoothness).");  
    pFits->pHDU().addKey("maxiter",M.maxiter,"Maximum of iterations at each multiresolution level.");
    pFits->pHDU().addKey("lb_sig",M.lb_sig,"Lower bounds for sigma during multiresolution.");  
    pFits->pHDU().addKey("ub_sig",M.ub_sig,"Upper bounds for sigma during multiresolution.");
    pFits->pHDU().addKey("amp_fact_init",M.amp_fact_init,"Amplitude factor for the initial spectrum (if computed with ROHSA).");  
    pFits->pHDU().addKey("sig_init",M.sig_init,"Sigma value for the initial spectrum (if computed with ROHSA).");  
    pFits->pHDU().addKey("lb_sig_init",M.lb_sig_init,"Lower bounds for sigma initialization (if computed with ROHSA).");  
    pFits->pHDU().addKey("ub_sig_init",M.ub_sig_init,"Upper bounds for sigma initialization (if computed with ROHSA).");  
    pFits->pHDU().addKey("init_option",M.init_option,"Option chosen to initialize the spectrum (if computed with ROHSA).");
    pFits->pHDU().addKey("maxiter_init",M.maxiter_init,"Maximum of iterations for the initialization of the spectrum (if computed with ROHSA).");
    pFits->pHDU().addKey("lstd",M.lstd,"Lower index for the computation of the standard deviation map (if the std_map is computed on the fly)"); 
    pFits->pHDU().addKey("ustd",M.ustd,"Upper index for the computation of the standard deviation map (if the std_map is computed on the fly)"); 

    // The function PHDU& FITS::pHDU() returns a reference to the object representing 
    // the primary HDU; PHDU::write( <args> ) is then used to write the data.
//    pFits->pHDU().write(fpixel,nelements,array_T);
    pFits->pHDU().write(fpixel,nelements,array_T);
        
    // PHDU's friend ostream operator. Doesn't print the entire array, just the
    // required & user keywords, and is provided largely for testing purposes [see 
    // readImage() for an example of how to output the image array to a stream].
    
//    std::cout << pFits->pHDU() << std::endl;

	//	pFits->pHDU().write(fpixel, nelements, noise_map_valarray, nullPtr);
    return 0;
}



template<typename T>
T hypercube<T>::model_function(int x, T a, T m, T s) {

	return a*exp(-pow((T(x)-m),2.) / (2.*pow(s,2.)));

}


template<typename T>
void hypercube<T>::mean_parameters(std::vector<std::vector<std::vector<T>>> &params, int num_gauss)
{
	for(int p=0; p<3*num_gauss;p++){
		T mean = 0.;
		for(int i=0;i<this->dim_data[0];i++){
			for(int j=0;j<this->dim_data[1];j++){
				mean += params[p][j][i];
			}
		}
		mean = mean/(this->dim_data[0]*this->dim_data[1]);
		if (p%3 ==0)
			printf("Gaussienne n°%d, par n°%d, moyenne a     = %f \n", (p-p%3)/3, p, mean);
		if (p%3 ==1)
			printf("Gaussienne n°%d, par n°%d, moyenne mu    = %f \n", (p-p%3)/3, p, mean);
		if (p%3 ==2)
			printf("Gaussienne n°%d, par n°%d, moyenne sigma = %f \n", (p-p%3)/3, p, mean);
	}
}



template <typename T> 
void hypercube<T>::save_result(std::vector<std::vector<std::vector<T>>>& grid_params, parameters<T>& M) {

  std::cout<<"dim_data[0] = "<<dim_data[0]<<std::endl;
  std::cout<<"dim_data[1] = "<<dim_data[1]<<std::endl;
  std::cout<<"dim_data[2] = "<<dim_data[2]<<std::endl;
  
//  std::string space = "           "; 

  std::ofstream myfile;

  std::string filename = M.name_without_extension + ".dat";

  myfile.open(filename);//, std::ofstream::out | std::ofstream::trunc);
//  myfile.open(M.fileout);//, std::ofstream::out | std::ofstream::trunc);
  myfile << std::setprecision(18);
  myfile << "# \n";
  myfile << "# ______Parameters_____\n";
  myfile << "# n_gauss = "<< M.n_gauss<<"\n";
//  myfile << "# n_gauss_add = "<< M.n_gauss<<"\n";
  myfile << "# lambda_amp = "<< M.lambda_amp<<"\n";
  myfile << "# lambda_mu = "<< M.lambda_mu<<"\n";
  myfile << "# lambda_sig = "<< M.lambda_sig<<"\n";
  myfile << "# lambda_var_sig = "<< M.lambda_var_sig<<"\n";
  myfile << "# amp_fact_init = "<< M.amp_fact_init<<"\n";
  myfile << "# sig_init = "<< M.sig_init<<"\n";
  myfile << "# lb_sig_init = "<< M.lb_sig_init<<"\n";
  myfile << "# ub_sig_init = "<< M.ub_sig_init<<"\n";
  myfile << "# lb_sig = "<< M.lb_sig<<"\n";
  myfile << "# ub_sig = "<< M.ub_sig<<"\n";
  myfile << "# init_option = "<< M.init_option<<"\n";
  myfile << "# maxiter_itit = "<< M.maxiter_init<<"\n";
  myfile << "# maxiter = "<< M.maxiter<<"\n";
  myfile << "# lstd = "<< M.lstd<<"\n";
  myfile << "# ustd = "<< M.ustd<<"\n";
  myfile << "# noise = "<< M.noise<<"\n";
  myfile << "# regul = "<< M.regul<<"\n";
  myfile << "# descent = "<< M.descent<<"\n";
  myfile << "# \n";
  myfile << "# \n";
  myfile << "# \n";
  myfile << "# \n";
  myfile << "# \n";

/*
    write(12,fmt=*) "# n_gauss_add = ", n_gauss_add
    write(12,fmt=*) "# lambda_lym_sig = ", lambda_lym_sig
*/
  myfile << "# i, j, A, mean, sigma\n";

  std::cout << "grid_params.size() : "<< grid_params.size() << " , " << grid_params[0].size()  << " , " << grid_params[0][0].size() << std::endl;

  for(int i = 0; i<dim_data[1]; i++){
  	for(int j = 0; j<dim_data[0]; j++){
  	  for(int k = 0; k<M.n_gauss; k++){
//        myfile << "           "<< i << "           " << j << "  " << grid_params[3*k+0][i][j] << "        " << grid_params[3*k+1][i][j] << "        " << grid_params[3*k+2][i][j] <<"     \n";
        myfile << "\t"<< i << "\t" << j << "\t" << grid_params[3*k+0][i][j] << "\t" << grid_params[3*k+1][i][j] << "\t" << grid_params[3*k+2][i][j] <<"\t\n";
      }
  	}
  }
}


template <typename T> void 
hypercube<T>::save_result_multires(std::vector<std::vector<std::vector<T>>>& grid_params, parameters<T>& M, int num) {

	std::string s = std::to_string(num);//"plot_through_regu_level_num_");
	char const *pchar = s.c_str();

	char str[220];
	strcpy (str,"result_level_");
	strcat (str,pchar);
	strcat (str,".dat");
	puts (str);

  std::cout << "grid_params.size() : "<< grid_params.size() << " , " << grid_params[0].size()  << " , " << grid_params[0][0].size() << std::endl;

//  std::string space = "           "; 

  std::ofstream myfile;
  myfile.open(str);//, std::ofstream::out | std::ofstream::trunc);
  myfile << std::setprecision(18);
  myfile << "# \n";
  myfile << "# ______Parameters_____\n";
  myfile << "# n_gauss = "<< M.n_gauss<<"\n";
//  myfile << "# n_gauss_add = "<< M.n_gauss<<"\n";
  myfile << "# lambda_amp = "<< M.lambda_amp<<"\n";
  myfile << "# lambda_mu = "<< M.lambda_mu<<"\n";
  myfile << "# lambda_sig = "<< M.lambda_sig<<"\n";
  myfile << "# lambda_var_sig = "<< M.lambda_var_sig<<"\n";
  myfile << "# amp_fact_init = "<< M.amp_fact_init<<"\n";
  myfile << "# sig_init = "<< M.sig_init<<"\n";
  myfile << "# lb_sig_init = "<< M.lb_sig_init<<"\n";
  myfile << "# ub_sig_init = "<< M.ub_sig_init<<"\n";
  myfile << "# lb_sig = "<< M.lb_sig<<"\n";
  myfile << "# ub_sig = "<< M.ub_sig<<"\n";
  myfile << "# init_option = "<< M.init_option<<"\n";
  myfile << "# maxiter_itit = "<< M.maxiter_init<<"\n";
  myfile << "# maxiter = "<< M.maxiter<<"\n";
  myfile << "# lstd = "<< M.lstd<<"\n";
  myfile << "# ustd = "<< M.ustd<<"\n";
  myfile << "# noise = "<< M.noise<<"\n";
  myfile << "# regul = "<< M.regul<<"\n";
  myfile << "# descent = "<< M.descent<<"\n";
  myfile << "# \n";
  myfile << "# \n";
  myfile << "# \n";
  myfile << "# \n";
  myfile << "# \n";

/*
    write(12,fmt=*) "# n_gauss_add = ", n_gauss_add
    write(12,fmt=*) "# lambda_lym_sig = ", lambda_lym_sig
*/
  myfile << "# i, j, A, mean, sigma\n";


  for(int i = 0; i<grid_params[0].size(); i++){
  	for(int j = 0; j<grid_params[0][0].size(); j++){
  	  for(int k = 0; k<M.n_gauss; k++){
//        myfile << "           "<< i << "           " << j << "  " << grid_params[3*k+0][i][j] << "        " << grid_params[3*k+1][i][j] << "        " << grid_params[3*k+2][i][j] <<"     \n";
        myfile << "\t"<< i << "\t" << j << "\t" << grid_params[3*k+0][i][j] << "\t" << grid_params[3*k+1][i][j] << "\t" << grid_params[3*k+2][i][j] <<"\t\n";
      }
  	}
  }
}


template<typename T>
void hypercube<T>::reshape_noise_up_data(std::vector<std::vector<T>>& std_data_raw, std::vector<std::vector<T>>& std_data_out, int dim1, int dim0)
{
	this->position_y = 650;
	this->position_x = 730;
		for(int j=0; j< this->dim_data[1]; j++)
		{
			for(int i=0; i< this->dim_data[0]; i++)
			{
				std_data_out[j][i]= std_data_raw[j+this->position_y-this->dim_data[1]/2][i+this->position_x-this->dim_data[0]/2];
			}
		}
		/*
	if(this->larger_power_of_two_option){
		int offset_y = int(abs(dim1-this->dim_data[1])/2);
		int offset_x = int(abs(dim0-this->dim_data[0])/2);
		for(int j=0; j< this->dim_cube[1]; j++)
		{
			for(int i=0; i< this->dim_cube[0]; i++)
			{
				std_data_out[j][i]= std_data_raw[j+offset_y][i+offset_x];
			}
		}
	}else if(this->get_a_square_of_a_given_size_option){
			int shift = this->size_side_square_option/2;
			for(int i=0; i< this->size_side_square_option; i++)
			{
				for(int j=0; j< this->size_side_square_option; j++)
				{
					std_data_out[j][i]= std_data_raw[j+this->position_y-shift][i+this->position_x-shift];
				}
			}
	}else if(true || this->position_given){
		this->position_y = 650;
		this->position_x = 730;
		for(int j=0; j< this->dim_data[1]; j++)
		{
			for(int i=0; i< this->dim_data[0]; i++)
			{
				std_data_out[j][i]= std_data_raw[j+this->position_y-this->dim_data[1]/2][i+this->position_x-this->dim_data[0]/2];
			}
		}
	}else{
		for(int j=0; j< this->dim_data[1]; j++)
		{
			for(int i=0; i< this->dim_data[0]; i++)
			{
				std_data_out[j][i]= std_data_raw[j][i];
			}
		}
	}
*/



/*

	int d_cube[] = {0,0};
	int d_map[] = {0,0};
	d_cube[0] = std_map_out.size();
	d_cube[1] = std_map_out[0].size();
	d_map[0] = this->std_map_in.size();
	d_map[1] = this->std_map_in[0].size();
	//compute the offset so that the data file lies in the center of a cube
//	int offset_x = (-d_cube[1]+d_map[1])/2;
//	int offset_y = (-d_cube[0]+d_map[0])/2;

	int offset_x = this->position_x;
	int offset_y = this->position_y;
	int half_size_0 = d_cube[0]/2;
	int half_size_1 = d_cube[1]/2;

	std::cout << "d_cube[0] = " <<d_cube[0]<< std::endl;
	std::cout << "d_cube[1] = " <<d_cube[1]<< std::endl;
	std::cout << "d_map[0] = " <<d_map[0]<< std::endl;
	std::cout << "d_map[1] = " <<d_map[1]<< std::endl;

	for(int j=offset_y; j<d_cube[0]+offset_y; j++)
	{
		for(int i=offset_x; i< d_cube[1]+offset_x; i++)
		{
			std_map_out[j-offset_y][i-offset_x]= std_map_in[j-half_size_0][i-half_size_1];
		}
	}
	*/
}

template<typename T>
void hypercube<T>::reshape_noise_up_cube(std::vector<std::vector<T>>& std_data, std::vector<std::vector<T>>& std_cube_out)
{
	int offset_y = int(abs(this->dim_cube[1]-this->dim_data[1])/2);
	int offset_x = int(abs(this->dim_cube[0]-this->dim_data[0])/2);
	for(int j=0; j< this->dim_cube[1]; j++)
	{
		for(int i=0; i< this->dim_cube[0]; i++)
		{
			std_cube_out[j][i]= std_data[j+offset_y][i+offset_x];
		}
	}


/*

	int d_cube[] = {0,0};
	int d_map[] = {0,0};
	d_cube[0] = std_map_out.size();
	d_cube[1] = std_map_out[0].size();
	d_map[0] = this->std_map_in.size();
	d_map[1] = this->std_map_in[0].size();
	//compute the offset so that the data file lies in the center of a cube
//	int offset_x = (-d_cube[1]+d_map[1])/2;
//	int offset_y = (-d_cube[0]+d_map[0])/2;

	int offset_x = this->position_x;
	int offset_y = this->position_y;
	int half_size_0 = d_cube[0]/2;
	int half_size_1 = d_cube[1]/2;

	std::cout << "d_cube[0] = " <<d_cube[0]<< std::endl;
	std::cout << "d_cube[1] = " <<d_cube[1]<< std::endl;
	std::cout << "d_map[0] = " <<d_map[0]<< std::endl;
	std::cout << "d_map[1] = " <<d_map[1]<< std::endl;

	for(int j=offset_y; j<d_cube[0]+offset_y; j++)
	{
		for(int i=offset_x; i< d_cube[1]+offset_x; i++)
		{
			std_map_out[j-offset_y][i-offset_x]= std_map_in[j-half_size_0][i-half_size_1];
		}
	}
	*/
}

template<typename T>
void hypercube<T>::get_noise_map_from_GHIGLS(parameters<T> &M, std::vector <std::vector<T>>& noise_map){
	std::auto_ptr<FITS> pInfile(new FITS(M.filename_fits,Read,true));

	PHDU& image = pInfile->pHDU();

    ExtHDU& extension = pInfile->extension(2);

	std::valarray<T> contents;
    extension.readAllKeys();

	int dim[] = {0,0};

    extension.read(contents);

	std::cout << extension << std::endl;

    long ax1(extension.axis(0));
    long ax2(extension.axis(1));

	dim[0]=ax1;
	dim[1]=ax2;

	std::vector <std::vector<T>> z(dim[0], std::vector<T>(dim[1],0.));//, std::vector<T>(dim_data[2],0.)));

	int i__=0;
	for(int j=0; j<dim[1]; j++)
	{
		for(int k=0; k<dim[0]; k++)
		{
			z[j][k] = contents[dim[0]*j+k];
		}
	}
	noise_map = z;
	//why not directly noise_map[][] = ... ?
/*
	for(int j=0; j<dim[1]; j++)
	{
		for(int k=0; k<dim[0]; k++)
		{
			printf("z[%d][%d] = %f\n",k,j,z[k][j]);
		}
	}
*/
}


#endif