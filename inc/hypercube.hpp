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
	hypercube(parameters<T> &M);
	hypercube(parameters<T> &M,int indice_debut, int indice_fin); // assuming whole_data_in_cube = false (faster and better provided the dimensions are close)
	hypercube(parameters<T> &M,int indice_debut, int indice_fin, bool whole_data_in_cube);
	hypercube(parameters<T> &M,int indice_debut, int indice_fin, bool whole_data_in_cube, bool one_level);

	void display_cube(int rang);
	void display_data(int rang);
	void display(std::vector<std::vector<std::vector<T>>> &tab, int rang);
	void plot_line(std::vector<std::vector<std::vector<T>>> &params, int ind_x, int ind_y, int n_gauss_i);
	void plot_lines(std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<std::vector<T>>> &cube_mean);
	void plot_multi_lines(std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<std::vector<T>>> &cube_mean);
	void plot_multi_lines(std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<std::vector<T>>> &cube_mean, std::string some_string);
	void display_result_and_data(std::vector<std::vector<std::vector<T>>> &params,int rang, int n_gauss_i, bool dat_or_not);
	void display_avec_et_sans_regu(std::vector<std::vector<std::vector<T>>> &params, int num_gauss, int num_par, int plot_numero);
	void display_2_gaussiennes(std::vector<std::vector<std::vector<T>>> &params,int rang, int n_gauss_i, int n1, int n2);
	void display_2_gaussiennes_par_par_par(std::vector<std::vector<std::vector<T>>> &params,int rang, int n_gauss_i, int n1, int n2);
	void mean_parameters(std::vector<std::vector<std::vector<T>>> &params, int num_gauss);
	void simple_plot_through_regu(std::vector<std::vector<std::vector<T>>> &params, int num_gauss, int num_par, int plot_numero);



	T model_function(int x, T a, T m, T s);
	void display_result(std::vector<std::vector<std::vector<T>>> &params, int rang, int n_gauss_i);

	int dim2nside(); //obtenir les dimensions 2^n
	void brute_show(const std::vector<std::vector<std::vector<T>>> &z, int depth, int length1, int length2);
	void multiresolution(int nside); 
	int get_binary_from_fits();
	void get_array_from_fits(parameters<T> &M);
	void get_vector_from_binary(std::vector<std::vector<std::vector<T>>> &z);
	void show_data(); 
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

	void write_vector_to_file(const std::vector<T>& myVector, std::string filename);
	std::vector<T> read_vector_from_file(std::string filename);

	void save_result(std::vector<std::vector<std::vector<T>>>&, parameters<T>&);
	void save_result_multires(std::vector<std::vector<std::vector<T>>>&, parameters<T>&, int);


	int indice_debut, indice_fin; //!< Only some spectral ranges of the hypercube are exploitable. We cut the hypercube, this will introduce an offset on the result values.
	std::vector<std::vector<std::vector<T>>> cube; //!< The hypercube "data" is centered into a larger hypercube "cube" for the purpose of multiresolution this hypercube is useful and its spatial dimensions are \f$  2^{n\_side} \times 2^{n\_side} \f$. Where \f$n\_side\f$ is computed by dim2nside(), it is the smallest power of 2 greater than the spatial dimensions. 
//	std::vector<std::vector<std::vector<T>>> data_not_reshaped;
	std::vector<std::vector<std::vector<T>>> data; //!< Hypercube array extracted from the fits file, its spectral range is changed according to indice_debut and indice_fin. 

	int dim_data[3];
	int dim_cube[3];
	std::vector<int> dim_data_v;
	std::vector<int> dim_cube_v;
	int nside;

	std::string filename; 
};



namespace plt = matplotlibcpp;
//using namespace std;
using namespace CCfits;

template<typename T>
hypercube<T>::hypercube(parameters<T> &M, int indice_debut, int indice_fin, bool whole_data_in_cube, bool last_level_power_of_two)
{
	this->indice_debut= indice_debut;
	this->indice_fin = indice_fin;
	if(M.file_type_fits){
//		if(last_level_power_of_two){
		get_array_from_fits(M);
	//	get_dimensions_from_fits(); //inutile
//		get_binary_from_fits(); // WARNING 
//		get_vector_from_binary(this->data);
		if(whole_data_in_cube){
			this->nside = dim2nside();
//			std::cout << "	nside =  " <<this->nside<< std::endl;		
		}else{
			this->nside = dim2nside()-1;
		}
	}
	if(M.file_type_dat){
		this->data = use_dat_file(M);
		this->nside = dim2nside();
	}
	if(last_level_power_of_two){
		this->nside = dim2nside()-1;
		std::cout<<"this->nside = "<<this->nside<<std::endl;
	}
//	std::cout<<dim_data[2]<<std::endl;
//	exit(0);
	
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

/*
			for(int i=0; i< this->dim_cube[0]/2; i++)
				{
					for(int j=0; j< this->dim_cube[1]/2	; j++)
					{
						for(int k= indice_debut; k<= indice_fin; k++)
						{
							data_reshaped_local[i][j][k-indice_debut]= this->data[i+this->dim_data[0]/4][j+this->dim_data[1]/4][k];
						}
					}
				}
			std::cout<<"DEBUG"<<std::endl;
*/
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


/*
	for(int k(0); k<dim_cube[0]; k++)
	{
		for(int j(0); j<dim_cube[1]; j++)
		{
			for(int i(0); i< dim_cube[2]; i++)
			{
			std::cout<<"cube["<<k<<"]["<<j<<"]["<<i<<"] = "<<data[k][j][i]<<std::endl;
			exit(0);
			}
		}
	}
*/
}

template<typename T>
hypercube<T>::hypercube(parameters<T> &M, int indice_debut, int indice_fin, bool whole_data_in_cube)
{
	this->indice_debut= indice_debut;
	this->indice_fin = indice_fin;
	if(M.file_type_fits){
		get_array_from_fits(M);
	//	get_dimensions_from_fits(); //inutile
//		get_binary_from_fits(); // WARNING 
//		get_vector_from_binary(this->data);
		if(whole_data_in_cube){
			this->nside = dim2nside();
//			std::cout << "	nside =  " <<this->nside<< std::endl;		
		}
		else{
			this->nside = dim2nside()-1;
		}
	}
	if(M.file_type_dat){
		this->data = use_dat_file(M);
		this->nside = dim2nside();
	}
//	std::cout << "	DEBUG " << std::endl;
//	std::cout<<dim_data[2]<<std::endl;
//	exit(0);
	
	dim_cube[0] =pow(2.0,nside);
	dim_cube[1] =pow(2.0,nside);
	if(M.file_type_fits){
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

	if(M.file_type_fits){
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

}

template<typename T>
hypercube<T>::hypercube(parameters<T> &M, int indice_debut, int indice_fin)
{
	this->indice_debut= indice_debut;
	this->indice_fin = indice_fin;
	if(M.file_type_fits){
		get_array_from_fits(M);
	//	get_dimensions_from_fits(); //inutile
//		get_binary_from_fits(); // WARNING 
//		get_vector_from_binary(this->data);
		this->nside = dim2nside()-1;
	}
	if(M.file_type_dat){
		this->data = use_dat_file(M);
		this->nside = dim2nside();
	}
//	std::cout << "	DEBUG " << std::endl;
//	std::cout<<dim_data[2]<<std::endl;

	dim_cube[0] =pow(2.0,nside);
	dim_cube[1] =pow(2.0,nside);
	dim_cube[2] = indice_fin-indice_debut+1;

	std::cout<<"dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"dim_data[2] = "<<dim_data[2]<<std::endl;

	std::cout<<"dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"dim_cube[2] = "<<dim_cube[2]<<std::endl;

	cube = reshape_up(indice_debut, indice_fin);

/*
	for(int k(0); k<dim_cube[0]; k++)
	{
		for(int j(0); j<dim_cube[1]; j++)
		{
			for(int i(0); i< dim_cube[2]; i++)
			{
			std::cout<<"cube["<<k<<"]["<<j<<"]["<<i<<"] = "<<cube[k][j][i]<<std::endl;
			}
		}
	}
*/
}

template<typename T>
hypercube<T>::hypercube(parameters<T> &M)
{
	if(M.file_type_fits){
		get_array_from_fits(M);
//		get_dimensions_from_fits();
//		get_binary_from_fits(); // WARNING 
//		get_vector_from_binary(this->data);
		this->nside = dim2nside()-1;
	}
	if(M.file_type_dat){
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
hypercube<T>::hypercube() //dummy constructor for initialization of an hypercube object
{

}

template<typename T>
std::vector<std::vector<std::vector<T>>> hypercube<T>::use_dat_file(parameters<T> &M)
{
   	int x,y,z;
	T v;

	std::ifstream fichier(M.filename_dat);

	fichier >> z >> x >> y;

	dim_data[2]=z;
	dim_data[1]=y;
	dim_data[0]=x;

	std::vector<std::vector<std::vector<T>>> data_(dim_data[0],std::vector<std::vector<T>>(dim_data[1],std::vector<T>(dim_data[2], 0.)));

	while(!fichier.std::ios::eof())
	{
   		fichier >> z >> y >> x >> v;
		data_[x][y][z] = v;
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
	return std::max( 0, std::min(int(ceil( log(T(dim_data[0]))/log(2.))), int(ceil( log(T(dim_data[1]))/log(2.))))  ) ;  
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

template<typename T>
void hypercube<T>::brute_show(const std::vector<std::vector<std::vector<T>>> &z, int depth, int length1, int length2)
{

	for (int k(0); k<length1; k++)
	{
		for (int i(0); i<length2; i++)
		{
			std::cout << "__résultat["<<i<<"]["<<k<<"]["<<0<<"]= " << z[i][k][0] << std::endl;
		}
	}
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
void hypercube<T>::write_in_file(std::vector<std::vector<std::vector<T>>> &file_in){
/*
	std::ofstream objetfichier;
 	objetfichier.open("./right_before_last_level.raw", std::ios::out | std::ofstream::binary ); //on ouvre le fichier en ecriture
	if (objetfichier.bad()) //permet de tester si le fichier s'est ouvert sans probleme
		std::cout<<"ERREUR À L'OUVERTURE DU FICHIER RAW AVANT ÉCRITURE"<< std::endl;
*/
	int dim_0 = file_in.size();
	int dim_1 = file_in[0].size();
	int dim_2 = file_in[0][0].size();

	printf("dim_0 = %d , dim_1 = %d , dim_1 = %d\n",dim_0, dim_1, dim_1);

	std::vector<T> file_in_flat(dim_0*dim_1*dim_2,0.);
/*
	T* file_in_flat = NULL;
	size_t size = dim_0*dim_1*dim_2*sizeof(T);
	file_in_flat = (T*)malloc(size);
*/

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
void hypercube<T>::get_array_from_fits(parameters<T> &M){
	std::auto_ptr<FITS> pInfile(new FITS(M.filename_fits,Read,true));

        PHDU& image = pInfile->pHDU();
	std::valarray<T> contents;
        image.readAllKeys();

        image.read(contents);

        // this doesn't print the data, just header info.
        // std::cout << image << std::endl;

        long ax1(image.axis(0));
        long ax2(image.axis(1));
        long ax3(image.axis(2));
	long ax4(image.axis(3));

	this->dim_data[0]=ax1;
	this->dim_data[1]=ax2;
	this->dim_data[2]=ax3;

	std::vector <std::vector<std::vector<T>>> z(dim_data[0], std::vector<std::vector<T>>(dim_data[1], std::vector<T>(dim_data[2],0.)));
	int i__=0;

	for(int i=0; i<dim_data[2]; i++)
	{
		for(int j=0; j<dim_data[1]; j++)
		{
		for(int k=0; k<dim_data[0]; k++)
			{
				z[k][j][i] = contents[i__];
				i__++;
//				std::cout<<"k,j,i = "<<k<<","<<j<<","<<i<<std::endl;
			}
		}
	}

	this->data=z;
	z=std::vector<std::vector<std::vector<T>>>();
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
void hypercube<T>::display_cube(int rang)
{
	std::vector<float> z(this->dim_cube[0]*this->dim_cube[1],0.);

	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){
			z[i*this->dim_cube[1]+j] = this->cube[i][j][rang];
		}
	}

	const float* zptr = &(z[0]);
	const int colors = 1;
	plt::clf();
	plt::title("Vue en coupe de cube");
	plt::imshow(zptr, this->dim_cube[0], this->dim_cube[1], colors);

	plt::save("imshow_cube.png");
	std::cout << "Result saved to 'imshow_cube.png'.\n"<<std::endl;
}

template<typename T>
void hypercube<T>::display_data(int rang)
{
	std::vector<float> z(this->dim_data[0]*this->dim_data[1],0.);
	for(int i=0;i<this->dim_data[0];i++){
		for(int j=0;j<this->dim_data[1];j++){
			z[i*this->dim_data[1]+j] = this->data[i][j][rang];
		}
	}

	const float* zptr = &(z[0]);
	const int colors = 1;

	plt::title("Vue en coupe de data");
	plt::imshow(zptr, this->dim_data[0], this->dim_data[1], colors);

	plt::save("imshow_data.png");
	std::cout << "Result saved to 'imshow_data.png'.\n"<<std::endl;
}


template<typename T>
void hypercube<T>::display(std::vector<std::vector<std::vector<T>>> &tab, int rang)
{
	int dim_[3];
	dim_[0]=tab.size();
	dim_[1]=tab[0].size();
	dim_[2]=tab[0][0].size();
	std::vector<float> z(dim_[0]*dim_[1],0.);

	for(int i=0;i<dim_[0];i++){
		for(int j=0;j<dim_[1];j++){
			z[i*dim_[1]+j] = tab[i][j][rang];
		}
	}

	const float* zptr = &(z[0]);
	const int colors = 1;
	plt::clf();
	plt::title("Vue en coupe de data");
	plt::imshow(zptr, dim_[0], dim_[1], colors);

	plt::save("imshow.png");
	std::cout << "Result saved to 'imshow.png'.\n"<<std::endl;
}

template<typename T>
void hypercube<T>::plot_line(std::vector<std::vector<std::vector<T>>> &params, int ind_x, int ind_y, int n_gauss_i) {

	std::vector<T> model(this->dim_cube[2],0.);
	std::vector<T> cube_line(this->dim_cube[2],0.);
	std::vector<T> params_line(3*n_gauss_i,0.);
	std::cout<< dim_cube[2] <<std::endl;
	for(int i=0; i<params_line.size(); i++) {
		params_line[i]=params[i][ind_y][ind_x];
	}

	for(int i=0; i<n_gauss_i; i++) {
		for(int k=0; k<this->dim_cube[2]; k++) {
			model[k]+= model_function(k, params_line[3*i], params_line[1+3*i], params_line[2+3*i]);
		}
	}

	for(int k=0; k<this->dim_cube[2]; k++) {
		cube_line[k]= cube[ind_x][ind_y][k];
	}



    std::vector<T> x(this->dim_cube[2]);
    for(int i=0; i<this->dim_cube[2]; ++i) {
        x.at(i) = i;
    }
   	plt::clf();
    // Set the size of output image = 1200x780 pixels
    plt::figure_size(1200, 780);

    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(x, cube_line,"r");

    // Plot line from given x and y data. Color is selected automatically.
    plt::plot(x, model,"b");
    plt::named_plot("data", x, cube_line);
    plt::named_plot("model", x, model);

    plt::xlim(0, this->dim_cube[2]);

    // Add graph title
    plt::title("Model vs Data Plot");
    // Enable legend.
    plt::legend();

	std::string s_x = std::to_string(ind_x);
	char const *pchar_x = s_x.c_str();
	std::string s_y = std::to_string(ind_y);
	char const *pchar_y = s_y.c_str();

	char str[100];//220
	strcpy (str,"./plot_");
	strcat (str,pchar_x);
	strcat (str,"_");
	strcat (str,pchar_y);
	strcat (str,".png");
	puts (str);

    // save figure
    std::cout << "Saving result to " << str << std::endl;;
    plt::save(str);
	//plt::show();
}

template<typename T>
void hypercube<T>::plot_lines(std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<std::vector<T>>> &cube_mean) {

	std::cout<<"params.size() ="<<params.size()<<std::endl;
	std::cout<<"params[0].size() ="<<params[0].size()<<std::endl;
	std::cout<<"params[0][0].size() ="<<params[0][0].size()<<std::endl;
	int nb_gaussian = int(floor(T(params.size())/3.));
	for(int ind_x=0; ind_x<params[0].size(); ind_x++) {
		for(int ind_y=0; ind_y<params[0][0].size(); ind_y++) {
			std::vector<T> model(this->dim_cube[2],0.);
			std::vector<T> cube_line(this->dim_cube[2],0.);
			std::vector<T> params_line(params.size(),0.);
//			std::cout<< dim_cube[2] <<std::endl;
			for(int i=0; i<params_line.size(); i++) {
				params_line[i]=params[i][ind_y][ind_x];
			}
			for(int i=0; i<params_line.size()/3; i++) {
				for(int k=0; k<this->dim_cube[2]; k++) {
					model[k]+= model_function(k+1, params_line[3*i], params_line[1+3*i], params_line[2+3*i]);
				}
			}

			for(int k=0; k<this->dim_cube[2]; k++) {
				cube_line[k]= cube_mean[ind_x][ind_y][k];
			}

			std::vector<T> x(this->dim_cube[2]);
			for(int i=0; i<this->dim_cube[2]; ++i) {
				x.at(i) = i;
			}

			printf("model[%d] = %f \n",3,model[3]);
			printf("cube_line[%d] = %f \n",3,cube_line[3]);

			plt::clf();
			// Set the size of output image = 1200x780 pixels
			plt::figure_size(1200, 780);

			// Plot line from given x and y data. Color is selected automatically.
			plt::plot(x, cube_line,"r");
			plt::plot(x, model,"b");

			// Plot line from given x and y data. Color is selected automatically.
			plt::named_plot("data", x, cube_line);
			plt::named_plot("model", x, model);

			plt::xlim(0, this->dim_cube[2]);

			// Add graph title
			plt::title("Model vs Data Plot");
			// Enable legend.
			plt::legend();

		//params[0].size()

			std::string s_dimensions_1 = std::to_string(params[0].size());
			char const *pchar_dimensions_1 = s_dimensions_1.c_str();
			std::string s_dimensions_2 = std::to_string(params[0][0].size());
			char const *pchar_dimensions_2 = s_dimensions_2.c_str();
			std::string s_x = std::to_string(ind_x);
			char const *pchar_x = s_x.c_str();
			std::string s_y = std::to_string(ind_y);
			char const *pchar_y = s_y.c_str();

			char str[100];//220
			strcpy (str,"./plot_dim_");
			strcat (str,pchar_dimensions_1);
			strcat (str,"_by_");
			strcat (str,pchar_dimensions_2);
			strcat (str,"_position_");
			strcat (str,pchar_x);
			strcat (str,"_by_");
			strcat (str,pchar_y);
			strcat (str,".png");
			puts (str);

			// save figure
			std::cout << "Saving result to " << str << std::endl;;
			plt::save(str);
			//plt::show();
		}
	}
}

template<typename T>
void hypercube<T>::plot_multi_lines(std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<std::vector<T>>> &cube_mean) {

	std::string color[] = {"g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k",};
	std::cout<<"params.size() ="<<params.size()<<std::endl;
	std::cout<<"params[0].size() ="<<params[0].size()<<std::endl;
	std::cout<<"params[0][0].size() ="<<params[0][0].size()<<std::endl;
	int nb_gaussian = int(floor(T(params.size())/3.));

	std::vector<T> x(this->dim_cube[2]);
	for(int i=0; i<this->dim_cube[2]; ++i) {
		x.at(i) = i;
	}
			plt::clf();
			// Set the size of output image = 1200x780 pixels
			plt::figure_size(1200, 780);
	for(int ind_x=0; ind_x<params[0].size(); ind_x++) {
		for(int ind_y=0; ind_y<params[0][0].size(); ind_y++) {

			std::vector<T> cube_line(this->dim_cube[2],0.);
			std::vector<T> params_line(params.size(),0.);
			std::vector<T> model(this->dim_cube[2],0.);

			for(int i=0; i<params_line.size(); i++) {
				params_line[i]=params[i][ind_y][ind_x];
			}

		for(int n_g=0; n_g<nb_gaussian; n_g++) {
			std::vector<T> model_g(this->dim_cube[2],0.);
			for(int k=0; k<this->dim_cube[2]; k++) {
				model_g[k]+= model_function(k+1, params_line[3*n_g], params_line[1+3*n_g], params_line[2+3*n_g]);
			}

/*
			std::string p_str_number = std::to_string(n_g);
			char const *p_name_model_g = p_str_number.c_str();
			char str_number[100];//220
			strcpy (str_number,"gaussian_number_");
			strcat (str_number,p_name_model_g);
			puts (str_number);
			plt::named_plot(str_number, x, model_g);
//			std::cout<<"str_number = "<<str_number<<std::endl;
*/

			plt::plot(x, model_g, color[n_g]);


		}
//			std::cout<< dim_cube[2] <<std::endl;
			for(int i=0; i<nb_gaussian; i++) {
				for(int k=0; k<this->dim_cube[2]; k++) {
					model[k]+= model_function(k+1, params_line[3*i], params_line[1+3*i], params_line[2+3*i]);
				}
			}


			for(int k=0; k<this->dim_cube[2]; k++) {
				cube_line[k]= cube_mean[ind_x][ind_y][k];
			}




			// Plot line from given x and y data. Color is selected automatically.
			plt::plot(x, cube_line, "r--");//color[0]);
			plt::plot(x, model, "b--");//color[0]);

			// Plot line from given x and y data. Color is selected automatically.
			plt::named_plot("data", x, cube_line);
			plt::named_plot("model", x, model);

			plt::xlim(0, this->dim_cube[2]);

			// Add graph title
			plt::title("Model vs Data Plot");
			// Enable legend.
			plt::legend();

		//params[0].size()

			std::string s_dimensions_1 = std::to_string(params[0].size());
			char const *pchar_dimensions_1 = s_dimensions_1.c_str();
			std::string s_dimensions_2 = std::to_string(params[0][0].size());
			char const *pchar_dimensions_2 = s_dimensions_2.c_str();
			std::string s_x = std::to_string(ind_x);
			char const *pchar_x = s_x.c_str();
			std::string s_y = std::to_string(ind_y);
			char const *pchar_y = s_y.c_str();

			char str[100];//220
			strcpy (str,"./plot_dim_");
			strcat (str,pchar_dimensions_1);
			strcat (str,"_by_");
			strcat (str,pchar_dimensions_2);
			strcat (str,"_position_");
			strcat (str,pchar_x);
			strcat (str,"_by_");
			strcat (str,pchar_y);
			strcat (str,".png");
			puts (str);

			// save figure
			std::cout << "Saving result to " << str << std::endl;;
			plt::save(str);
			plt::clf();
			//plt::show();

		}
	}
}

template<typename T>
void hypercube<T>::plot_multi_lines(std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<std::vector<T>>> &cube_mean, std::string some_string) {

	std::string color[] = {"g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k","g", "r", "c", "m", "y","k",};
	std::cout<<"params.size() ="<<params.size()<<std::endl;
	std::cout<<"params[0].size() ="<<params[0].size()<<std::endl;
	std::cout<<"params[0][0].size() ="<<params[0][0].size()<<std::endl;
	int nb_gaussian = int(floor(T(params.size())/3.));

	std::vector<T> x(this->dim_cube[2]);
	for(int i=0; i<this->dim_cube[2]; ++i) {
		x.at(i) = i;
	}
			plt::clf();
			// Set the size of output image = 1200x780 pixels
			plt::figure_size(1200, 780);
	for(int ind_x=0; ind_x<params[0].size(); ind_x++) {
		for(int ind_y=0; ind_y<params[0][0].size(); ind_y++) {

			std::vector<T> cube_line(this->dim_cube[2],0.);
			std::vector<T> params_line(params.size(),0.);
			std::vector<T> model(this->dim_cube[2],0.);

			for(int i=0; i<params_line.size(); i++) {
				params_line[i]=params[i][ind_y][ind_x];
			}

		for(int n_g=0; n_g<nb_gaussian; n_g++) {
			std::vector<T> model_g(this->dim_cube[2],0.);
			for(int k=0; k<this->dim_cube[2]; k++) {
				model_g[k]+= model_function(k+1, params_line[3*n_g], params_line[1+3*n_g], params_line[2+3*n_g]);
			}

/*
			std::string p_str_number = std::to_string(n_g);
			char const *p_name_model_g = p_str_number.c_str();
			char str_number[100];//220
			strcpy (str_number,"gaussian_number_");
			strcat (str_number,p_name_model_g);
			puts (str_number);
			plt::named_plot(str_number, x, model_g);
//			std::cout<<"str_number = "<<str_number<<std::endl;
*/

			plt::plot(x, model_g, color[n_g]);


		}
//			std::cout<< dim_cube[2] <<std::endl;
			for(int i=0; i<nb_gaussian; i++) {
				for(int k=0; k<this->dim_cube[2]; k++) {
					model[k]+= model_function(k+1, params_line[3*i], params_line[1+3*i], params_line[2+3*i]);
				}
			}


			for(int k=0; k<this->dim_cube[2]; k++) {
				cube_line[k]= cube_mean[ind_x][ind_y][k];
			}




			// Plot line from given x and y data. Color is selected automatically.
			plt::plot(x, cube_line, "r--");//color[0]);
			plt::plot(x, model, "b--");//color[0]);

			// Plot line from given x and y data. Color is selected automatically.
			plt::named_plot("data", x, cube_line);
			plt::named_plot("model", x, model);

			plt::xlim(0, this->dim_cube[2]);

			// Add graph title
			plt::title("Model vs Data Plot");
			// Enable legend.
			plt::legend();

		//params[0].size()

			std::string s_dimensions_1 = std::to_string(params[0].size());
			char const *pchar_dimensions_1 = s_dimensions_1.c_str();
			std::string s_dimensions_2 = std::to_string(params[0][0].size());
			char const *pchar_dimensions_2 = s_dimensions_2.c_str();
			std::string s_x = std::to_string(ind_x);
			char const *pchar_x = s_x.c_str();
			std::string s_y = std::to_string(ind_y);
			char const *pchar_y = s_y.c_str();
			char const *p_some_string = some_string.c_str();

			char str[100];//220
			strcpy (str,"./plot_dim_");
			strcat (str,pchar_dimensions_1);
			strcat (str,"_by_");
			strcat (str,pchar_dimensions_2);
			strcat (str,"_position_");
			strcat (str,pchar_x);
			strcat (str,"_by_");
			strcat (str,pchar_y);
			strcat (str,"_note_");
			strcat (str,p_some_string);
			strcat (str,".png");
			puts (str);

			// save figure
			std::cout << "Saving result to " << str << std::endl;;
			plt::save(str);
			plt::clf();
			//plt::show();

		}
	}
}

template<typename T>
T hypercube<T>::model_function(int x, T a, T m, T s) {

	return a*exp(-pow((T(x)-m),2.) / (2.*pow(s,2.)));

}

template<typename T>
void hypercube<T>::display_result(std::vector<std::vector<std::vector<T>>> &params,int rang, int n_gauss_i)
{
	std::vector<std::vector<T>> model(this->dim_cube[0],std::vector<T>(this->dim_cube[1],0.));

	for(int p(0); p<this->dim_cube[0]; p++) {
		for(int j(0); j<this->dim_cube[1]; j++) {
			for(int i(0); i<n_gauss_i; i++) {
		model[p][j]+= model_function(rang+1, params[3*i][j][p], params[1+3*i][j][p], params[2+3*i][j][p]);
			}
		}
	}

	std::vector<float> z_cube(this->dim_cube[0]*this->dim_cube[1],0.);

	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){ 
			z_cube[i*this->dim_cube[1]+j] = model[i][j];//this->cube[i][j][rang];
		}
	}

	const float* zptr = &(z_cube[0]);
	const int colors = 1;
	plt::clf();
	plt::title("Vue en coupe de model");//	plt::title("Vue en coupe de data");
	plt::imshow(zptr, this->dim_cube[0], this->dim_cube[1], colors);
	plt::save("imshow_result.png");
//	plt::show();
	std::cout << "Result saved to 'imshow_result.png'.\n"<<std::endl;
}

template<typename T>
void hypercube<T>::display_result_and_data(std::vector<std::vector<std::vector<T>>> &params,int rang, int n_gauss_i, bool dat_or_not)
{

	std::cout << "fonction affichage : params.size() : " << params.size() << " , " << params[0].size() << " , " << params[0][0].size() <<  std::endl;

	std::vector<std::vector<T>> model(this->dim_data[0],std::vector<T>(this->dim_data[1],0.));

//	std::cout<< "début de la fonction affichage"<<std::endl;

	for(int p(0); p<this->dim_data[0]; p++) {
		for(int j(0); j<this->dim_data[1]; j++) {
			for(int i(0); i<n_gauss_i; i++) {
		model[p][j]+= model_function(rang+1, params[3*i][j][p], params[1+3*i][j][p], params[2+3*i][j][p]);
			}
		}
	}

//	std::cout<< "milieu du début de la fonction affichage"<<std::endl;

	int offset_0 = (this->dim_cube[0]-this->dim_data[0])/2;
	int offset_1 = (this->dim_cube[1]-this->dim_data[1])/2;

	if (dat_or_not){
		offset_0=0;
		offset_1=0;
	}

	std::vector<float> z_model(this->dim_cube[0]*this->dim_cube[1],0.);


	for(int i=0;i<this->dim_data[0];i++){
		for(int j=0;j<this->dim_data[1];j++){
			z_model[(i+offset_1)*this->dim_cube[1]+j+offset_0] = model[i][j];//this->cube[i][j][rang];
		}
	}



	std::vector<float> z_cube(this->dim_cube[0]*this->dim_cube[1],0.);

	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){
			z_cube[i*this->dim_cube[1]+j] = this->cube[i][j][rang]; //index transpose, ordre : i,j 
		}
	}

	const float* zptr_model = &(z_model[0]);
	const float* zptr_cube = &(z_cube[0]);
	const int colors = 1;

	std::string s = std::to_string(rang);
	char const *pchar = s.c_str();

	char str[220];
	strcpy (str,"imshow_data_vs_model_");
	strcat (str,pchar);
	strcat (str,".png");
	puts (str);


	plt::clf();
//	std::cout<<" DEBUG "<<std::endl;
//	plt::title();

//	plt::suptitle("Données                      Modèle");

	plt::subplot(1, 2, 2);
		plt::imshow(zptr_model, this->dim_cube[0], this->dim_cube[1], colors);
//			plt::title("Modèle");
	plt::subplot(1, 2, 1);
		plt::imshow(zptr_cube, this->dim_cube[0], this->dim_cube[1], colors);
//			plt::title("Cube hyperspectral");
	plt::save(str);
//	plt::show();
	std::cout << "Result saved to "<<str<<".\n"<<std::endl;

}

template<typename T>
void hypercube<T>::display_2_gaussiennes(std::vector<std::vector<std::vector<T>>> &params,int rang, int n_gauss_i, int n1, int n2)
{
/*
	std::vector<std::vector<T>> model(this->dim_cube[0],std::vector<T>(this->dim_cube[1],0.));

	for(int p(0); p<this->dim_cube[0]; p++) {
		for(int j(0); j<this->dim_cube[1]; j++) {
		model[p][j]+= model_function(rang+1, params[3*0][j][p], params[1+3*0][j][p], params[2+3*0][j][p]) + model_function(rang+1, params[3*1][j][p], params[1+3*1][j][p], params[2+3*1][j][p]);
		}
	}

	std::vector<float> z_model(this->dim_cube[0]*this->dim_cube[1],0.);

	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){
			z_model[i*this->dim_cube[1]+j] = model[i][j];//this->cube[i][j][rang];
		}
	}
*/

	std::vector<std::vector<T>> model_premiere_gaussienne(this->dim_data[0],std::vector<T>(this->dim_data[1],0.));
	for(int p(0); p<this->dim_data[0]; p++) {
		for(int j(0); j<this->dim_data[1]; j++) {
		model_premiere_gaussienne[p][j]+= model_function(rang+1, params[3*n1][j][p], params[1+3*n1][j][p], params[2+3*n1][j][p]);
		}
	}

	std::vector<std::vector<T>> model_deuxieme_gaussienne(this->dim_data[0],std::vector<T>(this->dim_data[1],0.));
	for(int p(0); p<this->dim_data[0]; p++) {
		for(int j(0); j<this->dim_data[1]; j++) {
		model_deuxieme_gaussienne[p][j]+= model_function(rang+1, params[3*n2][j][p], params[1+3*n2][j][p], params[2+3*n2][j][p]);// + model_function(rang+1, params[3*n3][j][p], params[1+3*n3][j][p], params[2+3*n3][j][p]);
		}
	}

	std::vector<float> z_model_premiere_gaussienne(this->dim_data[0]*this->dim_data[1],0.);
	for(int i=0;i<this->dim_data[0];i++){
		for(int j=0;j<this->dim_data[1];j++){
			z_model_premiere_gaussienne[i*this->dim_data[1]+j] = model_premiere_gaussienne[i][j];//this->cube[i][j][rang];
		}
	}

	std::vector<float> z_model_deuxieme_gaussienne(this->dim_data[0]*this->dim_data[1],0.);
	for(int i=0;i<this->dim_data[0];i++){
		for(int j=0;j<this->dim_data[1];j++){
			z_model_deuxieme_gaussienne[i*this->dim_data[1]+j] = model_deuxieme_gaussienne[i][j];//this->cube[i][j][rang];
		}
	}
	// max_val = *std::max_element(z_model_deuxieme_gaussienne.begin(), z_model_deuxieme_gaussienne.end());

	std::vector<float> z_model(this->dim_data[0]*this->dim_data[1],0.);
	for(int i=0;i<this->dim_data[0];i++){
		for(int j=0;j<this->dim_data[1];j++){
			z_model[i*this->dim_data[1]+j] = z_model_premiere_gaussienne[i*this->dim_data[1]+j] + z_model_deuxieme_gaussienne[i*this->dim_data[1]+j];//this->cube[i][j][rang];
		}
	}

//	z_model_premiere_gaussienne[0]=max_val;
/*
	std::vector<float> z_gauss_1(this->dim_cube[0]*this->dim_cube[1],0.);
	std::vector<float> z_gauss_2(this->dim_cube[0]*this->dim_cube[1],0.);

	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){
			z_gauss_1[i*this->dim_cube[1]+j] = this->cube[i][j][rang];
		}
	}
	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){
			z_gauss_1[i*this->dim_cube[1]+j] = this->cube[i][j][rang];
		}
	}
*/
	const float* zptr_model = &(z_model[0]);
	const float* zptr_gauss_1 = &(z_model_premiere_gaussienne[0]);
	const float* zptr_gauss_2 = &(z_model_deuxieme_gaussienne[0]);

	const int colors = 1;

	std::string s = std::to_string(rang);
	char const *pchar = s.c_str();
	std::string s1 = std::to_string(n1);
	char const *pchar_1 = s1.c_str();
	std::string s2 = std::to_string(n2);
	char const *pchar_2 = s2.c_str();
	std::string s3 = std::to_string(n_gauss_i);
	char const *pchar_3 = s3.c_str();

	char str[220];
	strcpy (str,"decomp_two_gauss_");
	strcat (str,pchar);
	strcat (str,"_n1_");
	strcat (str,pchar_1);
	strcat (str,"_n2_");
	strcat (str,pchar_2);
	strcat (str,"_number_");
	strcat (str,pchar_3);
	strcat (str,".png");
	puts (str);


	plt::clf();
//	std::cout<<" DEBUG "<<std::endl;
//	plt::title();

//	plt::suptitle("Données                      Modèle");

	plt::subplot(3, 1, 1);
		plt::imshow(zptr_model, this->dim_data[0], this->dim_data[1], colors);
//			plt::title("Modèle");
	plt::subplot(3, 1, 2);
		plt::imshow(zptr_gauss_1, this->dim_data[0], this->dim_data[1], colors);

	plt::subplot(3, 1, 3);
		plt::imshow(zptr_gauss_2, this->dim_data[0], this->dim_data[1], colors);

//			plt::title("Cube hyperspectral");
	plt::save(str);
//	plt::show();
	std::cout << "Result saved to "<<str<<".\n"<<std::endl;
}

template<typename T>
void hypercube<T>::display_2_gaussiennes_par_par_par(std::vector<std::vector<std::vector<T>>> &params,int rang, int n_gauss_i, int n1, int n2)
{
/*
	std::vector<std::vector<T>> model(this->dim_cube[0],std::vector<T>(this->dim_cube[1],0.));

	for(int p(0); p<this->dim_cube[0]; p++) {
		for(int j(0); j<this->dim_cube[1]; j++) {
		model[p][j]+= model_function(rang+1, params[3*0][j][p], params[1+3*0][j][p], params[2+3*0][j][p]) + model_function(rang+1, params[3*1][j][p], params[1+3*1][j][p], params[2+3*1][j][p]);
		}
	}

	std::vector<float> z_model(this->dim_cube[0]*this->dim_cube[1],0.);

	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){
			z_model[i*this->dim_cube[1]+j] = model[i][j];//this->cube[i][j][rang];
		}
	}
*/

	std::vector<std::vector<T>> model_premiere_gaussienne(this->dim_data[0],std::vector<T>(this->dim_data[1],0.));
	for(int p(0); p<this->dim_data[0]; p++) {
		for(int j(0); j<this->dim_data[1]; j++) {
		model_premiere_gaussienne[p][j]+= model_function(rang+1, params[n1][j][p], params[n1][j][p], params[n1][j][p]);
		}
	}

	std::vector<std::vector<T>> model_deuxieme_gaussienne(this->dim_data[0],std::vector<T>(this->dim_data[1],0.));
	for(int p(0); p<this->dim_data[0]; p++) {
		for(int j(0); j<this->dim_data[1]; j++) {
		model_deuxieme_gaussienne[p][j]+= model_function(rang+1, params[n2][j][p], params[n2][j][p], params[n2][j][p]);// + model_function(rang+1, params[3*n3][j][p], params[1+3*n3][j][p], params[2+3*n3][j][p]);
		}
	}

	std::vector<float> z_model_premiere_gaussienne(this->dim_data[0]*this->dim_data[1],0.);
	for(int i=0;i<this->dim_data[0];i++){
		for(int j=0;j<this->dim_data[1];j++){
			z_model_premiere_gaussienne[i*this->dim_data[1]+j] = model_premiere_gaussienne[i][j];//this->cube[i][j][rang];
		}
	}

	std::vector<float> z_model_deuxieme_gaussienne(this->dim_data[0]*this->dim_data[1],0.);
	for(int i=0;i<this->dim_data[0];i++){
		for(int j=0;j<this->dim_data[1];j++){
			z_model_deuxieme_gaussienne[i*this->dim_data[1]+j] = model_deuxieme_gaussienne[i][j];//this->cube[i][j][rang];
		}
	}
	// max_val = *std::max_element(z_model_deuxieme_gaussienne.begin(), z_model_deuxieme_gaussienne.end());

	std::vector<float> z_model(this->dim_data[0]*this->dim_data[1],0.);
	for(int i=0;i<this->dim_data[0];i++){
		for(int j=0;j<this->dim_data[1];j++){
			z_model[i*this->dim_data[1]+j] = z_model_premiere_gaussienne[i*this->dim_data[1]+j] + z_model_deuxieme_gaussienne[i*this->dim_data[1]+j];//this->cube[i][j][rang];
		}
	}

//	z_model_premiere_gaussienne[0]=max_val;
/*
	std::vector<float> z_gauss_1(this->dim_cube[0]*this->dim_cube[1],0.);
	std::vector<float> z_gauss_2(this->dim_cube[0]*this->dim_cube[1],0.);

	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){
			z_gauss_1[i*this->dim_cube[1]+j] = this->cube[i][j][rang];
		}
	}
	for(int i=0;i<this->dim_cube[0];i++){
		for(int j=0;j<this->dim_cube[1];j++){
			z_gauss_1[i*this->dim_cube[1]+j] = this->cube[i][j][rang];
		}
	}
*/
	const float* zptr_model = &(z_model[0]);
	const float* zptr_gauss_1 = &(z_model_premiere_gaussienne[0]);
	const float* zptr_gauss_2 = &(z_model_deuxieme_gaussienne[0]);

	const int colors = 1;

	std::string s = std::to_string(rang);
	char const *pchar = s.c_str();
	std::string s1 = std::to_string(n1);
	char const *pchar_1 = s1.c_str();
	std::string s2 = std::to_string(n2);
	char const *pchar_2 = s2.c_str();
	std::string s3 = std::to_string(n_gauss_i);
	char const *pchar_3 = s3.c_str();

	char str[220];
	strcpy (str,"par_par_par_decomp_two_gauss_");
	strcat (str,pchar);
	strcat (str,"_n1_");
	strcat (str,pchar_1);
	strcat (str,"_n2_");
	strcat (str,pchar_2);
	strcat (str,"_number_");
	strcat (str,pchar_3);
	strcat (str,".png");
	puts (str);


	plt::clf();
//	std::cout<<" DEBUG "<<std::endl;
//	plt::title();

//	plt::suptitle("Données                      Modèle");

	plt::subplot(3, 1, 1);
		plt::imshow(zptr_model, this->dim_data[0], this->dim_data[1], colors);
//			plt::title("Modèle");
	plt::subplot(3, 1, 2);
		plt::imshow(zptr_gauss_1, this->dim_data[0], this->dim_data[1], colors);

	plt::subplot(3, 1, 3);
		plt::imshow(zptr_gauss_2, this->dim_data[0], this->dim_data[1], colors);

//			plt::title("Cube hyperspectral");
	plt::save(str);
//	plt::show();
	std::cout << "Result saved to "<<str<<".\n"<<std::endl;
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

template<typename T>
void hypercube<T>::display_avec_et_sans_regu(std::vector<std::vector<std::vector<T>>> &params, int num_gauss, int num_par,int plot_numero)
{
	int dim_0 = params[0][0].size();
	int dim_1 = params[0].size();

	std::vector<float> z_show(dim_0*dim_1,0.);

	for(int i=0;i<dim_0;i++){
		for(int j=0;j<dim_1;j++){
			z_show[i*dim_1+j] = params[num_par+num_gauss*3][j][i];
		}
	}

	const float* zptr_cube = &(z_show[0]);
	const int colors = 1;

	std::string s = std::to_string(plot_numero);
	char const *pchar = s.c_str();


	char str[220];
	strcpy (str,"essai_plot_param_numero_");
	strcat (str,pchar);
	strcat (str,".png");
	puts (str);


	plt::clf();

	plt::imshow(zptr_cube, dim_0, dim_1, colors);
//			plt::title("Cube hyperspectral");
	plt::save(str);
//	plt::show();
	std::cout << "Result saved to "<<str<<".\n"<<std::endl;
}

template<typename T>
void hypercube<T>::simple_plot_through_regu(std::vector<std::vector<std::vector<T>>> &params, int num_gauss, int num_par,int plot_numero)
{
	int dim_0 = params[0][0].size();
	int dim_1 = params[0].size();

	std::vector<float> z_show(dim_0*dim_1,0.);

	for(int i=0;i<dim_0;i++){
		for(int j=0;j<dim_1;j++){
			z_show[i*dim_1+j] = params[num_par+num_gauss*3][j][i];
		}
	}

	const float* zptr_cube = &(z_show[0]);
	const int colors = 1;

	std::string s_1 = std::to_string(num_par+num_gauss*3);
	char const *pchar_1 = s_1.c_str();
	std::string s_2 = std::to_string(plot_numero);
	char const *pchar_2 = s_2.c_str();


	char str[220];
	strcpy (str,"plot_param_through_regu_resolution_");
	strcat (str,pchar_2);
	strcat (str,"_param_numero_");
	strcat (str,pchar_1);
	strcat (str,".png");
	puts (str);


	plt::clf();

	plt::imshow(zptr_cube, dim_0, dim_1, colors);
//			plt::title("Cube hyperspectral");
	plt::save(str);
//	plt::show();
	std::cout << "Result saved to "<<str<<".\n"<<std::endl;
}


template <typename T> 
void hypercube<T>::save_result(std::vector<std::vector<std::vector<T>>>& grid_params, parameters<T>& M) {

  std::cout<<"dim_data[0] = "<<dim_data[0]<<std::endl;
  std::cout<<"dim_data[1] = "<<dim_data[1]<<std::endl;
  std::cout<<"dim_data[2] = "<<dim_data[2]<<std::endl;
  
//  std::string space = "           "; 

  std::ofstream myfile;
  myfile.open(M.fileout);//, std::ofstream::out | std::ofstream::trunc);
  myfile << std::setprecision(18);
  myfile << "# \n";
  myfile << "# ______Parameters_____\n";
  myfile << "# n_gauss = "<< M.n_gauss<<"\n";
//  myfile << "# n_gauss_add = "<< M.n_gauss<<"\n";
  myfile << "# lambda_amp = "<< M.lambda_amp<<"\n";
  myfile << "# lambda_mu = "<< M.lambda_mu<<"\n";
  myfile << "# lambda_sig = "<< M.lambda_sig<<"\n";
  myfile << "# lambda_var_amp = "<< M.lambda_var_amp<<"\n";
  myfile << "# lambda_var_mu = "<< M.lambda_var_mu<<"\n";
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
  myfile << "# lambda_var_amp = "<< M.lambda_var_amp<<"\n";
  myfile << "# lambda_var_mu = "<< M.lambda_var_mu<<"\n";
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




#endif