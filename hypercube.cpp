#include "hypercube.h"
#include <omp.h>
#include <cmath>
#include <iostream>
#include "./matplotlib-cpp/matplotlibcpp.h"

namespace plt = matplotlibcpp;
//using namespace std;

using namespace CCfits;


// mettre des const à la fin des déclarations si on ne modifie pas l'objet i.e. les attributs

hypercube::hypercube(model &M, int indice_debut, int indice_fin)
{
	this->indice_debut= indice_debut;
	this->indice_fin = indice_fin;
	if(M.file_type_fits){
	//	get_dimensions_from_fits();
		get_binary_from_fits(); // WARNING 
		get_vector_from_binary(this->data);
		this->nside = dim2nside()-1;
std::cout<<" BOUCLE "<<std::endl;
	}
	if(M.file_type_dat){
		this->data = use_dat_file();
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

/*	for(int k(0); k<dim_cube[2]; k++)
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

hypercube::hypercube(model &M)
{
	if(M.file_type_fits){
//		get_dimensions_from_fits();
		get_binary_from_fits(); // WARNING 
		get_vector_from_binary(this->data);
		this->nside = dim2nside()-1;
	}
	if(M.file_type_dat){
		this->data = use_dat_file();
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

}

hypercube::hypercube() //dummy constructor for initialization of an hypercube object
{

}

std::vector<std::vector<std::vector<double>>> hypercube::use_dat_file()
{
   	int x,y,z;
	double v;

	std::ifstream fichier("./GHIGLS_DFN_Tb.dat");

	fichier >> z >> x >> y;

	dim_data[2]=z;
	dim_data[1]=y;
	dim_data[0]=x;

	std::vector<std::vector<std::vector<double>>> data_(dim_data[0],std::vector<std::vector<double>>(dim_data[1],std::vector<double>(dim_data[2], 0.)));

	while(!fichier.std::ios::eof())
	{
   		fichier >> z >> y >> x >> v;
		data_[x][y][z] = v;
   	}

	return data_;
}

std::vector<int> hypercube::get_dim_data()
{
	dim_data_v.vector::push_back(dim_data[0]);
	dim_data_v.vector::push_back(dim_data[1]);
	dim_data_v.vector::push_back(dim_data[2]);
	return dim_data_v;
}


int hypercube::get_nside() const
{
	return nside;
}

std::vector<int> hypercube::get_dim_cube()
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

int hypercube::dim2nside()
{
	return std::max( 0, std::max(int(ceil( log(double(dim_data[0]))/log(2.))), int(ceil( log(double(dim_data[1]))/log(2.))))  ) ;  
}


std::vector<std::vector<std::vector<double>>> hypercube::reshape_up()
{
	std::vector<std::vector<std::vector<double>>> cube_(dim_cube[0],std::vector<std::vector<double>>(dim_cube[1],std::vector<double>(dim_cube[2])));

	for(int i(0); i< dim_cube[0]; i++)
	{
		for(int j(0); j<dim_cube[1]; j++)
		{
			for(int k(0); k<dim_data[2]; k++)
			{
				cube_[i][j][k]= data[i][j][k];
			}
		}
	}
	return cube_;
}

std::vector<std::vector<std::vector<double>>> hypercube::reshape_up(int borne_inf, int borne_sup)
{
/*	int center_x = (this->data)[0].size() / 2;
	int center_y = (this->data)[0][0].size() / 2;

	int indice_debut_x = center_x - dim_cube[0]/2+1;
	int indice_fin_x = indice_debut_x + dim_cube[0];//
	int indice_debut_y = center_y - dim_cube[1]/2+1;
	int indice_fin_y = indice_debut_y + dim_cube[1];//
*/
	std::vector<std::vector<std::vector<double>>> cube_(dim_cube[0],std::vector<std::vector<double>>(dim_cube[1],std::vector<double>(dim_cube[2])));

	for(int i(0); i< dim_cube[0]; i++)
	{
		for(int j(0); j<dim_cube[1]; j++)
		{
			for(int k(0); k<dim_cube[2]; k++)
			{
				cube_[i][j][k]= data[i][j][borne_inf+k];
			}
		}
	}

	return cube_;
}

void hypercube::brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2)
{

	for (int k(0); k<length1; k++)
	{
		for (int i(0); i<length2; i++)
		{
			std::cout << "__résultat["<<i<<"]["<<k<<"]["<<0<<"]= " << z[i][k][0] << std::endl;
		}
	}
}


int hypercube::get_binary_from_fits(){

	std::auto_ptr<FITS> pInfile(new FITS("./GHIGLS_DFN_Tb.fits",Read,true));

        PHDU& image = pInfile->pHDU();
	std::valarray<double> contents;
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

	std::vector <double> x;
	std::vector <std::vector<double>> y;
	std::vector <std::vector<std::vector<double>>> z;

	std::ofstream objetfichier;
 	objetfichier.open("./data.raw", std::ios::out | std::ofstream::binary ); //on ouvre le fichier en ecriture
	if (objetfichier.bad()) //permet de tester si le fichier s'est ouvert sans probleme
		std::cout<<"ERREUR À L'OUVERTURE DU FICHIER RAW AVANT ÉCRITURE"<< std::endl;

	int n(sizeof(double) * contents.size());

	objetfichier.write((char*)&contents[0], n);

	objetfichier.close();

	return n;
}


void hypercube::get_vector_from_binary(std::vector<std::vector<std::vector<double>>> &z)
{
   	int filesize = dim_data[0]*dim_data[1]*dim_data[2];

   	std::ifstream is("./data.raw", std::ifstream::binary);

   	std::cout<<"taille :"<<filesize<<std::endl;

   	const size_t count = filesize;
   	std::vector<double> vec(count);
   	is.read(reinterpret_cast<char*>(&vec[0]), count*sizeof(double));
   	is.close();

	std::vector <std::vector<double>> y;
	std::vector <double> x;
	int compteur(0);

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


}


void hypercube::display_cube(int rang)
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

void hypercube::display_data(int rang)
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

void hypercube::display(std::vector<std::vector<std::vector<double>>> &tab, int rang)
{
	int dim_[3];
	dim_[2]=tab.size();
	dim_[1]=tab[0].size();
	dim_[0]=tab[0][0].size();
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

void hypercube::plot_line(std::vector<std::vector<std::vector<double>>> &params, int ind_x, int ind_y, int n_gauss_i) {

	std::vector<double> model(this->dim_cube[2],0.);
	std::vector<double> cube_line(this->dim_cube[2],0.);
	std::vector<double> params_line(3*n_gauss_i,0.);

	std::cout<< dim_cube[2] <<std::endl;
	for(int i=0; i<params_line.size(); i++) {
		params_line[i]=params[i][ind_y][ind_x];
	}


	for(int i(0); i<n_gauss_i; i++) {
		for(int k=0; k<this->dim_cube[2]; k++) {
			model[k]+= model_function(k+1, params_line[3*i], params_line[1+3*i], params_line[2+3*i]);
		}
	}

	for(int k=0; k<this->dim_cube[2]; k++) {
		cube_line[k]= cube[ind_x][ind_y][k];
	}



    std::vector<double> x(this->dim_cube[2]);
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

    // save figure
    const char* filename = "./plot.png";
    std::cout << "Saving result to " << filename << std::endl;;
    plt::save(filename);
}

double hypercube::model_function(int x, double a, double m, double s) {

	return a*exp(-pow((double(x)-m),2.) / (2.*pow(s,2.)));

}

void hypercube::display_result(std::vector<std::vector<std::vector<double>>> &params,int rang, int n_gauss_i)
{
	std::vector<std::vector<double>> model(this->dim_cube[0],std::vector<double>(this->dim_cube[1],0.));


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
	std::cout << "Result saved to 'imshow_result.png'.\n"<<std::endl;
}
