#include <iostream>
#include <stdio.h>
#include <cmath>
#include <string>
#include <fstream>
#include <valarray>
#include <CCfits/CCfits>
#include <vector>


using namespace CCfits;


std::vector<int> get_dimensions_from_fits(){

       	std::auto_ptr<FITS> pInfile(new FITS("./GHIGLS_DFN_Tb.fits",Read,true));

        PHDU& image = pInfile->pHDU();
        std::valarray<double> contents;
        image.readAllKeys();

        image.read(contents);

        // this doesn't print the data, just header info.
        std::cout << image << std::endl;

        long ax1(image.axis(0));
        long ax2(image.axis(1));
        long ax3(image.axis(2));
	long ax4(image.axis(3));

	std::vector<int> tab(3);
	tab[0]=ax1;
	tab[1]=ax2;
	tab[2]=ax3;

	return tab;
}


void brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2){

	for (int j(0); j<depth; j++){

	for (int k(0); k<length1; k++){

	for (int i(0); i<length2; i++){

	std::cout << " résultat["<<j<<"]["<<k<<"]["<<i<<"]= " << z[j][k][i] << std::endl;
	}
	}
	}
}

void multiresolution(int nside, const std::vector<std::vector<std::vector<double>>> &data){

   std::vector<int> dim = get_dimensions_from_fits();
   int length1 = dim[0];
   int length2 = dim[1];

   int length = length1;
   int depth = dim[2];
   if (length%nside!=0){

std::cout<< "MULTIRESOLUTION ERROR : LENGTH/NSIDE IS NOT AN INTEGER, THE FITS FILE NEEDS TO BE RESIZED" <<std::endl;

	}

   std::vector<double> cube_1(nside);
   std::vector<std::vector<double>> cube_2(nside, cube_1);
   std::vector<std::vector<std::vector<double>>> cube(depth, cube_2);

   double avg(0.),S;

   for(int h=0; h<nside; h++){
   for(int t=0; t<nside; t++){
   for(int p=0; p<depth; p++){
   S=0.;

   for (int m = length/nside*t; m< length/nside*(t+1); m++)
   {
   for (int n= length/nside*h; n < length/nside*(h+1); n++)
   {
   S+=data[p][m][n];
   }
   }
   avg = S/pow(length/nside, 2);
   cube[p][t][h]=avg;
}
}
}
   brute_show(cube,depth,nside,nside);
}



int get_binary_from_fits(){

       std::auto_ptr<FITS> pInfile(new FITS("./GHIGLS_DFN_Tb.fits",Read,true));

        PHDU& image = pInfile->pHDU();
	std::valarray<double> contents;
        image.readAllKeys();

        image.read(contents);

        // this doesn't print the data, just header info.
        std::cout << image << std::endl;

        long ax1(image.axis(0));
        long ax2(image.axis(1));
        long ax3(image.axis(2));
	long ax4(image.axis(3));

	std::vector <double> x;
	std::vector <std::vector<double>> y;
	std::vector <std::vector<std::vector<double>>> z;

	std::ofstream objetfichier;
 	objetfichier.open("./data_test.raw", std::ios::out | std::ofstream::binary ); //on ouvre le fichier en ecriture
	if (objetfichier.bad()) //permet de tester si le fichier s'est ouvert sans probleme
		std::cout<<"ERREUR À L'OUVERTURE DU FICHIER RAW AVANT ÉCRITURE"<< std::endl;

	int n(sizeof(double) * contents.size());

	objetfichier.write((char*)&contents[0], n);

	objetfichier.close();

	return n;
}


void get_vector_from_binary(std::vector<std::vector<std::vector<double>>> &z){

   std::vector<int> dimensions = get_dimensions_from_fits();
   int ax1(dimensions[0]);
   int ax2(dimensions[1]);
   int ax3(dimensions[2]);
   int filesize = ax1*ax2*ax3;

   std::ifstream is("./data_test.raw", std::ifstream::binary);

   std::cout<<"taille :"<<filesize<<std::endl;

   const size_t count = filesize;
   std::vector<double> vec(count);
   is.read(reinterpret_cast<char*>(&vec[0]), count*sizeof(double));
   is.close();

	std::vector <std::vector<double>> y;
	std::vector <double> x;
	int compteur(0);
	for (int j(0); j<ax3; j++){

	for (int k(j*ax2); k<(j+1)*ax2; k++){

	for (int i(k*ax1); i<(k+1)*ax1; i++){

	x.vector::push_back(vec[i]);
//	std::cout<<"i= "<<i<<" j= "<< j <<" vec[i]= "<<vec[i]<<std::endl;
	}
	y.vector::push_back(x);
	x.clear();
	}
	z.vector::push_back(y);
	y.clear();
	}

}

void show(const std::vector<std::vector<std::vector<double>>> &z){

	std::vector<int> dim = get_dimensions_from_fits();

	for (int j(0); j<dim[2]; j++){

	for (int k(0); k<dim[1]; k++){

	for (int i(0); i<dim[0]; i++){

	std::cout << " résultat["<<j<<"]["<<k<<"]["<<i<<"]= " << z[j][k][i] << std::endl;
	}
	}
	}
}

int main()
{
	get_binary_from_fits();

	std::vector<std::vector<std::vector<double> > > data;

	get_vector_from_binary(data);

	multiresolution(1,data);
//	show(data);

	double S=0;

	for(int i(0); i<35; i++){
	for(int j(0); j<35; j++){
		S=S+data[489][i][j];
	}
	}
	std::cout<<S/(35*35);
	return 0;
}
