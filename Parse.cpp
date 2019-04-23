#include "Parse.h"

using namespace CCfits;



// mettre des const à la fin des déclarations si on ne modifie pas l'objet i.e. les attributs

Parse::Parse()
{
	filename = "./GHIGLS.fits";
	tab=get_dimensions_from_fits();
	//tab.std::push_back(tab[i]);
	
	dim_x = tab[0];
	dim_y = tab[1];
	dim_v = tab[2];

	get_binary_from_fits();
	get_vector_from_binary(data);

}

Parse::Parse(std::string filename_user)
{
	filename = "./"+filename_user;
	tab=get_dimensions_from_fits();
	dim_x = tab[0];
	dim_y = tab[1];
	dim_v = tab[2];

	get_binary_from_fits();
	get_vector_from_binary(data);


}

/*
Parse::~Parse()
{
//	faire des .clear() comme Cube.clear();
}
*/

// Compute nside value from \(dim_y\) and \(dim_x\) 
int Parse::dim2nside()
{
	return std::max( 0, std::max(int(ceil( log(double(dim_y))/log(2.))), int(ceil( log(double(dim_x))/log(2.))))  ) ;  
}



std::vector<int> Parse::get_dimensions_from_fits()
{

       	std::auto_ptr<FITS> pInfile(new FITS(filename,Read,true)); 

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
	tab[0]=ax1;//x
	tab[1]=ax2;//y
	tab[2]=ax3;//lambda

	return tab;
}


void Parse::brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2)
{

	for (int j(0); j<depth; j++){

	for (int k(0); k<length1; k++){

	for (int i(0); i<length2; i++){

	std::cout << " résultat["<<j<<"]["<<k<<"]["<<i<<"]= " << z[j][k][i] << std::endl;
	}
	}
	}
}


int Parse::get_binary_from_fits(){

	std::auto_ptr<FITS> pInfile(new FITS("./GHIGLS.fits",Read,true));

        PHDU& image = pInfile->pHDU();
	std::valarray<double> contents;
        image.readAllKeys();

        image.read(contents);

        // this doesn't print the data, just header info.
        // std::cout << image << std::endl;

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


void Parse::get_vector_from_binary(std::vector<std::vector<std::vector<double>>> &z)
{

   	int filesize = dim_x*dim_y*dim_v;

   	std::ifstream is("./data_test.raw", std::ifstream::binary);

   	std::cout<<"taille :"<<filesize<<std::endl;

   	const size_t count = filesize;
   	std::vector<double> vec(count);
   	is.read(reinterpret_cast<char*>(&vec[0]), count*sizeof(double));
   	is.close();

	std::vector <std::vector<double>> y;
	std::vector <double> x;
	int compteur(0);
	for (int j(0); j<dim_v; j++)
	{
		for (int k(j*dim_y); k<(j+1)*dim_y; k++)
		{
			for (int i(k*dim_x); i<(k+1)*dim_x; i++)
			{
				x.vector::push_back(vec[i]);
//				std::cout<<"i= "<<i<<" j= "<< j <<" vec[i]= "<<vec[i]<<std::endl;
			}
			y.vector::push_back(x);
			x.clear();
		}
		z.vector::push_back(y);
		y.clear();
	}
}

void Parse::show(const std::vector<std::vector<std::vector<double>>> &z)
{
	for (int j(0); j<dim_v; j++)
	{
		for (int k(0); k<dim_y; k++)
		{
			for (int i(0); i<dim_x; i++)
			{
				std::cout << " résultat["<<j<<"]["<<k<<"]["<<i<<"]= " << z[j][k][i] << std::endl;
			}
		}
	}
}


void Parse::multiresolution(int nside)	//pour faire passer data en argument : const std::vector<std::vector<std::vector<double>>> &data
{	
	std::vector<int> dim = get_dimensions_from_fits();
	if (dim_x%nside!=0 or dim_y%nside!=0)
	{
		std::cout<< "MULTIRESOLUTION ERROR : LENGTH/NSIDE IS NOT AN INTEGER, THE FITS FILE NEEDS TO BE RESIZED" <<std::endl;
	}

	std::vector<std::vector<std::vector<double>>> cube(dim_v, std::vector<std::vector<double>> (nside, std::vector<double> (nside)));

	double avg(0.),S;

	for(int h=0; h<nside; h++)
	{
		for(int t=0; t<nside; t++)
		{
			for(int p=0; p<dim_v; p++)
			{
				S=0.;

				for (int m = dim_y/nside*t; m< dim_y/nside*(t+1); m++)
				{
					for (int n= dim_x/nside*h; n < dim_x/nside*(h+1); n++)
					{
						S+=data[p][m][n];
					}
				}
				avg = S/(dim_x*dim_y)/pow(nside,2);
				cube[p][t][h]=avg;
			}
		}
	}

	brute_show(cube,dim_v,nside,nside); //À ENLEVER, PERMET DE VÉRIFIER LE RÉSULTAT
	}



