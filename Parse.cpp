#include "Parse.h"

using namespace CCfits;


// mettre des const à la fin des déclarations si on ne modifie pas l'objet i.e. les attributs
/*
Parse::Parse()
{
	data = use_rubbish_dat_file();
	nside = dim2nside();

	std::cout<<"__dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"__dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"__dim_data[2] = "<<dim_data[2]<<std::endl;

	dim_cube.push_back(int(pow(2.0,nside)));
	dim_cube.push_back(dim_cube[0]);
	dim_cube.push_back(dim_data[2]);

	std::cout<<"__dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"__dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"__dim_cube[2] = "<<dim_cube[2]<<std::endl;

	std::cout<<"__nside = "<<nside<<std::endl;

	cube = reshape_up();
}
*/

Parse::Parse()
{
	filename = "./GHIGLS.fits";
	dim_data = get_dimensions_from_fits();

//	get_binary_from_fits(); // WARNING 
	get_vector_from_binary(data);

	nside = dim2nside()-1;

	std::cout << "	nside = " << nside << std::endl;	

	std::cout<<"__dim_data[0] = "<<dim_data[0]<<std::endl;
	std::cout<<"__dim_data[1] = "<<dim_data[1]<<std::endl;
	std::cout<<"__dim_data[2] = "<<dim_data[2]<<std::endl;

	dim_cube.push_back(int(pow(2.0,nside)));
	dim_cube.push_back(dim_cube[0]);
	dim_cube.push_back(dim_data[2]);

	std::cout<<"__dim_cube[0] = "<<dim_cube[0]<<std::endl;
	std::cout<<"__dim_cube[1] = "<<dim_cube[1]<<std::endl;
	std::cout<<"__dim_cube[2] = "<<dim_cube[2]<<std::endl;

	std::cout<<"__nside = "<<nside<<std::endl;

	cube = reshape_up();

	std::cout << "-_-_-_résultat cube " << std::endl;
}

std::vector<std::vector<std::vector<double>>> Parse::use_rubbish_dat_file()
{
   	int x,y,z;
	double v;

	std::ifstream fichier("./GHIGLS_DFN_Tb.dat");

	fichier >> z >> x >> y;

	dim_data.vector::push_back(x);
	dim_data.vector::push_back(y);
	dim_data.vector::push_back(z);

	std::vector<std::vector<std::vector<double>>> data_(dim_data[0],std::vector<std::vector<double>>(dim_data[1],std::vector<double>(dim_data[2], 0.)));

	while(!fichier.std::ios::eof())
	{
   		fichier >> z >> y >> x >> v;
		data_[x][y][z] = v;
   	}

	return data_;
}

std::vector<int> Parse::get_dim_data() const
{
	return dim_data;
}


int Parse::get_nside() const
{
	return nside;
}

std::vector<int> Parse::get_dim_cube() const
{
	return dim_cube;
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
	return std::max( 0, std::max(int(ceil( log(double(dim_data[1]))/log(2.))), int(ceil( log(double(dim_data[0]))/log(2.))))  ) ;  
}


std::vector<std::vector<std::vector<double>>> Parse::reshape_up()
{
	std::vector<std::vector<std::vector<double>>> cube_(dim_cube[0],std::vector<std::vector<double>>(dim_cube[1],std::vector<double>(dim_cube[2])));

	int offset_w, offset_h;

	offset_w = int(0.5*(double(dim_cube[1])-double(dim_data[1])));
	offset_h = int(0.5*(double(dim_cube[0])-double(dim_data[0])));

	std::cout<< offset_w << "_____" << offset_h << std::endl;

	for(int i(1); i< dim_data[0]-2; i++)
	{
		for(int j(1); j<dim_data[1]-2; j++)
		{
			for(int k(0); k<dim_data[2]; k++)
			{
				std::cout << "cube["<<i<<","<<j<<","<<k<<"] = " <<data[i][j][k]<< std::endl;
				cube_[offset_h+i][offset_w+j][k]= data[i][j][k];
			}
		}
	}
	return cube_;
}


std::vector<std::vector<std::vector<double>>> Parse::copy_cube_data()
{
	std::vector<std::vector<std::vector<double>>> cube_(dim_data[0],std::vector<std::vector<double>>(dim_data[1], std::vector<double>(dim_data[2])));

	for (int j(0); j<dim_data[2]; j++)
	{
		for (int k(0); k<dim_data[1]; k++)
		{
			for (int i(0); i<dim_data[0]; i++)
			{
//				std::cout <<"__i,j,k = "<<i<<","<<j<<","<<k<< std::endl;	
				cube_[i][k][j] = data[i][k][j];
			}
		}
	}
	return cube_;
}


std::vector<int> Parse::get_dimensions_from_fits()
{

       	std::auto_ptr<FITS> pInfile(new FITS(filename,Read,true)); 

        PHDU& image = pInfile->pHDU();
        std::valarray<double> contents;
        image.readAllKeys();

        image.read(contents);

        // this doesn't print the data, just header info.
//        std::cout << image << std::endl;

        long ax1(image.axis(0));
        long ax2(image.axis(1));
        long ax3(image.axis(2));
	long ax4(image.axis(3));

	dim_data.vector::push_back(ax1);
	dim_data.vector::push_back(ax2);
	dim_data.vector::push_back(ax3);

	return dim_data;
}


void Parse::brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2)
{

	for (int j(0); j<depth; j++)
	{
		for (int k(0); k<length1; k++)
		{
			for (int i(0); i<length2; i++)
			{

				std::cout << "__résultat["<<i<<"]["<<k<<"]["<<j<<"]= " << z[i][k][j] << std::endl;
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
   	int filesize = dim_data[0]*dim_data[1]*dim_data[2];

   	std::ifstream is("./data_test.raw", std::ifstream::binary);

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
//				std::cout<<"i= "<<i<<" j= "<< j <<" vec[i]= "<<vec[i]<<std::endl;
			}
			y.vector::push_back(x);
			x.clear();
		}
		z.vector::push_back(y);
		y.clear();
	}
}

void Parse::show_data()
{
	for (int j(0); j<dim_data[0]; j++)
	{
		for (int k(0); k<dim_data[1]; k++)
		{
			for (int i(0); i<dim_data[2]; i++)
			{
				std::cout << " data["<<j<<"]["<<k<<"]["<<i<<"]= " << data[j][k][i] << std::endl;
			}
		}
	}
}


void Parse::multiresolution(int nside)
{	
	
	if (dim_cube[0]%nside!=0 or dim_cube[1]%nside!=0)
	{
		std::cout<< "MULTIRESOLUTION ERROR : LENGTH/NSIDE IS NOT AN INTEGER, THE CUBE NEEDS TO BE RESIZED" <<std::endl;
	}

	std::vector<std::vector<std::vector<double>>> subcube(nside, std::vector<std::vector<double>> (nside, std::vector<double>(dim_cube[2])));

	double avg(0.),S;

	for(int h=0; h<nside; h++)
	{
		for(int t=0; t<nside; t++)
		{
			for(int p=0; p<dim_cube[2]; p++)
			{
				S=0.;

				for (int m = dim_cube[1]/nside*t; m< dim_cube[1]/nside*(t+1); m++)
				{
					for (int n= dim_cube[0]/nside*h; n < dim_cube[0]/nside*(h+1); n++)
					{
//						std::cout <<"__n,m,p = "<<n<<","<<m<<","<<p<< std::endl;
						S+=cube[n][m][p];
					}
				}
				avg = S/((dim_cube[0]*dim_cube[1])/pow(nside,2));
				subcube[h][t][p]=avg;
			}
		}
	}

	brute_show(subcube,dim_cube[2],nside,nside); //À ENLEVER, PERMET DE VÉRIFIER LE RÉSULTAT
}

