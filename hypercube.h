#ifndef DEF_HYPERCUBE
#define DEF_HYPERCUBE

#include <iostream>
#include <stdio.h>
#include <cmath>
//#include <math.h>
#include <string>
#include <fstream>
#include <valarray>
#include <CCfits/CCfits>
#include <vector>

// mettre des const à la fin des déclarations si on ne modifie pas l'objet i.e. les attributs

class hypercube
{
	public:

	hypercube();
	hypercube(std::string filename);

	int dim2nside(); //obtenir les dimensions 2^n
	std::vector<int> get_dimensions_from_fits();
	void brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2);
	void multiresolution(int nside); 
	int get_binary_from_fits(std::string &filename);
	void get_vector_from_binary(std::vector<std::vector<std::vector<double>>> &z);
	void show_data(); 
	std::vector<int> get_dim_data() const;
	int get_nside() const;
	std::vector<int> get_dim_cube() const;
	std::vector<std::vector<std::vector<double>>> use_dat_file();
	std::vector<std::vector<std::vector<double>>> reshape_up();


	std::vector<std::vector<std::vector<double>>> cube; //data format 2^n
	std::vector<std::vector<std::vector<double>>> data; //data brut
	std::vector<int> dim_data;
	std::vector<int> dim_cube;
	int nside;

	std::string filename; 
};






#endif
