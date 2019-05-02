#ifndef DEF_PARSE
#define DEF_PARSE

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

class Parse
{
	public:

	Parse();
	int dim2nside();
	std::vector<int> get_dimensions_from_fits();
	void brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2);
	void multiresolution(int nside);
	int get_binary_from_fits();
	void get_vector_from_binary(std::vector<std::vector<std::vector<double>>> &z);
	void show_data();
	std::vector<int> get_dim_data() const;
	int get_nside() const;
	std::vector<int> get_dim_cube() const;

	std::vector<std::vector<std::vector<double>>> use_rubbish_dat_file();
	std::vector<std::vector<std::vector<double>>> reshape_up();
	std::vector<std::vector<std::vector<double>>> copy_cube_data();


	std::vector<std::vector<std::vector<double>>> cube;
	std::vector<std::vector<std::vector<double>>> data;

	private:


	std::string filename;
	std::vector<int> dim_data;
	std::vector<int> dim_cube;
  	int nside;
};

#endif


