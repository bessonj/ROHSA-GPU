#include "Parse.h"
#include "Reshape_data.h"
#include "Gaussian.h"




int main()
{

        Parse file; //Création de data depuis le FITS en faisant une copie dans un fichier brut 

//	Reshape_data var(file);


	std::cout << "-_-_-_résultat = " << file.cube[0][0][0] << std::endl;


	std::vector<int> dim_cube(3);
	dim_cube = file.get_dim_cube();


/*
	for (int i(0); i<dim_cube[0]; i++)
	{
		for (int k(0); k<dim_cube[1]; k++)
		{
			for (int j(0); j<dim_cube[2]; j++)
			{

				std::cout << "résultat["<<j<<"]["<<k<<"]["<<i<<"]= " << file.cube[i][k][j] << std::endl;
			}
		}
	}
*/
	int nside = file.get_nside();
	std::cout << " nside = " << nside <<std::endl;
	file.multiresolution(nside); //Multirésolution pour nside = 5 (multiple de dim_v, la longueur native du FITS)	

	return 0;
}

