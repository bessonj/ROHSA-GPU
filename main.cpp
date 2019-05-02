#include "Parse.h"
#include "Reshape_data.h"
#include "Gaussian.h"




int main()
{

        Parse file; //Créée data depuis le FITS en faisant une copie dans un fichier brut 
//	Reshape_data var(file);

	std::vector<int> dim_cube = file.get_dim_cube();
	std::vector<int> dim_data = file.get_dim_data();


/*	afficher le cube issu de data redimensionné :

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
	file.multiresolution(2); //Multirésolution pour nside = 5 (multiple de 35, la longueur native du FITS)	

	return 0;
}

