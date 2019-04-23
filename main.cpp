#include "Parse.h"
#include "Tool.h"


int main()
{
        Parse file; //Créée data depuis le FITS en faisant une copie dans un fichier brut 

	file.multiresolution(5); //Multirésolution pour nside = 5 (multiple de 35, la longueur native du FITS)	

	return 0;
}
