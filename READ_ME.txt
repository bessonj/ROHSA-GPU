À compléter ...



Purpose and function : 

This program is built onto the original ROHSA code. Its aim is to read the FITS file, generate a raw data file so that gaussian decomposition algorithm can be performed using GPU [...]


Requirements and warning :

Some particular tools are required to read the FITS file (GHIGLS), such as CCfits which uses cfitsio ; a basic script called "run_first.sh" is provided in order to install these tools. 

	/!\ ==> You shall run this one as root, it will fill the /usr directory (lib and include especially) 

	sudo sh run_first.sh   or   sudo ./run_first.sh

	a "chmod +x run_first.sh" might be needed...

If your system is not compatible please install "cfitsio" first:

	https://heasarc.gsfc.nasa.gov/docs/software/fitsio/

Then you will be able to install "CCfits" :

	https://heasarc.gsfc.nasa.gov/fitsio/CCfits/


Compile and execute :

Using gcc you can compile and run the program using : 

	g++ main.cpp -o swap -L/usr/lib/x86_64-linux-gnu/ -std=c++11 -lCCfits -lcfitsio && ./swap

[... pourrait changer à cause de la partie GPU ...]
