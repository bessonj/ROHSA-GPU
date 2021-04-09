https://www.cita.utoronto.ca/GHIGLS/FITS/




I. Instructions before use :

	In order to satisfy the requirements below, a unix script is given :
	    install_CCfits.sh
	(you may run it as root/sudo; in some cases chmod +x *.sh could be required)

	Requirements :

	  -  cfitsio
	  -  CCfits

	  And the following packages :

	    -  python-matplotlib 
	    -  python-numpy 
	    -  python2.7-dev

	You may also manually check whether or not cmake and libomp-dev (openMP) are installed.

II. Compiling and launching the program :

	A cmake file is written in the root directory, you can get the Makefile with :

		cmake .

	Then,

		make 

	You will get a binary file, ROHSA-GPU :

		./ROHSA-GPU

II. Instructions for use (WARNING) :

	parameters.txt needs to be modified with respect to the hypercube file

	You may put a at least a .fits or .dat file and specify true or false for the criteria below

	You will get pictures of the results with the binary file of the results.

	Commentary :

