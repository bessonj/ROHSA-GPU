sudo apt-get install python-matplotlib python-numpy python2.7-dev
tar -xvf default_data.tar.xz
mkdir ccfits_and_cfitsio
cd ccfits_and_cfitsio
wget https://heasarc.gsfc.nasa.gov/fitsio/CCfits/CCfits-2.5.tar.gz
wget heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3450.tar.gz
tar zxvf CCfits-2.5.tar.gz
tar zxvf cfitsio3450.tar.gz
cd cfitsio
./configure --prefix=/usr
make
sudo make install
cd ..
cd CCfits
./configure --with-cfitsio=/usr/lib --prefix=/usr
make 
sudo make install
cd ..
cd ..
rm -r ccfits_and_cfitsio
