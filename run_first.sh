mkdir ccfits_and_cfitsio
cd ccfits_and_cfitsio
wget https://heasarc.gsfc.nasa.gov/fitsio/CCfits/CCfits-2.5.tar.gz
wget heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio3450.tar.gz
tar zxvf CCfits-2.5.tar.gz
tar zxvf cfitsio3450.tar.gz
cd cfitsio
sudo ./configure --prefix=/usr
make
sudo make install
cd ..
cd CCfits
sudo ./configure --with-cfitsio=/usr --prefix=/usr
make 
sudo make install
cd ..
cd ..
sudo rm -r ccfits_and_cfitsio
