//#ifndef DEF_FLATARRAY
//#define DEF_FLATARRAY

//#include <iostream>
//#include <vector>

template <typename T>
class FlatArray {       // The class

  public:             // Access specifier
   ~FlatArray();
//   template<class T> 
    FlatArray(int* taille_tab);
   int length;
   int* taille_tableau;
   double* tab_p;

   int product(int* X, int X_length);
   int combinaison_kd_to_1d(int* tab_x_y);

//   template<class T>
    T vec_acc(int* tab_x_y);

//   template <class T>
    void vec_add(int* tab_x_y, T val);







};

template<typename T>
FlatArray<T>::FlatArray(int* taille_tab)
{
	this->length = sizeof(*taille_tab);///sizeof(taille_tab[0]);
	this->taille_tableau = taille_tab;

	this->tab_p = NULL;
	this->tab_p = (T*)malloc(product(this->taille_tableau, this->length)*sizeof(T));

//	std::cout<<"vérification taille : "<<length<<std::endl;

}

template <typename T>
FlatArray<T>::~FlatArray()
{

//	free(this->tab_p);

}

template <typename T>
int FlatArray<T>::product(int* X, int X_length)
{
	int product_value = 1;
	for(int p=0; p< X_length; p++)
	{
		product_value *= X[p];
	}
	return product_value;
}

//cache optimization on the last index (index x) of tab_x_y = (a,b,...,x)
template <typename T>
int FlatArray<T>::combinaison_kd_to_1d(int* tab_x_y)
{
	int product =1;
	int indice = tab_x_y[this->length-1];
	for(int p=this->length-2; p>=0; p--)
        {
		product *= this->taille_tableau[p+1];
		indice += product*tab_x_y[p];
        }
	return indice;
}

//template<class T>
template <typename T>
T FlatArray<T>::vec_acc(int* tab_x_y)
{
        return this->tab_p[combinaison_kd_to_1d(tab_x_y)];
}

//template <class T>
template <typename T>
void FlatArray<T>::vec_add(int* tab_x_y, T val)
{
	this->tab_p[combinaison_kd_to_1d(tab_x_y)]= val;
}



//#endif
