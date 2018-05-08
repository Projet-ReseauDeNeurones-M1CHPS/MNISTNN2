/// \file

#include <iostream>
#include <stdio.h>
#include <fstream>
#include <Eigen/Dense>
#include "Perceptron.hpp"
#include <cstdlib>
#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>

#define BLACK '#'
#define WHITE '_'

/// \brief Permet de savoir si dans quelle mode lire les metadonnees d'un fichier IDX (low-endien ou big-endian)
/// \param f Pointeur sur la chaine de caractere contenant le nom du fichier IDX
/// \return le mode selon lequel lire le fichier IDX (0 ou 1)
/// \warning teste uniquement sur des processeurs Intel!!!
int idxEndianness(char* f)
{
  int n = 0;
  char* r = nullptr;
  r = (char*)&n;
  std::ifstream in(f, std::ifstream::in);
  //On lit les 4 premiers octets
  in.read(r, 4);
  if(n != 2049 && n!= 2051)
  {
      return 0;
  }
  else
  {
      return 1;
  }
}

/// \brief Lire les metadonnees d'un fichier IDX (fonction recursive)
/// \param in Flux du fichier IDX a lire
/// \param mode Mode Mode pour lire le fichier selon l'endianness
/// \param n Reference vers la variable dans laquelle mettre la prochaine metadonnee a lire
/// \param args References vers les prochaines variable dans laquelle mettre les metadonnees suivantes a lire
/// \warning teste uniquement sur des processeurs Intel!!!
template<typename... Args>
void idxMeta(std::ifstream& in, int mode, std::size_t& n, Args& ...args)
{
  char* r = nullptr;
  r = (char*)&n;
  //On lit les 4 premiers octets
  in.read(r, 4);
  //Sur Intel, on les inverses
  if(mode==0)
  {
    char tmp;
    tmp = r[0];
    r[0] = r[3];
    r[3] = tmp;
    tmp = r[1];
    r[1] = r[2];
    r[2] = tmp;
  }
  idxMeta(in, mode, args...);
}

/// \brief Dernier appel recursif de idxMeta
/// \warning teste uniquement sur des processeurs Intel!!!
template<>
void idxMeta(std::ifstream& in, int mode, std::size_t& n)
{
  char* r = nullptr;
  r = (char*)&n;
  //On lit les 4 premiers octets
  in.read(r, 4);
  //Sur Intel, on les inverses
  if(mode == 0)
  {
    char tmp;
    tmp = r[0];
    r[0] = r[3];
    r[3] = tmp;
    tmp = r[1];
    r[1] = r[2];
    r[2] = tmp;
  }
}

/// \brief On charge dans une entree-sortie de reseau le prochain element qui se trouve dans un fichier IDX
/// \param fin Flux du fichier IDX a lire
/// \param vout Reference vers l'entree-sortie de reseau dans laquelle on charge le prochain element du fichier
/// \param size Taille du prochain element a lire
/// \return Rien
/// \tparam t Type de l'entree-sortie de reseau dans laquelle on charge le prochain element du fichier IDX
/// \tparam SIZE Taille de l'entree-sortie de reseau dans laquelle on charge le prochain element du fichier IDX
/// \warning teste uniquement sur des processeurs Intel!!!
template<typename t, int SIZE>
void nextIdx(std::ifstream& fin, neuralnetwork::InOut<t, SIZE>& vout, std::size_t size)
{
  for (unsigned int i = 0; i < size; ++i)
  {
    unsigned char r;
    fin.read((char*)&r, 1);
    vout(i) = r;   
  }
}

/// \brief Conversion d'un label entier en label vecteur pour la classification de caractere numerique
/// \param in Label entier
/// \param out Label vecteur en sortie
/// \return Rien
/// \tparam t Type d'entree du reseau (a priori un double)
/// \tparam INPUT_SIZE Taille d'entree du reseau
template<typename t, int INPUT_SIZE>
void labelToVector(int in, neuralnetwork::InOut<t, INPUT_SIZE>& out)
{
    out << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    out(in) = 1;
}

/// \brief Affiche une entree-sortie de reseau contenant une images de MNIST (principalement utile au debugage)
/// \param rows Nombre de lignes de l'image (a priori 28)
/// \param cols Nombre de colonnes de l'image (a priori 28)
/// \param v Entree-sortie de reseau a afficher
/// \return Rien
/// \tparam t Type de l'entree-sortie de reseau a afficher
/// \tparam SIZE Taille de l'entree-sortie a afficher
template<typename t, int SIZE>
void printInOut(std::size_t rows, std::size_t cols, neuralnetwork::InOut<t, SIZE> v)
{
  for(int i = 0; i < rows; ++i)
  {
    for(int j = 0; j < cols; ++j)
    {
      if (v(i*rows + j) != 0)
      {
        std::cout << BLACK;
      }
      else
      {
        std::cout << WHITE;
      }
    }
    std::cout << std::endl;
  }
}

/// \brief Entrainement du reseau pour la classification de caracteres
/// \param pathImages Pointeur vers la chaine de caractere contenant le nom de fichier des images d'entrainement
/// \param pathLabels Pointeur vers la chaine de caractere contenant le nom de fichier des labels d'entrainement
/// \param net Reseau a entrainer
/// \param step Pas d'apprentissage
/// \param nbIter Nombre d'iterations a effectuer sur l'ensemble des images d'entrainement
/// \param nbImages Nombre d'images a charger en memoire simultanement
/// \return -1 en cas de probleme d'allocation memoires (trop d'images a charger simultanement)
/// \tparam DEPTH Profondeur du reseau de neurones
template<std::size_t DEPTH>
int train(char* pathImages, char* pathLabels, neuralnetwork::Perceptron<DEPTH>& net, double step, std::size_t nbIter, std::size_t nbImages)
{

  std::size_t magicImTrain = 0, magicLabTrain = 0;
  std::size_t nbImagesTrain = 0;
  std::size_t rows = 0;
  std::size_t cols = 0;
  neuralnetwork::InOut<double, 784>* inIm = NULL;
  neuralnetwork::InOut<double, 1>* inLab = NULL; 
  neuralnetwork::InOut<double, 10>* inLabVect = NULL;
  int mode = idxEndianness(pathLabels);
  //Si on a 0 en nombre d'exemples a charger en memoire on chargera tous les exemples de la base de donnees
  if(nbImages == 0)
  {
    std::ifstream flabels(pathLabels, std::ifstream::in);
    idxMeta(flabels, mode, magicLabTrain, nbImagesTrain);
    nbImages = nbImagesTrain;
    flabels.close();
  }
  
  inIm = new neuralnetwork::InOut<double, 784> [nbImages];
  inLab = new neuralnetwork::InOut<double, 1> [nbImages];
  inLabVect = new neuralnetwork::InOut<double, 10> [nbImages];
  if(inIm == NULL || inLab == NULL || inLabVect == NULL)
  {
    return -1;
  }

  //On entraine autant de fois que voulu
  for(unsigned int iter = 0; iter < nbIter; ++iter)
  {
    std::cout << "Iteration [" << iter + 1 << "/" << nbIter << "]" << std::endl;
    std::ifstream fimages(pathImages, std::ifstream::in);
    std::ifstream flabels(pathLabels, std::ifstream::in);
    idxMeta(flabels, mode, magicLabTrain, nbImagesTrain);
    idxMeta(fimages, mode, magicImTrain, nbImagesTrain, rows, cols);
    //Tant que l'on a pas fait tous les exemples
    for(unsigned int n = 0; n < nbImagesTrain; n+=nbImages)
    { 
      std::cout << "\tLot [" << n + 1 << " - " << n + nbImages << "/" << nbImagesTrain << "]" << std::endl;
      //On charge les exemples et les labels
      for(unsigned int i = 0; i < nbImages; ++i)
      {
        //On charge l'exemple et le label
        nextIdx(fimages, inIm[i], 784);
        nextIdx(flabels, inLab[i], 1);
        int valLab = inLab[i](0);
        //Conversion du label en vecteur
        labelToVector(valLab, inLabVect[i]);
      }
    //std::cout << "Iteration " << iter + 1 << ", entrainement avec les image de " << n << " a " << n+nbImages << "..." << std::endl;
    //On entraine sur le lot charge
    net.parallel_backpropagation(inIm, inLabVect, nbImages, step);
    //net.backpropagation(inIm, inLabVect, nbImages, step);
    }
    
    fimages.close();
    flabels.close();
  }

  delete [] inIm;
  delete [] inLab;
  delete [] inLabVect;
  
  return 0;
}
  
/// \brief Test du reseau pour la classification de caracteres
/// \param pathImages Pointeur vers la chaine de caractere contenant le nom de fichier des images de test
/// \param pathLabels Pointeur vers la chaine de caractere contenant le nom de fichier des labels de test
/// \param net Reseau a tester 
/// \return Pourcentage de reussite
/// \tparam DEPTH Profondeur du reseau de neurones
template<std::size_t DEPTH>
double test(char* pathImages, char* pathLabels, neuralnetwork::Perceptron<DEPTH>& net)
{
  std::size_t magicImTest = 0, magicLabTest = 0;
  std::size_t nbImagesTest = 0;
  std::size_t rows = 0;
  std::size_t cols = 0;
  neuralnetwork::InOut<double, 784> inImTest;
  neuralnetwork::InOut<double, 1> inLabTest;
  int mode = idxEndianness(pathLabels);  
  std::ifstream fimagesTest(pathImages, std::ifstream::in);
  std::ifstream flabelsTest(pathLabels, std::ifstream::in);
  
  idxMeta(flabelsTest, mode, magicLabTest, nbImagesTest);
  idxMeta(fimagesTest, mode, magicImTest, nbImagesTest, rows, cols);
  
  int count = 0;
  //Pour tous les exemples de tests
  for(int i = 0; i < 10000; ++i)
  {
    neuralnetwork::InOut<double, 10> out;
    neuralnetwork::InOut<double, 10> realOut;

    //On charge les exemples avec leurs labels
    nextIdx(fimagesTest, inImTest, 784);
    nextIdx(flabelsTest, inLabTest, 1);
    
    //On convertit le label en vecteur
    labelToVector(inLabTest(0), realOut);

    //On fait evaluer l'exemple par le reseau de neurones
    //net.feedForward(inImTest, out, false, false);
    net.parallel_feedForward(inImTest, out);
    
    //Pour chaque element i du vecteur de sortie, s'il est superieur a 0.5, on estime que le reseau a trouve que l'exemple est i - 1
    //Par exemple si le reseau renvoi [0.12, 0.23, 0.87, 0.34, 0.45, 0.47, 0.37, 0.28, 0.19 , 0.09] alors on interprete que le solution trouvee est [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] et donc que le nombre trouve est 2
    for(int j = 0; j < 10; ++j)
    {
      if(out(j) > 0.5)
      {
        out(j) = 1.0;
      }
      else
      {
        out(j) = 0.0;
      }
    }
    //Si la bonne solution est trouvee, alors on incremente le coompteur
    if(out == realOut)
    {
      ++count;
    }
  }
  
  //On retourne le pourcentage de reussite
  return count*100.0/10000.0;

}

/// \brief Itere un certain nombre de fois sur la base d'entrainement et apres chaque iteration test le reseau sur la base de test, le taux de reussite de chaque test est stocke dans un fichier
/// \param pathImTrain Pointeur vers la chaine de caractere contenant le nom de fichier des images d'entrainement
/// \param pathLabTrain Pointeur vers la chaine de caractere contenant le nom de fichier des labels d'entrainement
/// \param pathImTest Pointeur vers la chaine de caractere contenant le nom de fichier des images de test
/// \param pathLabTest Pointeur vers la chaine de caractere contenant le nom de fichier des labels de test
/// \param plot Pointeur vers la chaine de caractere contenant le nom de fichier dans lequel stocker les resultats
/// \param net Reseau a entrainer
/// \param step Pas d'apprentissage
/// \param nbIter Nombre d'iterations a effectuer sur l'ensemble des images d'entrainement
/// \param nbImages Nombre d'images a charger en memoire simultanement
/// \return Rien
/// \tparam DEPTH Profondeur du reseau de neurones
template<std::size_t DEPTH>
void bench(char* pathImTrain, char* pathLabTrain, char* pathImTest, char* pathLabTest, char* plot, neuralnetwork::Perceptron<DEPTH>& net, double step, std::size_t nbIter, std::size_t nbImages)
{
  std::ofstream fplot(plot, std::ofstream::out);
  for(int i = 0; i < nbIter; ++i)
  {
    train(pathImTrain, pathLabTrain, net, step, 1, nbImages);
    double resTest = test(pathImTest, pathLabTest, net);
    fplot << resTest << std::endl;
  }
}


int main()
{
  char pathImTrain [] = "../datas/imTrain";
  char pathLabTrain [] = "../datas/labTrain";
  char pathImTest [] = "../datas/imTest";
  char pathLabTest [] = "../datas/labTest";
 
  neuralnetwork::Perceptron<3>  net(
    -0.0001,
    0.0001,
    784, 
    200, neuralnetwork::activation::SIGMOID, neuralnetwork::activation::D_SIGMOID,
    200, neuralnetwork::activation::SIGMOID, neuralnetwork::activation::D_SIGMOID,
    10, neuralnetwork::activation::SIGMOID, neuralnetwork::activation::D_SIGMOID    
  );
  net.initGraph(0.1, 4);
  train(pathImTrain, pathLabTrain,  net, 0.1, 35, 1000);
  double res = test(pathImTest, pathLabTest, net);
  std::cout << "Reussite de " << res << " pourcents apres entrainement" << std::endl;

  return 0;
}
