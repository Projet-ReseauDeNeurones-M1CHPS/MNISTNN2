/// \file

#ifndef PERCEPTRON_INCLUDED
#define PERCEPTRON_INCLUDED

#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>
#include <tbb/tbb.h>

namespace neuralnetwork
{
  
  enum InOutType
  {
    DYNAMIC = -1
  };
  
  /// \brief Les entrees et sorties du Perceptron sont un simple vecteur Eigen
  template<typename t, int INPUT_SIZE>
  using InOut = Eigen::Matrix<t, INPUT_SIZE, 1>;
  
  /// \brief On definit dans ce namespace quelques fonctions d'activations, on peut tout de meme en definir ailleurs l'important etant qu'elles aient un unique argument de type double et qu'elles renvoient un double
  namespace activation
  {
    double SIGMOID(double x)
    {
      return 1.0/(1 + exp(-x));
      
    }
    
    double D_SIGMOID(double x)
    {
      double res = exp(x)/pow(exp(x)+1, 2);
      if (res!=res)
      {
        res = 0.0001;
      }
      return res;
    }
    double AFFINE(double x)
    {
      return x;
    }
    double D_AFFINE(double x)
    {
      return 1.0;
    }
  }
 
  /// \brief Classe de noeud de propagation (calculs) du graphe de feed forward
  /// \class Forward_Node
  class Forward_Node
  {
    private:
      
      /// \brief Matrice a laquelle est affecte le noeud du graphe
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* weights;
      
      /// \brief Agregation en sortie
      Eigen::Matrix<double, Eigen::Dynamic, 1>* agreg_out;
      
      /// \brief Activation en sortie
      Eigen::Matrix<double, Eigen::Dynamic, 1>* act_out;
      
      /// \brief Activation en entree
      Eigen::Matrix<double, Eigen::Dynamic, 1>* act_in;
      
      /// \brief Fonction d'activation
      double (*f)(double);
      
      /// \brief Premiere ligne de la matrice dont s'occupe le noeud
      int first_row;
      
      /// \brief Premiere colonne de la matrice dont s'occupe le noeud (a priori 0)
      int first_col;
      
      /// \brief Nombre de lignes dont s'occupe le noeud
      int nb_rows;
      
      /// \brief Nombre de colonnes dont s'occupe le noeud (a priori la largeur de la matrice)
      int nb_cols;
        
    public:
      
      /// \brief Constructeur
      /// \param w Matrice a laquelle est affecte le noeud du graphe
      /// \param act Activation en entree
      /// \param agreg Agregation en sortie
      /// \param act2 Activation en sortie
      /// \param f_act Fonction d'activation
      /// \param f_row Premiere ligne de la matrice dont s'occupe le noeud
      /// \param f_col Premiere colonne de la matrice dont s'occupe le noeud (a priori 0)
      /// \param n_rows Nombre de lignes dont s'occupe le noeud
      /// \param n_cols Nombre de colonnes dont s'occupe le noeud (a priori la largeur de la matrice)
      Forward_Node
      (
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* w,
        Eigen::Matrix<double, Eigen::Dynamic, 1>* act,
        Eigen::Matrix<double, Eigen::Dynamic, 1>* agreg,
        Eigen::Matrix<double, Eigen::Dynamic, 1>* act2,
        double(*f_act)(double),
        int f_row,
        int f_col,
        int n_rows,
        int n_cols
      ):
      weights(w),
      agreg_out(agreg),
      act_out(act2),
      act_in(act),
      f(f_act),
      first_row(f_row),
      first_col(f_col),
      nb_rows(n_rows),
      nb_cols(n_cols)
      {}

      void operator()(tbb::flow::continue_msg m)
      {
        agreg_out->block(first_row, 0, nb_rows, 1) = (weights->block(first_row, first_col, nb_rows, nb_cols) * (*act_in));
        for(int i = first_row; i < first_row + nb_rows; ++i)
        {
          act_out->operator()(i) = f(agreg_out->operator()(i));
        }
      }
        
  };

  /// \brief Classe de noeud de retropropagation (calculs) du graphe de retropropagation
  /// \class Backward_Node
  class Backward_Node
  {
    private:
      
      /// \brief Matrice a laquelle est affecte le noeud du graphe
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* weights;
      
      /// \brief Agregation en entree
      Eigen::Matrix<double, Eigen::Dynamic, 1>* agreg_in;
      
      /// \brief Erreur en sortie
      Eigen::Matrix<double, Eigen::Dynamic, 1>* err_out;
      
      /// \brief Erreur en entree
      Eigen::Matrix<double, Eigen::Dynamic, 1>* err_in;
      
      /// \brief Derivee de la fonction d'activation
      double (*df)(double);
      
      /// \brief Premiere ligne de la matrice dont s'occupe le noeud
      int first_row;
      
      /// \brief Premiere colonne de la matrice dont s'occupe le noeud (a priori 0)
      int first_col;
      
      /// \brief Nombre de lignes dont s'occupe le noeud
      int nb_rows;
      
      /// \brief Nombre de colonnes dont s'occupe le noeud (a priori la largeur de la matrice)
      int nb_cols;
              
    public:
      /// \brief Constructeur
      /// \param w Transposee de la matrice a laquelle est affecte le noeud du graphe
      /// \param err Erreur en entree
      /// \param agreg Agregation en entree
      /// \param err2 Erreur en sortie
      /// \param df_act Derive de la fonction d'activation
      /// \param f_row Premiere ligne de la tranposee dont s'occupe le noeud
      /// \param f_col Premiere colonne de la transposee dont s'occupe le noeud (a priori 0)
      /// \param n_rows Nombre de lignes dont s'occupe le noeud
      /// \param n_cols Nombre de colonnes dont s'occupe le noeud (a priori la largeur de la tranposee)
      Backward_Node
      (
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* w,
        Eigen::Matrix<double, Eigen::Dynamic, 1>* err,
        Eigen::Matrix<double, Eigen::Dynamic, 1>* agreg,
        Eigen::Matrix<double, Eigen::Dynamic, 1>* err2,
        double(*df_act)(double),
        int f_row,
        int f_col,
        int n_rows,
        int n_cols
      ):
      weights(w),
      agreg_in(agreg),
      err_out(err2),
      err_in(err),
      df(df_act),
      first_row(f_row),
      first_col(f_col),
      nb_rows(n_rows),
      nb_cols(n_cols) 
      {}
      
      void operator()(tbb::flow::continue_msg m)
      {
        err_out->block(first_row, 0, nb_rows, 1) = weights->transpose().block(first_row, first_col, nb_rows, nb_cols)* (*err_in);
        for(int i = first_row; i < first_row + nb_rows; ++i)
        {
          err_out->operator()(i) = df(agreg_in->operator()(i)) * err_out->operator()(i);
        }
      }
  };
    
  /// \brief Classe de noeud de mise a jour (calculs) du graphe de retropropagation
  /// \class Update_Node
  class Update_Node
  {
    private:
      
      /// \brief Matrice a laquelle est affecte le noeud du graphe
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* weights;
      
      /// \brief Pas d'apprentissage
      double step;
      
      /// \brief Erreur en entree
      Eigen::Matrix<double, Eigen::Dynamic, 1>* err;
      
      /// \brief Activation en sortie
      Eigen::Matrix<double, Eigen::Dynamic, 1>* act;
      
      /// \brief Premiere ligne de la matrice dont s'occupe le noeud
      int first_row;
      
      /// \brief Premiere colonne de la matrice dont s'occupe le noeud (a priori 0)
      int first_col;
      
      /// \brief Nombre de lignes dont s'occupe le noeud
      int nb_rows;
      
      /// \brief Nombre de colonnes dont s'occupe le noeud (a priori la largeur de la matrice)
      int nb_cols;
        
    public:
      /// \brief Constructeur
      /// \param w Matrice a laquelle est affecte le noeud du graphe
      /// \param s Pas d'apprentissage
      /// \param e Erreur en entree
      /// \param a Activation en sortie
      /// \param f_act Fonction d'activation
      /// \param f_row Premiere ligne de la matrice dont s'occupe le noeud
      /// \param f_col Premiere colonne de la matrice dont s'occupe le noeud (a priori 0)
      /// \param n_rows Nombre de lignes dont s'occupe le noeud
      /// \param n_cols Nombre de colonnes dont s'occupe le noeud (a priori la largeur de la matrice)
      Update_Node
      (
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* w,
        double s,
        Eigen::Matrix<double, Eigen::Dynamic, 1>* e,
        Eigen::Matrix<double, Eigen::Dynamic, 1>* a,
        int f_row,
        int f_col,
        int n_rows,
        int n_cols
      ):
      weights(w),
      step(s),
      err(e),
      act(a),
      first_row(f_row),
      first_col(f_col),
      nb_rows(n_rows),
      nb_cols(n_cols) 
      {}
      
      void operator()(tbb::flow::continue_msg m)
      {
          for(int i = first_row; i < first_row + nb_rows; ++i)
          {
            weights->block(i, first_col, 1, nb_cols) += step * err->operator()(i) * act->transpose();
          }
      }
        
  }; 
  
  /// \brief Classe de noeud d'attente des graphes de feed forwards et de retropropagation
  /// \class Wait_Node 
  class Wait_Node
  {
    private:
        
    public:
      Wait_Node()
      {}
      
      void operator()(tbb::flow::continue_msg m)
      {
      }        
  };
  
  
  /// \brief Classe Perceptron
  /// \class Perceptron
  /// \tparam DEPTH nombres de couches du perceptron sans compter la couche d'entrée
  template<std::size_t DEPTH>
  class Perceptron
  {

    public:
      
      /// \brief Constructeur, les poids du reseau sont initialises entre deux bornes de maniere aleatoire, on definit egalement la taille des couches et leurs fonctions d'activation
      /// \param min Valeur minimal d'un poids du reseau
      /// \param max Valeur maximal d'un poids du reseau
      /// \param args Trois arguments pour definie une couche, le premier est le nombre de neurone de la couche, le deuxieme est est la fonction d'activation, le troisieme est la derivee de la fonction d'activation. Les trois premiers arguments definissent la premiere couche cachee, les trois suivants la deuxieme couche cachee, etc... la fonction d'activation et sa derivee doivent etre des pointeurs sur fonction qui prennent un double et retourne un double
      template <typename ...Args>
      Perceptron(double min, double max, Args... args);
      
      /// \brief Initialisation aleatoire des poids du reseau entre deux bornes
      /// \param min Valeur minimal d'un poids du reseau
      /// \param max Valeur maximal d'un poids du reseau
      /// \return Rien
      void initWeights(double min, double max);
      
      /// \brief Accesseur des poids du reseau
      /// \param couch La couche a laquelle appartient le neurone qui a le poids sur une de ses entrees
      /// \param neuron Le neurone de la couche
      /// \param previous Le neurone de la couche precedente
      /// \return Accesseur sur le poids
      double& weight(std::size_t couch, std::size_t neuron, std::size_t previous);
      
      /// \brief Nombre de poids de chaque neurone d'une couche
      /// \param couch La couche en question
      /// \return le nombre de poids pour un neurone de la couche en question
      std::size_t weights_count(std::size_t couch);
      
      /// \brief Profondeur du reseau
      /// \return La profondeur du reseau sans prendre en compte la couche d'entree
      std::size_t depth();
      
      /// \brief Nombre de neurones d'une couche donnee
      /// \param couch La couche en question
      /// \return Nombre de neurones de la couche en question
      std::size_t couch_size(std::size_t couch);
      
      //Eigen::Matrix<double, Eigen::Dynamic, 1> agreg(std::size_t couch);
      /// \brief Les dernieres agregations enregistrees pour une couche donnee (methode principalement utile au debugage)
      /// \param couch La couche en question
      /// \return une entree-sortie de neurones qui comporte toutes les dernieres agregation enregistrees de la couche en question
      neuralnetwork::InOut<double,neuralnetwork::InOutType::DYNAMIC> agreg(std::size_t couch);
      
      //Eigen::Matrix<double, Eigen::Dynamic, 1> activation(std::size_t couch);
      /// \brief Les dernieres activations enregistrees pour une couche donnee (methode principalement utile au debugage)
      /// \param couch La couche en question
      /// \return une entree-sortie de neurones qui comporte toutes les dernieres activations enregistrees de la couche en question
      neuralnetwork::InOut<double,neuralnetwork::InOutType::DYNAMIC> activation(std::size_t couch);
      
      
      /// \brief Ecrit la matrice des poids d'une couche, une ligne correspond a un neurone, les poids sont dans l'odre des entrees provenant de la couche precedente et les agregations et activations de cette couche (methode principalement utile au debugage)
      /// \param out reference sur le flux de sortie ou l'on veut afficher les informations
      /// \param i La couche en question
      /// \return Rien
      void printMatrixCouch(std::ostream& out, std::size_t i);
      
      /// \brief Evalue une entree avec le reseau
      /// \param in L'entree a evaluer
      /// \param out La sortie du reseau
      /// \param saveAgreg Si place a true, les agregations des differentes couches seront enregistrees
      /// \param saveActivation Si place a true, les activations des differentes couches seront enregistrees
      /// \tparam tin type d'entree du reseau (a priori un double)
      /// \tparam INPUT_SIZE taille d'entree du reseau
      /// \tparam tout type des sorties du reseau (a priori un double)
      /// \tparam OUTPUT_SIZE taille de sortie du reseau
      template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
      void feedForward(neuralnetwork::InOut<tin, INPUT_SIZE>& in, neuralnetwork::InOut<tout, OUTPUT_SIZE>& out, bool saveAgreg, bool saveActivation);
     
      /// \brief Applique la fonction d'activation sur l'agregation calulee d'une couche
      /// \param in L'aggregation calculee de la couche fi
      /// \param fi La couche en question
      /// \return Rien
      /// \tparam tin type de l'agreation (a priori un double)
      /// \tparam INPUT_SIZE taille de la couche
      /// \tparam tout type de l'activation (a priori un double)
      /// \tparam OUTPUT_SIZE taille de la couche
      /// \todo possibilite de se passer de in si feedForward enregistre systematiquement l'agregation
      template<typename t, int INPUT_SIZE>
      void applyActivation(neuralnetwork::InOut<t, INPUT_SIZE>& in, std::size_t fi);
      
      /// \brief Applique l'algorithme de retropropagation du gradient avec un jeu d'exemples afin d'entrainer le reseau
      /// \param ins Tableau des exemples d'entrees du reseau
      /// \param outs Tableau des sorties attendues pour chaque exemple de ins (organises dans le meme ordre)
      /// \param n Nombre d'exemples
      /// \param step Pas d'apprentissage
      /// \return Rien
      /// \tparam tin type des entrees (a priori double)
      /// \tparam INPUT_SIZE taille d'entree du reseau
      /// \tparam tout type des sorties du reseau (a priori double)
      /// \tparam OUTPUT_SIZE taille de sortie du reseau
      template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
      void backpropagation(neuralnetwork::InOut<tin, INPUT_SIZE>* ins, neuralnetwork::InOut<tout, OUTPUT_SIZE>* outs, std::size_t n, double step);
      
      /// \brief Initailise les graphes TBB pour le feed forward et la retropropagation
      /// \param step Pas d'apprentissage
      /// \param nb_threads En combien de lignes on decoupe chaque matrice et chaque transposee (meme nombre que les threads conseille)
      /// \return Rien
      void initGraph(double step, int nb_threads);
      
      /// \brief Evalue une entree avec le reseau avec TBB
      /// \param in L'entree a evaluer
      /// \param out La sortie du reseau
      /// \return Rien
      /// \tparam tin type d'entree du reseau (a priori un double)
      /// \tparam INPUT_SIZE taille d'entree du reseau
      /// \tparam tout type des sorties du reseau (a priori un double)
      /// \tparam OUTPUT_SIZE taille de sortie du reseau
      /// \return Rien
      template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
      void parallel_feedForward(neuralnetwork::InOut<tin, INPUT_SIZE>& in, neuralnetwork::InOut<tout, OUTPUT_SIZE>& out);
      
      /// \brief Applique l'algorithme de retropropagation du gradient avec un jeu d'exemples afin d'entrainer le reseau avec TBB
      /// \param ins Tableau des exemples d'entrees du reseau
      /// \param outs Tableau des sorties attendues pour chaque exemple de ins (organises dans le meme ordre)
      /// \param n Nombre d'exemples
      /// \param step Pas d'apprentissage
      /// \return Rien
      /// \tparam tin type des entrees (a priori double)
      /// \tparam INPUT_SIZE taille d'entree du reseau
      /// \tparam tout type des sorties du reseau (a priori double)
      /// \tparam OUTPUT_SIZE taille de sortie du reseau
      template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
      void parallel_backpropagation(neuralnetwork::InOut<tin, INPUT_SIZE>* ins, neuralnetwork::InOut<tout, OUTPUT_SIZE>* outs, std::size_t n, double step);
      
    private:
      
      /// \brief Methode utilitaire pour le constructeur, on definit egalement la taille des couches et leurs fonctions d'activation. Cette methode parcourt recursivement toutes les couches
      /// \param n Couche a parametrer
      /// \param input_size Nombres de poids pour les neurones de cette couche, egal au nombre de neurones de la couche precedente
      /// \param neurons Taille de cette couche en nombres de neurones
      /// \param f Fonction d'activation de cette couche
      /// \param df Derivee de la fonction d'activation de cette couche
      /// \param args Parametres pour les couches suivantes
      /// \return Rien
      template<typename... Args>
      void init(unsigned int n, int input_size, int neurons, double (*f)(double), double (*df)(double), Args ...args);
      
      /// \brief Dernier appel de init pour la derniere couche a parametrer
      void init(unsigned int n, int input_size, int neurons, double (*f)(double), double (*df)(double));
      
      /// \brief tableau des fonctions d'activations, une par couche
      double (*m_array_f[DEPTH])(double);
      
      /// \brief tableau des derivees des fonctions d'activations, une par couche
      double (*m_array_df[DEPTH])(double);
      
      /// \brief profondeur du reseau
      const std::size_t m_size = DEPTH;
     
      /// \brief tableau des matrices des couches. Une ligne represente les poids d'un neurone, l'element i de la ligne j represente le poids reliant le sortie du neurones i de la couche precedente avec le neurone j de la couche actuelle
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m_array_matrix[DEPTH];
      
      /// \brief tableau des agregations
      Eigen::Matrix<double, Eigen::Dynamic, 1> m_array_agreg[DEPTH];
      
      /// \brief tableau des activations
      Eigen::Matrix<double, Eigen::Dynamic, 1> m_array_act[DEPTH];
      
      /// \brief tableau des erreurs
      Eigen::Matrix<double, Eigen::Dynamic, 1> m_array_err[DEPTH];
      
      /// \brief Vecteur d'entree du graph
      Eigen::Matrix<double, Eigen::Dynamic, 1> m_first_graph_vector;
      
      /// \brief Noeuds des neurones du graphe pour le feed forward
      std::vector<std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>>> m_forward_neurons;
      
      /// \brief Noeuds d'attente des neurones du graphe pour le feed forward
      std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>> m_forward_waits;
      
      /// \brief Graphe du reseau pour le feed forward
      tbb::flow::graph m_graph_forward;

      /// \brief Noeuds des neurones du graphe pour la retropropagation
      std::vector<std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>>> m_backward_neurons;
      
      /// \brief Noeuds d'attente des neurones du graphe pour la retropropagation
      std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>> m_backward_waits;      
      
      /// \brief Graphe du reseau pour le backward
      tbb::flow::graph m_graph_backward;
      
      /// \brief Noeuds des neurones du graphe pour la mise a jour des poids
      std::vector<std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>>> m_update_neurons;

      
  };
}

/// \fn std::ostream& operator<<(std::ostream& out, neuralnetwork::Perceptron<DEPTH> p)
/// \brief Affiche les matrices de poids du reseau et les agregations et activations de chaque couche (methode principalement utile au debugage)
/// \param out reference sur le flux de sortie ou l'on veut afficher les informations
/// \param p Perceptron a afficher
/// \return out avec de nouvelles informations concernant le reseau dedans
/// \warning La fonction est incoherente, une partie des informations est ecrite dans out et l'autre dans std::cout. Cette fonction sert principalement au debugage
template<std::size_t DEPTH>
std::ostream& operator<<(std::ostream& out, neuralnetwork::Perceptron<DEPTH> p)
{
  out << DEPTH <<" couches" << std::endl;
  for(int i = 0; i < DEPTH; ++i)
  {
    out << "Couche " << i <<std::endl;
    p.printMatrixCouch(out, i);
    //getchar();
    out << std::endl;
    out << "Agregation:" << std::endl << p.agreg(i);
    //getchar();
    out << std::endl;
    out << "Activation:" << std::endl << p.activation(i);
    //getchar();
    out << std::endl;
  }

  return out;
}

template<std::size_t DEPTH>
template <typename ...Args>
neuralnetwork::Perceptron<DEPTH>::Perceptron(double min, double max, Args... args)
{
  init(0, args...);
  initWeights(min, max);
  m_first_graph_vector.resize(m_array_matrix[0].cols(), 1);
}


template<std::size_t DEPTH>
template<typename... Args>
void neuralnetwork::Perceptron<DEPTH>::init(unsigned int n, int input_size, int neurons, double (*f)(double), double (*df)(double), Args ...args)
{
  //On dimensionne la matrice representant la couche et les vecteurs d'agreations, d'activations et d'erreurs pour cette meme couche
  m_array_err[n].resize(neurons, 1);
  m_array_agreg[n].resize(neurons, 1);
  m_array_act[n].resize(neurons, 1);
  m_array_matrix[n].resize(neurons, input_size);

  //On donne la bonne fonction et la bonne derivee a la couche
  m_array_f[n] = f;
  m_array_df[n] = df;
  if(n < DEPTH )
  {
    init(n+1, neurons, args...);
  }
}

template<std::size_t DEPTH>
void neuralnetwork::Perceptron<DEPTH>::init(unsigned int n, int input_size, int neurons, double (*f)(double), double (*df)(double))
{
  m_array_agreg[n].resize(neurons, 1);
  m_array_act[n].resize(neurons, 1);
  m_array_err[n].resize(neurons, 1);
  m_array_matrix[n].resize(neurons, input_size);
  m_array_f[n] = f;
  m_array_df[n] = df;
}

template<std::size_t DEPTH>
void neuralnetwork::Perceptron<DEPTH>::initWeights(double min, double max)
{
  for(unsigned int i = 0; i < depth() ;++i)
  {
    for(unsigned int j = 0; j < couch_size(i); ++j)
    {
      for(unsigned int k = 0; k < weights_count(i); ++k)
      {
        weight(i, j, k) = (std::rand()/(static_cast<double>(RAND_MAX) + 1.0)) * (max - min) + min;
      }
    }
    
  }
}

template<std::size_t DEPTH>
void neuralnetwork::Perceptron<DEPTH>::initGraph(double step, int nb_threads)
{
  
  //Feed forward
  for(unsigned int i = 0; i < depth(); ++i)
  {
    int first_row = 0;
    tbb::flow::continue_node<tbb::flow::continue_msg> new_wait (m_graph_forward, Wait_Node());
    m_forward_waits.push_back(new_wait);
    std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>> new_couch;
    for(int j = 0; j < nb_threads; ++j)
    {
      int nb_rows;
      int nb_cols = m_array_matrix[i].cols();
      //Repartition des lignes entre les noeuds
      if(j < (m_array_matrix[i].rows() % nb_threads))
      {
        nb_rows = couch_size(i) / nb_threads + 1;
      }
      else
      {
        nb_rows = couch_size(i) / nb_threads;
      }
      int first_col = 0;
      
      //Pour n'importe quel couche hormis la premiere, l'entree est l'activation de la i-1eme couche, sinon c'est le vecteur d'entree
      if(i != 0)
      {
        tbb::flow::continue_node<tbb::flow::continue_msg> new_node (
                                  m_graph_forward,
                                  neuralnetwork::Forward_Node(
                                         &(m_array_matrix[i]),
                                         &(m_array_act[i - 1]),
                                         &(m_array_agreg[i]),
                                         &(m_array_act[i]),
                                         m_array_f[i],
                                         first_row,
                                         first_col,
                                         nb_rows,
                                         nb_cols
                                         
                                  )
                                  );
        new_couch.push_back(new_node);
      }
      else
      {
        tbb::flow::continue_node<tbb::flow::continue_msg> new_node (
                                  m_graph_forward,
                                  neuralnetwork::Forward_Node(
                                         &(m_array_matrix[i]),
                                         &(m_first_graph_vector),
                                         &(m_array_agreg[i]),
                                         &(m_array_act[i]),
                                         m_array_f[i],
                                         first_row,
                                         first_col,
                                         nb_rows,
                                         nb_cols
                                  )
                                  );
        new_couch.push_back(new_node);
      }
      first_row += nb_rows;
    }
    m_forward_neurons.push_back(new_couch);
  }
  
  //On lie les couches aux noeuds barrieres
  for(unsigned int i = 0; i < m_forward_waits.size(); ++i)
  {
    for(unsigned int j = 0; j < m_forward_neurons[i].size(); ++j)
    {
      tbb::flow::make_edge(m_forward_neurons[i][j], m_forward_waits[i]);
      if(i != 0)
      {
        tbb::flow::make_edge(m_forward_waits[i - 1], m_forward_neurons[i][j]);
      }
    }
  }

  //Backward
  for(unsigned int i = 1; i < DEPTH; ++i)
  {
    int first_row = 0;
    tbb::flow::continue_node<tbb::flow::continue_msg> new_wait (m_graph_backward, Wait_Node());
    m_backward_waits.push_back(new_wait);
    std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>> new_couch;
    for(int j = 0; j < nb_threads; ++j)
    {
        int nb_rows;
        int nb_cols = m_array_matrix[i].transpose().cols();
        //Repartition des lignes entre les noeuds
        if(j < (m_array_matrix[i].transpose().rows() % nb_threads))
        {
          nb_rows = m_array_matrix[i].transpose().rows() / nb_threads + 1;
        }
        else
        {
          nb_rows = m_array_matrix[i].transpose().rows() / nb_threads;
        }
        int first_col = 0;
        tbb::flow::continue_node<tbb::flow::continue_msg> new_node (
                                  m_graph_backward,
                                  neuralnetwork::Backward_Node(
                                         &(m_array_matrix[i]),
                                         &(m_array_err[i]),
                                         &(m_array_agreg[i - 1]),
                                         &(m_array_err[i - 1]),
                                         m_array_df[i - 1],
                                         first_row,
                                         first_col,
                                         nb_rows,
                                         nb_cols
                                  )
                                  );
        new_couch.push_back(new_node);
        first_row += nb_rows;
    }
    m_backward_neurons.push_back(new_couch);
  }
  
  //On lie les couches aux noeuds barrieres
  for(unsigned int i = 0; i < m_backward_neurons.size(); ++i)
  {
    for(unsigned int j = 0; j < m_backward_neurons[i].size(); ++j)
    {
      tbb::flow::make_edge(m_backward_neurons[i][j], m_backward_waits[i]);
      if(i != m_backward_neurons.size() - 1)
      {
        tbb::flow::make_edge(m_backward_waits[i + 1], m_backward_neurons[i][j]);
      }
    }
  }
  
  //Update
  for(unsigned int i = 0; i < DEPTH; ++i)
  {
    int first_row = 0;
    std::vector<tbb::flow::continue_node<tbb::flow::continue_msg>> new_couch;
    for(int j = 0; j < nb_threads; ++j)
      {
          int nb_rows;
          int nb_cols = m_array_matrix[i].cols();
          //Repartition des lignes entre les noeuds
          if(j < (m_array_matrix[i].rows() % nb_threads))
          {
            nb_rows = couch_size(i) / nb_threads + 1;
          }
          else
          {
            nb_rows = couch_size(i) / nb_threads;
          }
          int first_col = 0;
          //Pour n'importe quel couche hormis la premiere, l'entree est l'activation de la i-1eme couche, sinon c'est le vecteur d'entree
          if(i > 0)
          {
            tbb::flow::continue_node<tbb::flow::continue_msg> new_node (
                                  m_graph_backward,
                                  neuralnetwork::Update_Node(
                                         &(m_array_matrix[i]),
                                         step,
                                         &(m_array_err[i]),
                                         &(m_array_act[i - 1]),
                                         first_row,
                                         first_col,
                                         nb_rows,
                                         nb_cols
                                  )
                                  );
            new_couch.push_back(new_node);
          }
          else
          {
            tbb::flow::continue_node<tbb::flow::continue_msg> new_node (
                                  m_graph_backward,
                                  neuralnetwork::Update_Node(
                                         &(m_array_matrix[i]),
                                         step,
                                         &(m_array_err[i]),
                                         &(m_first_graph_vector),
                                         first_row,
                                         first_col,
                                         nb_rows,
                                         nb_cols
                                  )
                                  );
            new_couch.push_back(new_node);
          }
          first_row += nb_rows;
      }
      m_update_neurons.push_back(new_couch);
    }
    
    for(unsigned int i = 1; i < m_update_neurons.size(); ++i)
    {
      for(unsigned int j = 0; j < m_update_neurons[i].size(); ++j)
      {
        tbb::flow::make_edge(m_backward_waits[i - 1], m_update_neurons[i][j]);
      }
    }
    for(unsigned int j = 0; j < m_update_neurons[0].size(); ++j)
    {
        tbb::flow::make_edge(m_backward_waits[0], m_update_neurons[0][j]);
    }
  
}

template<std::size_t DEPTH>
double& neuralnetwork::Perceptron<DEPTH>::weight(std::size_t couch, std::size_t neuron, std::size_t previous)
{
  return m_array_matrix[couch](neuron, previous);
}
 
template<std::size_t DEPTH>
std::size_t neuralnetwork::Perceptron<DEPTH>::weights_count(std::size_t couch)
{
  return m_array_matrix[couch].cols();
}

template<std::size_t DEPTH>
std::size_t neuralnetwork::Perceptron<DEPTH>::depth()
{
  return m_size;
}

template<std::size_t DEPTH>
std::size_t neuralnetwork::Perceptron<DEPTH>::couch_size(std::size_t couch)
{
  return m_array_matrix[couch].rows();
}

template<std::size_t DEPTH>
void neuralnetwork::Perceptron<DEPTH>::printMatrixCouch(std::ostream& out, std::size_t i)
{
  std::cout << m_array_matrix[i];
}

template<std::size_t DEPTH>
Eigen::Matrix<double, Eigen::Dynamic, 1> neuralnetwork::Perceptron<DEPTH>::agreg(std::size_t couch)
{
  return m_array_agreg[couch];
}
  
template<std::size_t DEPTH>
Eigen::Matrix<double, Eigen::Dynamic, 1> neuralnetwork::Perceptron<DEPTH>::activation(std::size_t couch)
{
  return m_array_act[couch];
}

template<std::size_t DEPTH>
template<typename t, int INPUT_SIZE>
void neuralnetwork::Perceptron<DEPTH>::applyActivation(neuralnetwork::InOut<t, INPUT_SIZE>& in, std::size_t fi)
{
  for(int i = 0; i < in.size(); ++i)
  {
    in(i) = m_array_f[fi](in(i));
  }
}

template<std::size_t DEPTH>
template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
void neuralnetwork::Perceptron<DEPTH>::feedForward(neuralnetwork::InOut<tin, INPUT_SIZE>& in, neuralnetwork::InOut<tout, OUTPUT_SIZE>& out, bool saveAgreg, bool saveActivation)
{
  InOut<tin, neuralnetwork::InOutType::DYNAMIC> tmp;
  
  //feed forward pour le premiere couche
  tmp = m_array_matrix[0]*in;
  
  if(saveAgreg)
  {
    m_array_agreg[0] = tmp;
  }
  applyActivation(tmp, 0);
  if(saveActivation)
  {
    m_array_act[0] = tmp;
  }

  //feed forward pour les autres couches
  for(unsigned int i = 1; i < DEPTH; ++i)
  {
    tmp = m_array_matrix[i] * tmp;
    if(saveAgreg)
    {
      m_array_agreg[i] = tmp;
    }
    applyActivation(tmp, i);
    if(saveActivation)
    {
      m_array_act[i] = tmp;
    }
  }

  out = tmp;
}

template<std::size_t DEPTH>
template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
void neuralnetwork::Perceptron<DEPTH>::backpropagation(neuralnetwork::InOut<tin, INPUT_SIZE>* ins, neuralnetwork::InOut<tout, OUTPUT_SIZE>* outs, std::size_t n, double step)
{
  neuralnetwork::InOut<tout, Eigen::Dynamic> out;
  neuralnetwork::InOut<tout, Eigen::Dynamic> err;
  
  //Pour chaque exemple d'entrainement
  for(unsigned int i = 0; i < n; ++i)
  {
    //On evalue un exemple avec le reseau
    feedForward(ins[i], out, true, true);
    
    //On calcul l'erreur en sortie
    
    //difference entre la sortie desiree et la sortie attendue
    err = outs[i] - out;
    //df(agreg)*(sortie_desiree - sortie_attendue)
    for(unsigned int j = 0; j < err.size(); ++j)
    {
      err(j) = m_array_df[DEPTH - 1](m_array_agreg[DEPTH - 1](j)) * err(j);
    }
    m_array_err[DEPTH - 1] = err;

    //Erreur pour les autres couches
    //erreur(j - 1) = df(agreg) * matriceJ.transpose * erreur(j)
    for(unsigned int j = DEPTH - 1; j > 0; --j)
    {
      err = m_array_matrix[j].transpose() * m_array_err[j];
      for(int k = 0; k < err.size(); ++k)
      {
        err(k) = m_array_df[j - 1](m_array_agreg[j - 1](k)) * err(k);
      }
      m_array_err[j - 1] = err;
    }

    //Mise a jour des poids
    for(unsigned int j = 0; j < DEPTH; ++j)
    {
      for(unsigned int k = 0; k < couch_size(j); ++k)
      {
        for(unsigned int l = 0; l < weights_count(j); ++l)
        {
          if(j > 0)
          {
            weight(j, k, l) += step * m_array_err[j](k) * m_array_act[j - 1](l);
          }
          //Pour la premiere couche on prend le vecteur d'entree
          else
          {
            weight(j, k, l) += step * m_array_err[j](k) * ins[i](l);
          }
        }
      }
    }

  }
}

template<std::size_t DEPTH>
template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
void neuralnetwork::Perceptron<DEPTH>::parallel_feedForward(neuralnetwork::InOut<tin, INPUT_SIZE>& in, neuralnetwork::InOut<tout, OUTPUT_SIZE>& out)
{ 
  //On initialise le vecteur d'entree
  m_first_graph_vector.resize(in.size());
  for(int i = 0; i < in.size(); ++i)
  {
    m_first_graph_vector[i] = in[i];
  }
  
  //On lance le feed forward
  for(unsigned int i = 0; i < m_forward_neurons[0].size(); ++i)
  {
    m_forward_neurons[0][i].try_put(tbb::flow::continue_msg());
  }
  
  m_graph_forward.wait_for_all();
  out = m_array_act[DEPTH - 1];
}

template<std::size_t DEPTH>
template<typename tin, int INPUT_SIZE, typename tout, int OUTPUT_SIZE>
void neuralnetwork::Perceptron<DEPTH>::parallel_backpropagation(neuralnetwork::InOut<tin, INPUT_SIZE>* ins, neuralnetwork::InOut<tout, OUTPUT_SIZE>* outs, std::size_t n, double step)
{
  neuralnetwork::InOut<tout, Eigen::Dynamic> out;
  neuralnetwork::InOut<tout, Eigen::Dynamic> err;
  
  //Pour chaque exemple d'entrainement
  for(unsigned int i = 0; i < n; ++i)
  {
    //On evalue un exemple avec le reseau
    parallel_feedForward(ins[i], out);
    //feedForward(ins[i], out, true, true);
    
    //On calcul l'erreur en sortie
    //difference entre la sortie desiree et la sortie attendue
    err = outs[i] - out;
    for(unsigned int j = 0; j < err.size(); ++j)
    {
      err(j) = m_array_df[DEPTH - 1](m_array_agreg[DEPTH - 1](j)) * err(j);
    }
    m_array_err[DEPTH - 1] = err;
    
    //On lance la retropropagation
    for(unsigned int j = 0; j < m_backward_neurons.back().size(); ++j)
    {
      m_backward_neurons.back()[j].try_put(tbb::flow::continue_msg());
    }
    m_graph_backward.wait_for_all();
    
  }
}

#endif
