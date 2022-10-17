#include "extraction.h"
#include <stdlib.h>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <fstream>

/*Para capturar (leer) el fichero de entrada "csv"
*(archivo separado por comas), se considera
*crear una función que trate un vector de
*vectores del tipo string. Es decir, cada
*linea representa un vector de strings,
*y la matriz completa (fichero.csv)
*representa un vector*/

/*La idea es almacenar el fichero en un puntero del tipo
*ifstream (https://cplusplus.com/reference/),
*es decir, se crea un buffer de la cadena del dichero csv.
*La variable dataset, ha da armarse cuando se haga el
*constructor de la clase, para ser recibida por argumentos
*de entrada de la función*/
std::vector<std::vector<std::string>> Extraction::LeerCSV(){
    //Se almacena el fichero en un buffer (variable temporal, "archivo")
    std::ifstream archivo(dataset);
    //Se crea un vector de vectores del tipo string
    std::vector<std::vector<std::string>>datosString;
    //Se busca recorrer cada linea del fichero y enviarla al vector de vectores
    std::string linea="";
    while(getline(archivo,linea)){
        std::vector<std::string>vector;
        //Se identifica cada elemento que compone el vector
        //Se divide o segmenta cada elemento con boost
        boost::algorithm::split(vector,linea,boost::is_any_of(delimitador));
        //Finalmente se ingresa al buffer temporañ
        datosString.push_back(vector);

    }
    //Se cierra el archivo csv
    archivo.close();
    //Se retorna el vector de vectores
    return datosString;
}

/*Segunda función miembro:
 * Pasar el vector de vectores del tipo string
 * a un objeto del tipo Eigen: para las
 * correspondientes operaciones*/
Eigen::MatrixXd Extraction::CSVtoEigen(
        std::vector<std::vector<std::string>>dataSet,
        int filas,
        int columnas){
    //Se revisa si tiene o no cabecera
    if(header==true){
        filas=filas-1;
    }
    Eigen::MatrixXd matriz(columnas,filas);
    //Se llena la matriz con los datos del dataset
    for(int i=0; i<filas; i++){
        for (int j=0;j<columnas;j++){
            //Se pasa a flotante el tipo string
            matriz(j,i)=atof(dataSet[i][j].c_str());
        }
    }
    //Se retorna la matriz transpuesta
    return matriz.transpose();
}

/*Función para extraer el promedio
 *Cuando no s está seguro de que tipo de dato va a retornar
 *entonces se hace uso de "auto nombrefuncion decltype" */

auto Extraction::Promedio(Eigen::MatrixXd datos)->decltype(datos.colwise().mean()){
    return datos.colwise().mean();
}

/*Función para extraer la desviación estandar*/
auto Extraction::Desviacion(Eigen::MatrixXd datos)->decltype(((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt()){
   return((datos.array().square().colwise().sum())/(datos.rows()-1)).sqrt();
}

/*Función para normalizar datos*/
/*Se retorna la matriz de datos normalizada
 * y la función recibe como argumentos la matriz de datos*/

Eigen::MatrixXd Extraction::Norm(Eigen::MatrixXd datos){
    /*Se calcula el promedio de los datos*/
    Eigen::MatrixXd diferenciaPromedio = datos.rowwise() - Promedio(datos);

   Eigen::MatrixXd matrizNormalizada = diferenciaPromedio.array().rowwise()/Desviacion(diferenciaPromedio);

   return matrizNormalizada;

}

/*Función para dividir en 4 grandes grupos
 * X_train, Y_train, X_test,Y_test*/

std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> Extraction::TrainTestSplit(Eigen::MatrixXd datos, float size_train){
    /*Cantidad de filas totales*/
    int filas_totales=datos.rows();

    //Cantidad de filas para entrenamiento
    int filas_train=round(filas_totales*size_train);

    //Cantidad de filas para prueba
    int filas_test=filas_totales-filas_train;

    Eigen::MatrixXd Train = datos.topRows(filas_train);

    /*Se desprenden para independientes y dependientes*/
    Eigen::MatrixXd X_Train = Train.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_Train = Train.rightCols(1);

    Eigen::MatrixXd Test= datos.bottomRows(filas_test);

    Eigen::MatrixXd X_Test = Test.leftCols(datos.cols()-1);
    Eigen::MatrixXd y_Test = Test.rightCols(1);

    /*Se compacta la tupla yse retorna*/
    return std::make_tuple(X_Train,y_Train,X_Test,y_Test);
}


/*Para efectos de visualización se creará la función vector a fichero*/

void Extraction::VectortoFile(std::vector<float> vector,std::string file_name){
    std::ofstream file_salida(file_name);

    //Se crea un iterador para almacenar la salida del vector
    std::ostream_iterator<float> salida_iterador(file_salida,"\n");

    //Se copia cada valor desde el inicio hasta el fin del iterador en el fichero
    std::copy(vector.begin(),vector.end(),salida_iterador);
}

/*Para efectos de manipulación y visualización se crea la función matriz eigen a fichero*/

void Extraction::EigentoFile(Eigen::MatrixXd matriz, std::string file_name){
    std::ofstream file_salida(file_name);
    if(file_salida.is_open()){
        file_salida<<matriz <<"\n";
    }
}











































