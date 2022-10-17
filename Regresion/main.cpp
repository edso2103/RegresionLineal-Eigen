#include "Extraccion/extraction.h"
#include "Regresion/linearregression.h"
#include <stdlib.h>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <cmath>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <fstream>

using namespace std;

int main(int argc,char *argv[])
{
    /*Se necesitan tres argumentos de entrad*/

    /*Se crea un objeto del tipo ClassExtraction*/
    Extraction ExData (argv[1], argv[2], argv[3]);

    //Se instancia la clase de regresion lineal
    LinearRegression modeloLR;

    /*Se crea un vector de vectores del tipo string para cargar objeto ExData*/
    std::vector<std::vector<std::string>>dataframe= ExData.LeerCSV();

    /*Cantidad de filas y columnas*/
    int filas = dataframe.size()+1;
    int columnas = dataframe[0].size();


    /*Se crea una matriz Eigen, para ingresar los valores a esa matriz*/
    Eigen::MatrixXd matData=ExData.CSVtoEigen(dataframe,filas,columnas);

    /*Se normaliza la matriz de datos*/
    Eigen::MatrixXd matNorm=ExData.Norm(matData);

    /*Se divide en datos de entrenamiento y datos de prueba*/
    Eigen::MatrixXd X_train, y_train, X_test, y_test;

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd>div_datos=ExData.TrainTestSplit(matNorm,0.8);

    /*Se descomprime la tupla en cuatro conjuntos*/
    std::tie(X_train, y_train, X_test, y_test) = div_datos;

    /*Se crean vectores auxiliares para prueba y entrenamiento incializados en 1*/
    Eigen:: VectorXd vectorTrain = Eigen::VectorXd::Ones(X_train.rows());
    Eigen:: VectorXd vectorTest = Eigen::VectorXd::Ones(X_test.rows());

    /*Se redimensiona la matriz de entrenamiento y de prueba para ser
     * ajustados a los vectores auxiliares anteriores*/
    X_train.conservativeResize(X_train.rows(),X_train.cols()+1);
    X_train.col(X_train.cols()-1)=vectorTrain;

    X_test.conservativeResize(X_test.rows(),X_test.cols()+1);
    X_test.col(X_test.cols()-1)=vectorTest;


    /*Se crea el vector de coeficientes theta*/
    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());

    /*Se crea un alpha como ratio de aprendizaje de tipo flotante*/
    //Las iteraciones corresponden a los pasos

    float alpha = 0.01;
    int iteraciones= 1000;

    /* Se definen la variables de salida que representan a los coeficientes (m y b) y el vector de costo */

    Eigen::VectorXd thetaOut;
    std::vector<float>costo;

    //Se calcula el gradiente
    std::tuple<Eigen::VectorXd,std::vector<float>>gradiente=modeloLR.Gradiente(X_train,y_train,theta,alpha,iteraciones);

    //Se desempaqueta el gradiente
    std::tie(thetaOut,costo)=gradiente;

    //Se almacenan los valores de thetas y costos en un fichero para posteriormente ser visualizador
    //ExData.VectortoFile(costo,"Costos.txt");
    //ExData.EigentoFile(thetaOut,"Thetas.txt");

    //Se extrae el promedio de la matriz de entrada
    auto prom_data=ExData.Promedio(matData);

    //Se extraen los valores de la variable independiente
    //De las 12 columnas, teniendo 11 caracteristicas( x) y 1 'y'
    auto var_prom_independientes=prom_data(0,11);

    //Se escalan los datos
    auto datos_escalados=matData.rowwise()-matData.colwise().mean();

    //Se extrae la desviación estandar de los datos escalados
    auto desviacion=ExData.Desviacion(datos_escalados);

    //Se extraen los valores de la variable independiente
    auto var_desviacion_ind=desviacion(0,11);

    //Se crea una matriz para almacenar los valores estimados de entrenamiento
    Eigen::MatrixXd y_train_hat=(X_train*thetaOut*var_desviacion_ind).array()+var_prom_independientes;

    //Matriz para valores reales de y
    Eigen::MatrixXd y=matData.col(11).topRows(1279);

    //Se revisa que tan bueno fue el modelo a traves de la metrica de rendimiento
    float metrica_R2=modeloLR.R2_Score(y,y_train_hat);

    std::cout<<"\nR2: "<<metrica_R2<<std::endl;



    return EXIT_SUCCESS;
}
