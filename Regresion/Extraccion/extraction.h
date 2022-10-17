#ifndef EXTRACTION_H
#define EXTRACTION_H
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <vector>
#include <fstream>


/*Clase Extracción de datos:
     * Leer un fichero csv
     * Entrar con argumentos a la clase:
     *  -Lugar del dataset (csv)
     *  -Separador
     *  -Tiene cabecera?
 *Pasar a un vector de vectores del tipo string
 *Para el vector de vectores string a Eigen
     * -Promedio
     * -Desviación
     * -Normalización
     * -Métricas
*/

class Extraction
{
    /*Argumentos de entrada a la clase*/
    std::string dataset;          //ruta del dataset
    std::string delimitador;      //separador entre datos
    bool header;                  //cabecera o no


public:
    /*Se crea el constructor con los argumentos de entrada*/
    //Vector de vectores del tipo de string
    Extraction(std::string data,
               std::string separador,
               bool cabecera):
    dataset(data),delimitador(separador), header(cabecera){}

    //Prototipo de métodos o funciones
    std::vector<std::vector<std::string>>LeerCSV();

    Eigen::MatrixXd CSVtoEigen(
            std::vector<std::vector<std::string>>dataSet,
            int filas,
            int columnas);

auto Promedio(Eigen::MatrixXd datos)->decltype(datos.colwise().mean());
auto Desviacion(Eigen::MatrixXd datos)->decltype(((datos.array().square().colwise().sum()) / (datos.rows()-1)).sqrt());
Eigen::MatrixXd Norm(Eigen::MatrixXd datos);
std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd datos, float size_train);
void VectortoFile(std::vector<float> vector,std::string file_name);
void EigentoFile(Eigen::MatrixXd matriz, std::string file_name);
};
#endif // EXTRACTION_H
