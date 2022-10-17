#include "linearregression.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>
/*Primera función: Función de costo para la regresión lineal
 * basada en los minimos cuadrados ordinarios*/

float LinearRegression::F_OLS_Costo(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd theta)
{
    Eigen::MatrixXd m_interior=pow((X*theta-y).array(),2);

    return m_interior.sum()/(2*X.rows()); //penaliza los errores
}

/*Función de gradiente descendente: En función de la tasa de aprendizaje
 *El cual es el paso para avanzar dentro de la función del gradiente
 *COn el fin de obtener el minimos valor, "optimo" */

std::tuple<Eigen::VectorXd,std::vector<float>>LinearRegression::Gradiente(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd  theta,float alpha,int num_iter){
     Eigen::MatrixXd temporal=theta;
     int parametros=theta.rows();

     std::vector<float> costo;
     costo.push_back(F_OLS_Costo(X,y,theta));


     //Se itera según el número de iteraciones y el ratio de aprendizaje
     //para encontrar los valores óptimos


      for(int i=0;i<num_iter;i++){
         Eigen::MatrixXd error=X*theta-y;
         for(int j=0; j<parametros;j++){
             Eigen::MatrixXd X_i=X.col(j);
             Eigen::MatrixXd termino=error.cwiseProduct(X_i);

             temporal(j,0)=theta(j,0)-((alpha/X.rows())*termino.sum());

         }


         theta=temporal;
         costo.push_back(F_OLS_Costo(X,y,theta));
     }

     return std::make_tuple(theta,costo);
}
/*A continuacuón se presenta la función para revisar que tan bueno es nuestro proyecto
 *Se crea la metrica de rendimiento que diga que tan bueno es nuestro modelo,
 *esta metrica será R²(coef. de determinación), donde el mejor valor será 1 */
//Recibe reales y estimados
float LinearRegression::R2_Score(Eigen::MatrixXd y,Eigen::MatrixXd y_hat){
    auto numerador= pow((y-y_hat).array(),2).sum();
    auto denominador= pow((y.array()-y.mean()),2).sum();
    std::cout<<"\n-> R2: "<<1-(numerador/denominador)<<"\n\n";
    return 1-(numerador/denominador);
}
