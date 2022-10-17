#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <vector>

class LinearRegression
{
public:
    LinearRegression()
        {}
    float F_OLS_Costo(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd  theta);
    std::tuple<Eigen::VectorXd,std::vector<float>>Gradiente(Eigen::MatrixXd X,Eigen::MatrixXd y,Eigen::MatrixXd  theta,float alpha,int num_iter);
    float R2_Score(Eigen::MatrixXd y,Eigen::MatrixXd y_hat);
};
#endif // LINEARREGRESSION_H
