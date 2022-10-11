#include "NumCpp.hpp"
#include <iostream>
#define PI 3.14159265359

// Y >> Gerçek
// X >> Gözlemlenen

nc::NdArray<double> determine_cores(const nc::NdArray<double> y_n, const nc::NdArray<double> x_n);
nc::NdArray<double> calc_transformation_params(const nc::NdArray<double> y_n, nc::NdArray<double>& x_n, nc::NdArray<double>& cores);
nc::NdArray<double> error(const nc::NdArray<double> y_n, const nc::NdArray<double> x_hat);
nc::NdArray<double> dataAlignment(const nc::NdArray<double> y_n, nc::NdArray<double>& x_n);


int main()
{
//Real landmarks coordinates, y_n    
nc::NdArray<double> y_n = {{7200,1800}, {16200,900}, {28800,2700}, {29700,9000},
                           {25200,16200}, {10800,15300}, {3600,14400}, {38500,6600},
                           {38500,15400}, {-6383,8487}, {2098,-5144}, {6600,-13200},
                           {23100,-13200}, {28600,-3300}, {-29700,-13200},
                           {-11000,-15400},{-5020,-6469}};
//Rotation and translation matrices initialization 
//Observed reflectors coordinates, x_n                         
nc::NdArray<double> x_n = {{7200,1800}, {16200,900}, {25200,16200}, {10800,15300}, {28800,2700}, {-5020,-6469}};
//nc::NdArray<double> x_n = {{25200,16200}, {10800,15300}, {3600,14400}, {38500,6600}, {7200,1800}, {16200,900}};

float rot_ang = 30 * (PI /180);
nc::NdArray<double> rotate = {{nc::cos(rot_ang), -nc::sin(rot_ang)},
                              {nc::sin(rot_ang), nc::cos(rot_ang)}};
nc::NdArray<double> translate = {{150, 210}};
auto oteleme = translate;
for(int i=0; i<x_n.numRows()-1; i++){
   translate = nc::append(translate, oteleme, nc::Axis::ROW);
}
x_n = nc::dot(rotate, x_n.transpose()).transpose();
x_n = nc::add(x_n, translate);

auto cores = dataAlignment(y_n, x_n);
int rows = x_n.numRows(); int cols = x_n.numCols();
double x_est[rows][cols];
int i = 0;  
for(auto& it : cores(1, cores.cSlice())){
    x_est[i][0] = y_n(it, 0);
    x_est[i][1] = y_n(it, 1);
    i++;
}
std::cout<<"Tahminler : "<<std::endl;
for(int i=0; i<x_n.numRows(); i++){
    for(int j=0; j<x_n.numCols(); j++)
        std::cout<<x_est[i][j]<<" ";
    std::cout<<std::endl;    
}   
//cores.print();



    return EXIT_SUCCESS;
}


//Cores Table:
//           first row : indices of x_n, current'observed' scan
//           second row : corresponding indices of y_n, reference scan
//           third row : corresponding min distance between each two points
nc::NdArray<double> determine_cores(const nc::NdArray<double> y_n, const nc::NdArray<double> x_n){
int scanSize = x_n.shape().rows;
//int idx;
auto cores = nc::zeros<double>(3, scanSize);
for (int i=0; i<scanSize; i++){
    double minDist = 5000000.;
    int idx;
    for (int j=0; j<y_n.shape().rows; j++){
        auto d = y_n(j, y_n.cSlice()) - x_n(i, x_n.cSlice());
        d = nc::norm(d, nc::Axis::COL).astype<double>();
        if(d[0] < minDist){
            minDist = d[0];
            idx = j;
        }
    }
    cores(0, i) = i;
    cores(1, i) = idx;
    cores(2, i) = minDist;
}
//********************************************************************//
    return cores;
}
//Transformation parameters(3x2) >> Rotation matrix(2x2) and translation vector(2x1).
//We extract them like this:
//trans = calc_transformation_params(...)
//R = trans({0, 2}, trans.cSlice())
//t = trans(2, trans.cSlice())
nc::NdArray<double> calc_transformation_params(const nc::NdArray<double> y_n, nc::NdArray<double>& x_n, nc::NdArray<double>& cores){
    auto H = nc::zeros<double>(2, 2);
    auto R = nc::empty<double>(2, 2);
    auto t = nc::empty<double>(2, 1);
    auto U = nc::empty<double>(2, 2);
    auto S = nc::empty<double>(2, 2);
    auto Vt = nc::empty<double>(2, 2);
    auto x_0 = nc::zeros<double>(1, 2);
    auto y_0 = nc::zeros<double>(1, 2);
    auto trans = nc::empty<double>(3, 2);
    int cntCores = 0;
    for (int i=0; i<cores.shape().cols; i++){
            H = H + nc::dot( y_n(int(cores(1, i)), y_n.cSlice()).transpose(), x_n(int(cores(0, i)), x_n.cSlice()) );
            x_0 = x_0 + x_n(int(cores(0, i)), x_n.cSlice());
            y_0 = y_0 + y_n(int(cores(1, i)), y_n.cSlice());
            cntCores ++;
    }
    x_0[0] = x_0[0] / cntCores; x_0[1] = x_0[1] / cntCores;
    y_0[0] = y_0[0] / cntCores; y_0[1] = y_0[1] / cntCores;
    nc::linalg::svd(H ,U, S, Vt);
    R = nc::dot(Vt.transpose(), U.transpose());
    t = y_0.transpose() - nc::dot(R, x_0.transpose());
    trans = nc::hstack({R, t});
    
    return trans.transpose();
}
//Calculating the error function
/*double loss(const nc::NdArray<double> y_n, const nc::NdArray<double> x_hat){
    auto error = nc::norm(y_n - x_hat);
    return error;
}
*/
//Main DATA ALIGNMENT algorithm
nc::NdArray<double> dataAlignment(const nc::NdArray<double> y_n, nc::NdArray<double>& x_n){
    auto x_hat = x_n;
    int rows = x_n.numRows(); int cols = x_n.numCols();
    double x_est[rows][cols];

    nc::NdArray<double> R = {{1, 0}, {0, 1}};
    nc::NdArray<double> t = {{0, 0}};
    //auto t = nc::empty<double>(2, 1);
    auto tr = nc::empty_like(x_hat);
    auto cores = nc::zeros<double>(3, x_n.shape().rows);
    auto trans = nc::empty<double>(3, 2);

    for (int it=0; it<20; it++){
        cores = determine_cores(y_n, x_hat);
        trans = calc_transformation_params(y_n, x_hat, cores);
        //R = nc::dot(R, trans({0, 2}, trans.cSlice()) );
        R = trans({0, 2}, trans.cSlice());
        t = trans(2, trans.cSlice());
        auto oteleme = t;
        for(int i=0; i<x_n.numRows()-1; i++){t = nc::append(t, oteleme, nc::Axis::ROW);}
        x_hat = nc::dot(R, x_hat.transpose()).transpose();
        x_hat = x_hat + t;
        
        //if (nc::norm(t)[0] < 20)
        //  break;   

    }
    cores = determine_cores(y_n, x_hat);
  
    return cores;
}


