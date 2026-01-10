#pragma once
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <filesystem>
#include "two_d_solver.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

using EM = Eigen::MatrixXd;
using EV = Eigen::VectorXd;

inline void output(double t_current,int i,Mesh2D& td){
            //get data required for data writing
            int npts = td.nx*td.ny;

            double shited_x = 0.0+0.5*td.dx;
            double shited_y = 0.0+0.5*td.dy;
            //flatten the uc,vc
            for(int j = 0; j<td.ny; j++){
                for(int i = 0; i<td.nx; i++){
                    int k = i+td.nx*j;
                    td.uc_flat(k) = td.uc(j,i);
                    td.vc_flat(k) = td.vc(j,i);
                    td.zc_flat(k) = td.zc(j,i);
                    td.omega_flat(k) = td.omega(j,i);
                }
            }
            //get the file name
            //NOTE!! ofstream can't create directories so we need to have a directory 
            //called vtk_output
            //FIX this issue!!
            std::ostringstream oss;
            //<< "_t_"<< t_current you can add this but problem for paraview
            oss << "out_"<<std::setw(7)<<std::setfill('0')<< i << ".vtk";
            std::string file_name = oss.str();
            std::string working_dir = std::filesystem::current_path().string(); 
            file_name = working_dir+"\\vtk_output\\"+ file_name;
            std::ofstream FileOP(file_name);
            if(FileOP.is_open()){
                //Header
                FileOP << "# vtk DataFile Version 3.0\n";
                FileOP << "Cumhur's Data Viewer\n";
                FileOP << "ASCII\n";
                FileOP << "DATASET STRUCTURED_POINTS\n";
                FileOP << "DIMENSIONS " << td.nx <<" "<< td.ny<<" "<< "1\n";
                FileOP << "ORIGIN " << shited_x <<" "<< shited_y <<" "<< "0\n";
                FileOP << "SPACING " << td.dx <<" "<< td.dy <<" "<<"1\n";
                //Time Data
                FileOP << "FIELD FieldData 1\n";
                FileOP << "TIME 1 1 double\n";
                FileOP << t_current << "\n";
                //Point Data
                //pressure
                FileOP << "POINT_DATA"<< " " << npts << "\n";
                FileOP << "SCALARS P double 1\n";
                FileOP << "LOOKUP_TABLE default\n";
                for(int k = 0; k<(npts); k++){
                    FileOP << td.p_flat(k)<<"\n"; 
                }
                //vorticity
                FileOP << "SCALARS w double 1\n";
                FileOP << "LOOKUP_TABLE default\n";
                for(int k = 0; k<(npts); k++){
                    FileOP << td.omega_flat(k)<<"\n"; 
                }
                //Vector Data
                FileOP << "VECTORS U double\n";
                td.UVZ.col(0) = td.uc_flat;
                td.UVZ.col(1) = td.vc_flat;
                td.UVZ.col(2) = td.zc_flat;
                for(int k = 0; k<(td.ny*td.nx); k++){
                    FileOP << td.UVZ(k,0)<<" "<<td.UVZ(k,1)<<" "<<td.UVZ(k,2)<<"\n";
                }
            }
            else{std::cout << "Error occured during file writing\n";}            

        }