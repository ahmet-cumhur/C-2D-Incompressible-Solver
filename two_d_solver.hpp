#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include "Eigen/PardisoSupport" 
#include <mkl.h>
#include <filesystem>

class Mesh2D{
    public:

        int nx;
        int ny;
        double l;
        double h;

        double re;
        double dt;
        double dt_min;
        double u_inlet;
        double nu;
        double eps;
        Eigen::MatrixXd un;
        Eigen::MatrixXd us;
        Eigen::MatrixXd vs;
        Eigen::MatrixXd vn;
        Eigen::MatrixXd pn;
        Eigen::MatrixXd pc;
        Eigen::VectorXd b;
        Eigen::MatrixXd b_2d;

        Eigen::MatrixXd uc;
        Eigen::MatrixXd vc;

        Eigen::MatrixXd u_cell;
        Eigen::MatrixXd v_cell;

        Eigen::MatrixXd zc;

        Eigen::MatrixXd UVZ;
        Eigen::VectorXd p_flat;
        Eigen::MatrixXd omega;
        Eigen::VectorXd omega_flat;
        Eigen::VectorXd uc_flat;
        Eigen::VectorXd vc_flat;
        Eigen::VectorXd zc_flat;

        Mesh2D(int aNx, int aNy, double aL, double aH,double aRe,double aNu, double aU_inlet):
        nx(aNx),
        ny(aNy),
        l(aL),
        h(aH),

        un(ny+2,nx+1),
        us(ny+2,nx+1),
        vn(ny+1,nx+2),
        vs(ny+1,nx+2),
        pn(ny,nx),
        pc(ny,nx),
        b(nx*ny),
        p_flat(nx*ny),
        b_2d(ny,nx),
        
        uc(ny,nx),
        vc(ny,nx),
        zc(ny,nx),
        UVZ(ny*nx,3),


        u_cell(ny+2,nx+1),
        v_cell(ny+1,nx+2),

        re(aRe),
        nu(aNu),
        u_inlet(aU_inlet),
        dt(1e-6),
        eps(1e-6),
        dt_min(1e-10),
        omega(ny,nx),
        omega_flat(ny*nx),
        uc_flat(ny*nx),
        vc_flat(ny*nx),
        zc_flat(ny*nx)
        
        {


            dt = 1e-6;
            eps = 1e-6;
            dt_min = 1e-10;
            un.setZero();
            us.setZero();
            vn.setZero();
            vs.setZero();
            pn.setZero();
            pc.setZero();
            b.setZero();
            b_2d.setZero();
            un.col(0).setConstant(u_inlet);
            us.col(0).setConstant(u_inlet);

            uc.setZero();
            vc.setZero();
            zc.setZero();
            UVZ.setZero();

            u_cell.setZero();
            v_cell.setZero();
            p_flat.setZero();
            omega.setZero();
            omega_flat.setZero();
            uc_flat.setZero();
            vc_flat.setZero();
            zc_flat.setZero();
    
        }
        //get other variables
        double get_dx(){
            double dx = l/(nx-1);
            return dx;
        }
        double get_dy(){
            double dy = h/(ny-1);
            return dy;
        }
        auto get_x(double dx){
            std::vector<double>x(nx,0.0);
            for(int i = 0; i <nx; i++){
                x[i] = i*dx; 
            }
            return x;
        }
        auto get_y(double dy){
            std::vector<double>y(ny,0.0);
            for(int j = 0; j <ny; j++){
                y[j] = j*dy; 
            }
            return y;
        }
        
        //get posiitons 
        int get_mem_pos(int j,int i){
            int k = i+nx*j;
            return k;
        }
        int get_mat_pos_x(int k){
            int i = k%nx;
            return i;
        }
        int get_mat_pos_y(int k){
            int j = k/nx;
            return j;
        }
    
        //find neighbours
        int get_n_loc(int k){
            int n = k + nx;
            return n; 
        }
        int get_s_loc(int k){
            int s = k - nx;
            return s; 
        }
        int get_w_loc(int k){
            int w = k - 1;
            return w; 
        }
        int get_e_loc(int k){
            int e = k + 1;
            return e; 
        }

        //check the boundaries
        std::string get_boundary(int k){
            std::string boundary = "";
            int x_loc = get_mat_pos_x(k);
            int y_loc = get_mat_pos_y(k);

            if(y_loc == 0){
                boundary += "s";
            }
            else if(y_loc == ny-1){
                boundary += "n";
            } 
            if(x_loc == 0){
                boundary += "w";
            }
            else if(x_loc == nx-1){
                boundary += "e";
            }
            return boundary;
        }

        std::array <double,2> get_inner_data(int k){
            std::array <double,2> inner_data;
            int x_loc = get_mat_pos_x(k);
            int y_loc = get_mat_pos_y(k);
            int n = get_n_loc(k);
            int s = get_s_loc(k);
            int w = get_w_loc(k);
            int e = get_e_loc(k);

            double dn;
            double n_in; 
            double dx = get_dx();
            double dy = get_dy();

             if(y_loc == 0){
                n_in = s;
                dn = dy;
                inner_data[0] = n;
                inner_data[1] = dy;
            }
            else if(y_loc == ny-1){
                n_in = n;
                dn = dy;
                inner_data[0] = s;
                inner_data[1] = dy;
            } else{
                n_in = 0.0;
                dn = 0.0;
            }
            if(x_loc == 0){
                n_in = w;
                dn = dx;
                inner_data[0] = e;
                inner_data[1] = dx;
            }
            else if(x_loc == nx-1){
                n_in = e;
                dn = dx;
                inner_data[0] = w;
                inner_data[1] = dx;
            } else{
                n_in = 0.0;
                dn = 0.0;
            }
            return inner_data;
        }
        //get poisson matrix
        
        using SparseMatrix = Eigen::SparseMatrix<double>;
        SparseMatrix get_sparse_matrix_full_neumann(){
            double dx = get_dx();
            double dy = get_dy(); 
            int N = nx*ny;
            SparseMatrix A(N,N);
            
            for(int k = 0; k<N; k++){

                int n = get_n_loc(k);
                int s = get_s_loc(k);
                int w = get_w_loc(k);
                int e = get_e_loc(k);

                int ix = get_mat_pos_x(k);
                int iy = get_mat_pos_y(k); 

                std::string boundary = get_boundary(k);
                auto i_data = get_inner_data(k);
                int n_in = i_data[0];
                int dn = i_data[1]; 

                A.coeffRef(k,k) = 2/(dx*dx)+2/(dy*dy);

                if(boundary.empty()){
                    A.coeffRef(k,w) = -1/(dx*dx);
                    A.coeffRef(k,e) = -1/(dx*dx);
                    A.coeffRef(k,n) = -1/(dy*dy);
                    A.coeffRef(k,s) = -1/(dy*dy);
                }
                //npos here is no position if "n" is not in no position 
                //means its inside
                else{
                    if(boundary.find("n")!=std::string::npos){
                        A.coeffRef(k,k) -= 1/(dy*dy);
                    }else{
                        A.coeffRef(k,n) = -1/(dy*dy); 
                    }
                    if(boundary.find("s")!=std::string::npos){
                        A.coeffRef(k,k) -= 1/(dy*dy);
                    }else{
                        A.coeffRef(k,s) = -1/(dy*dy);
                    }
                    if(boundary.find("w")!=std::string::npos){
                        A.coeffRef(k,k) -= 1/(dx*dx);
                    }else{
                        A.coeffRef(k,w) = -1/(dx*dx);
                    }
                    if(boundary.find("e")!=std::string::npos){
                        A.coeffRef(k,k) -= 1/(dx*dx);
                    }
                    else{
                        A.coeffRef(k,e) = -1/(dx*dx);
                    }
                }
                
            }

            //bcs of full of neumann bc we need to set a constant in our matrixes otherwise we will have a singularity
            //for this is will apply A(5,5) to 1 and A(5,:) = 0 and A(:,5)= 0 and b(5) = 0;
            int k0 = 200;
            A.row(k0) *= 0;
            A.col(k0) *= 0;
            A.coeffRef(k0,k0) = 1;

            A.prune(0.0);
            A.makeCompressed();
            return A;
        }
        
        SparseMatrix get_sparse_matrix_triplets(){
                double dx = get_dx();
                double dy = get_dy();

                const int N = nx*ny;

                double diag = 2/(dx*dx)+2/(dy*dy);
                double ux = -1/(dx*dx);
                double uy = -1/(dy*dy);

                int k0 = 200; 
                std::vector<Eigen::Triplet<double>> T;
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        int k = get_mem_pos(j,i);

                        int n = get_n_loc(k);
                        int s = get_s_loc(k);
                        int w = get_w_loc(k);
                        int e = get_e_loc(k);

                        //for singularity
                        T.emplace_back(k,k,diag);
                        //west
                        if(i>0){T.emplace_back(k,w,ux);}else{T.emplace_back(k,k,(ux));}
                        //east
                        if(i<nx-1){T.emplace_back(k,e,ux);}else{T.emplace_back(k,k,(ux));}
                        //north
                        if(j<ny-1){T.emplace_back(k,n,uy);}else{T.emplace_back(k,k,(uy));}
                        //south
                        if(j>0){T.emplace_back(k,s,uy);}else{T.emplace_back(k,k,(uy));}
                    }    
                }


                SparseMatrix A (N,N);
                A.setFromTriplets(T.begin(),T.end());
                A.row(k0) *= 0;
                A.col(k0) *= 0;
                A.coeffRef(k0,k0) = 1;

                A.prune(0.0);
                A.makeCompressed();
                return A;
            }
        
        // get the intermediate time velocities
        //lets add some discretezation possibilities
        void get_star(){
            double dx = get_dx();
            double dy = get_dy();
            for(int i = 1; i<nx; i++){
                for(int j = 0; j<ny; j++){
                    int i_u = i;
                    int j_u = j+1;
                    int i_v = i+1;
                    int j_v = j;
                    // fill here according to
                    double v_int = ((vn(j_v,i_v)+vn(j_v,i_v+1))/2);
                    double u_adv_x = (un(j_u,i_u)*((un(j_u,i_u+1)-un(j_u,i_u-1))/(2*dx)));
                    double u_adv_y = v_int*((un(j_u+1,i_u)-un(j_u-1,i_u))/(2*dy));
                    double p_diff_x = (pn(j,i)-pn(j,i-1))/dx;
                    double u_visc_x = (un(j_u,i_u+1)-2*un(j_u,i_u)+un(j_u,i_u-1))/(dx*dx);
                    double u_visc_y = (un(j_u+1,i_u)-2*un(j_u,i_u)+un(j_u-1,i_u))/(dy*dy);

                    us(j_u,i_u) = un(j_u,i_u)+dt*(
                        -u_adv_x-u_adv_y-p_diff_x+(1.0/re)*(u_visc_x+u_visc_y)
                    );
                }
            }


            for(int i = 0; i<nx; i++){
                for(int j = 1; j<ny; j++){
                    int i_u = i;
                    int j_u = j+1;
                    int i_v = i+1;
                    int j_v = j;
                    // fill here according to
                    double u_int = (un(j_u,i_u+1)+un(j_u,i_u))/2;
                    double v_adv_x = u_int*(vn(j_v,i_v+1)-vn(j_v,i_v-1))/(2*dx);
                    double v_adv_y = vn(j_v,i_v)*(vn(j_v+1,i_v)-vn(j_v-1,i_v))/(2*dy);
                    double p_diff_v = (pn(j,i)-pn(j-1,i))/dy;
                    double v_visc_x = (vn(j_v,i_v+1)-2*vn(j_v,i_v)+vn(j_v,i_v-1))/(dx*dx);
                    double v_visc_y = (vn(j_v+1,i_v)-2*vn(j_v,i_v)+vn(j_v-1,i_v))/(dy*dy);

                    vs(j_v,i_v) = vn(j_v,i_v)+dt*(
                        -v_adv_x-v_adv_y-p_diff_v+(1.0/re)*(v_visc_x+v_visc_y)
                    );

                }
            }
            //fill the boundary conditions 
            
            //U
            //inlet bc
            us.col(0).setConstant(u_inlet);
            //us.col(1).setConstant(u_inlet);
            //  north and southern wall bc
            us.row(ny+1) = -us.row(ny);
            us.row(0) = -us.row(1);
            //eastern wall bc
            us.col(nx) = us.col(nx-1);

            //V
            //inlet bc
            vs.col(0).setConstant(0);
            //eastern boundary or outlet
            vs.col(nx+1) = vs.col(nx);
            //north south wall bc
            vs.row(0).setConstant(0);
            vs.row(ny).setConstant(0);
        }

       
        //define the LLT solver here,
        using Sp_Matrix = Eigen::SparseMatrix<double>;
        using LLT_solver = Eigen::SimplicialLLT<Sp_Matrix>;
        void LLT_compute_poisson(LLT_solver& Solver, Sp_Matrix& A_sprs){
            Solver.compute(A_sprs);
            if (Solver.info() != Eigen::Success) std::cout << "LLT compute failed\n";
        }
        void LLT_solve_poisson_matrix_neumann(LLT_solver& Solver){
            double dx = get_dx();
            double dy = get_dy();
            // b = us-us ......
            double sum_div = 0.0;
            for(int i = 0; i<nx; i++ ){
                for(int j = 0; j<ny; j++){
                    int k = get_mem_pos(j,i);
                    double div = ((us(j+1,i+1)-us(j+1,i))/dx)
                    + ((vs(j+1,i+1)-vs(j,i+1))/dy);
                    b[k] = div;
                    sum_div += div;
                }
            }
            //calculate the mean of b 
            const double mean_div = sum_div / double(nx*ny);
            double b_mean = b_2d.mean();
            b.array() -= mean_div;
            b.array() /= dt; 
            b[200] = 0;
            //bcs of full of neumann bc we need to set a constant in our matrixes otherwise we will have a singularity
            //for this is will apply A(5,5) to 1 and A(5,:) = 0 and A(:,5)= 0 and b(5) = 0;
            //define the solver

            p_flat = Solver.solve(-b);
            if (Solver.info() != Eigen::Success) std::cout << "LLT solve failed\n";

            //unflatten
            for(int j = 0; j<ny; j++ ){
                for(int i = 0; i<nx; i++){
                    int k = get_mem_pos(j,i);
                    pc(j,i)= p_flat[k];
                    }
                }
        }

            using Sp_Matrix = Eigen::SparseMatrix<double>;
            using LLT_Pardiso_Solver = Eigen::PardisoLLT<Sp_Matrix>;
            void LLT_Pardiso_compute(LLT_Pardiso_Solver& Solver, Sp_Matrix& A_sprs){
                Solver.analyzePattern(A_sprs);
                Solver.factorize(A_sprs);
                if(Solver.info() != Eigen::Success) {std::cout<<"Pardiso Failed.\n";}
            }
            void LLT_Pardiso_Solve(LLT_Pardiso_Solver& Solver){
                double dx = get_dx();
                double dy = get_dy();
                // b = us-us ......
                double sum_div = 0.0;
                for(int i = 0; i<nx; i++ ){
                    for(int j = 0; j<ny; j++){
                        int k = get_mem_pos(j,i);
                        double div = ((us(j+1,i+1)-us(j+1,i))/dx)
                        + ((vs(j+1,i+1)-vs(j,i+1))/dy);
                        b[k] = div;
                        sum_div += div;
                    }
                }
                const double mean_div = sum_div / double(nx*ny);
                double b_mean = b_2d.mean();
                b.array() -= mean_div;
                b.array() /= dt; 
                b[200] = 0;
                //
                //
                Eigen::VectorXd rhs = -b;           
                p_flat = Solver.solve(rhs);
                if (Solver.info() != Eigen::Success) std::cout << "LLT solve failed\n";

                //unflatten
                for(int j = 0; j<ny; j++ ){
                    for(int i = 0; i<nx; i++){
                        int k = get_mem_pos(j,i);
                        pc(j,i)= p_flat[k];
                        }
                    }



            }
        
    
        double time_step(){
            double dx = get_dx();
            double dy = get_dy();

            double u_crit = un.cwiseAbs().maxCoeff();
            double v_crit = vn.cwiseAbs().maxCoeff();

            double cfl_x = dt*u_crit/dx;
            double cfl_y = dt*v_crit/dy;
            double four_x = nu*dt/(dx*dx);
            double four_y = nu*dt/(dy*dy);

            double dt_cfl_x = 0.1*dx/std::max(u_crit,eps);
            double dt_cfl_y = 0.1*dy/std::max(v_crit,eps);
            double dt_four_x = 0.1*(dx*dx)/nu;
            double dt_four_y = 0.1*(dy*dy)/nu;

            dt = std::max(dt_min,std::min({dt_cfl_x, dt_cfl_y,dt_four_x,dt_four_y}));
            //std::cout << "umax=" << u_crit << " vmax=" << v_crit<< " dt_advx=" << dt_cfl_x <<" dt_advy=" << cfl_y <<" dt_diffx=" << dt_four_x <<" dt_diffy=" << dt_four_y<< " dt=" << dt << std::endl;


            return dt;
        }
        void apply_velocity_BC(){
            //U
            //inlet bc
            un.col(0).setConstant(u_inlet);
            //us.col(1).setConstant(u_inlet);
            //  north and southern wall bc
            un.row(ny+1) = -un.row(ny);
            un.row(0) = -un.row(1);
            //eastern wall bc
            un.col(nx) = un.col(nx-1);

            //V
            //inlet bc
            vn.col(0).setConstant(0);
            //eastern boundary or outlet
            vn.col(nx+1) = vn.col(nx);
            //north south wall bc
            vn.row(0).setConstant(0);
            vn.row(ny).setConstant(0);
        }

        void apply_pressure_BC(){
            pn.col(nx-1).setZero();
            pc.col(nx-1).setZero();
        }

        void time_roll(){
            double dx = get_dx();
            double dy = get_dy();
            for(int i = 1; i<nx; i++){
                for(int j = 0; j<ny; j++){
                    un(j+1,i) = us(j+1,i)-(dt/dx)*(pc(j,i)-pc(j,i-1));
                }    
            }
            for(int i = 0; i<nx; i++){
                for(int j = 1; j<ny; j++){
                    vn(j,i+1) = vs(j,i+1)-(dt/dy)*(pc(j,i)-pc(j-1,i));
                }
            }
        }
        void update_pressure(){
            pn += pc;
        }
        void get_cell_centered_vel(){
            for(int i = 0; i<nx;i++){
                for(int j = 0; j<ny;j++){
                    uc(j,i) = 0.5*(un(j+1,i+1)+un(j+1,i));
                    vc(j,i) = 0.5*(vn(j+1,i+1)+vn(j,i+1));
                }
            }
        }


        void apply_square_IBM(){
            int mid_y = (ny/2)-10;
            int mid_x = (nx/2)-100; 
            un.block(mid_y,mid_x,20,20).setZero();
            vn.block(mid_y,mid_x,20,20).setZero();
            us.block(mid_y,mid_x,20,20).setZero();
            vs.block(mid_y,mid_x,20,20).setZero();
        }
        void calculate_vorticity(){
            double dx = get_dx();
            double dy = get_dy();
            //w = dv/dx - du/dy w/ central differences:
            for(int j = 0; j < ny; j++){
                for(int i = 0; i < nx; i++){
                    int k = get_mem_pos(j,i); 
                    omega(j,i) = ((vc(j,i+1)-vc(j,i-1))/(2*dx))-((uc(j+1,i)-uc(j-1,i))/(2*dy));                     
                }
            }
            omega.row(0) = -omega.row(1);
            omega.row(ny-1) = -omega.row(ny-2);
            omega.col(0) = -omega.col(1);
            omega.col(nx-1) = -omega.col(nx-2);    
        }

        void data_output(double t_current,int i){
            //get data required for data writing
            const double dx = get_dx();
            const double dy = get_dy();
            int npts = nx*ny;
            double o_x = 0.0;
            double o_y = 0.0;

            double shited_x = o_x+0.5*dx;
            double shited_y = o_y+0.5*dy;
            //flatten the uc,vc
            for(int j = 0; j<ny; j++){
                for(int i = 0; i<nx; i++){
                    int k = get_mem_pos(j,i);
                    uc_flat(k) = uc(j,i);
                    vc_flat(k) = vc(j,i);
                    zc_flat(k) = zc(j,i);
                    omega_flat(k) = omega(j,i);
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
                FileOP << "DIMENSIONS " << nx <<" "<< ny<<" "<< "1\n";
                FileOP << "ORIGIN " << shited_x <<" "<< shited_y <<" "<< "0\n";
                FileOP << "SPACING " << dx <<" "<< dy <<" "<<"1\n";
                //Time Data
                FileOP << "FIELD FieldData 1\n";
                FileOP << "TIME 1 1 double\n";
                FileOP << t_current << "\n";
                //Point Data
                //pressure
                FileOP << "POINT_DATA"<< " " << npts << "\n";
                FileOP << "SCALARS P double 1\n";
                FileOP << "LOOKUP_TABLE default\n";
                for(int k = 0; k<(ny*nx); k++){
                    FileOP << p_flat(k)<<"\n"; 
                }
                //vorticity
                calculate_vorticity();
                FileOP << "SCALARS w double 1\n";
                FileOP << "LOOKUP_TABLE default\n";
                for(int k = 0; k<(ny*nx); k++){
                    FileOP << omega_flat(k)<<"\n"; 
                }
                //Vector Data
                FileOP << "VECTORS U double\n";
                UVZ.col(0) = uc_flat;
                UVZ.col(1) = vc_flat;
                UVZ.col(2) = zc_flat;
                for(int k = 0; k<(ny*nx); k++){
                    FileOP << UVZ(k,0)<<" "<<UVZ(k,1)<<" "<<UVZ(k,2)<<"\n";
                }
            }
            else{std::cout << "Error occured during file writing\n";}            

        }
        //define the CG solver here as a Global
        
};

