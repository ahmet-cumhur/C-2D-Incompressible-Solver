#include "two_d_solver.hpp"


int main(){
    //compute A the sparse matrix before the loop so we dont need to
    //re-compute all the time
    mkl_set_num_threads(8);
    std::cout << "Sim Staring.....\n";
    auto t_0 = std::chrono::steady_clock::now(); 
    double t_final = 3.0;
    double t_current = 0.0;
    Mesh2D two_D(700,100,7.0,1.0,100.0,0.01,5.0);
    //here is where we compute the poissons matrix before
    Mesh2D::LLT_solver LLT_solver;
    Mesh2D::LLT_Pardiso_Solver LLT_Pardiso_Solver;
    //this is SPD matrix w/ neumann BC 
    auto t0_matrix = std::chrono::steady_clock::now();
    auto A_sprs = two_D.get_sparse_matrix_triplets();
    auto t1_matrix = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_for_matrix = t1_matrix-t0_matrix;
    std::cout << "Total time spent on building matrix: " << time_for_matrix.count()<<std::endl;
    std::cout<< A_sprs.nonZeros()<<"\n";
    std::cout<<"factorization begins\n";
    two_D.LLT_Pardiso_compute(LLT_Pardiso_Solver,A_sprs);
    //two_D.LLT_compute_poisson(LLT_solver,A_sprs);
    std::cout<<"factorization ends\n";
    //
    //
    int i = 0;
    while(t_current<t_final){
        i +=1;
        double dt = two_D.time_step();
        //double dt = 1e-4;
        t_current += dt;
        //we dont apply pressure bc here!
        //two_D.apply_pressure_BC();
        two_D.apply_velocity_BC();
        two_D.apply_square_IBM();
        two_D.get_star();
        two_D.apply_square_IBM();

        //this is for LLT
        //two_D.LLT_solve_poisson_matrix_neumann(LLT_solver);

        //two_D.solve_poisson_matrix_CG(cg);
        two_D.LLT_Pardiso_Solve(LLT_Pardiso_Solver);

        two_D.time_roll();
        two_D.update_pressure();
        //we dont apply pressure bc here!
        //two_D.apply_pressure_BC();
        two_D.apply_velocity_BC();

        //data save time steps
        if(i%250==0){
            two_D.get_cell_centered_vel();
            two_D.data_output(t_current,i);
            
        }
        //give output about the time spent for sim etc...
        if(i%250== 0){
            std::cout << "Flow time: "<<t_current<<std::endl;
            auto t_now = std::chrono::steady_clock::now();
            std::chrono::duration<double> time_elapsed = t_now-t_0;
            std::cout << "Total time spent on sim: "<<time_elapsed.count() << "\n";
            std::cout << "Time step size: " << dt << "\n";
            std::cout << "MKL threads: " << mkl_get_max_threads() << "\n";
        }


    }
    std::cout << "Sim Finished.";
    int n;
    std::cin >> n; 

}
