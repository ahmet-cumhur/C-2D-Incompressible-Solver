# C++ 2D CFD Solver For Incompressible Flows

A solver for 2D incompressible flows uses finitie difference method (FDM) to discretize the PDE's. For pressure and velocity coupling utilizes the fractional step methods (FSM) with immersed boundary methods(IBM) for simulating the external flow over objects. 
Gives its outputs as .vtk files which are suitable to analyzed on ParaView.
## Demo (GIF)
![Demo](media/square_vortex_shed.gif)

## Features
- Staggered Grid (u, v on faces; p at cell centers)
- Fractional Step Methods (FSM)
- Immersed boundary methods (IBM)
- Adaptive time depending on (CFL/diffusion)
- .vtk output for ParaView

## Dependencies
- C++ compiler
- Eigen (Not included here but you need to add into working directory)
- IntelMKL (Needs to be set up for working directory)
- Needs a file named vtk_output for saving output files

> Note: Eigen/MKL are not uploaded to this repository.  
> You should set the Eigen include path in your build/IDE, and configure MKL via oneAPI.

## Future
- More immersed objects / obstacle shapes
- Sinh geometric streching for better resolution where object lies
- OpenMP parallelization
- Update the poissons matrix build ✅
- Add energy equation
- Add RANS (k-eps or k-omg) turbulence equations
- Better structured code ✅
