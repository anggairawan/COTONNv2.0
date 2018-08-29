/*
 * simulate.cc
 *
 *  created: Oct 2016
 *   author: Matthias Rungger
 */

/*
 * information about this example is given in
 * http://arxiv.org/abs/1503.03715
 * doi: 10.1109/TAC.2016.2593947
 */

#include <iostream>
#include <array>
#include <cmath>
#include <fstream>

/* SCOTS header */
#include "scots.hh"
/* ode solver */
#include "RungeKutta4.hh"

/* state space dim */
const int state_dim=3;
/* input space dim */
const int input_dim=2;

/* sampling time */
const double tau = 0.3;

/*
 * data types for the state space elements and input space
 * elements used in uniform grid and ode solvers
 */
using state_type = std::array<double,state_dim>;
using input_type = std::array<double,input_dim>;

/* we integrate the vehicle ode by tau sec (the result is stored in x)  */
auto  vehicle_post = [](state_type &x, const input_type &u) {
  /* the ode describing the vehicle */
  auto rhs =[](state_type& xx,  const state_type &x, const input_type &u) {
    double alpha=std::atan(std::tan(u[1])/2.0);
    xx[0] = u[0]*std::cos(alpha+x[2])/std::cos(alpha);
    xx[1] = u[0]*std::sin(alpha+x[2])/std::cos(alpha);
    xx[2] = u[0]*std::tan(u[1]);
  };
  /* simulate (use 10 intermediate steps in the ode solver) */
  scots::runge_kutta_fixed4(rhs,x,u,state_dim,tau,10);
};

int main() {

  /* define function to check if we are in target */
  auto target = [](const state_type& x) {
    if (9 <= x[0] && x[0] <= 9.5 && 0 <= x[1] && x[1] <= 0.5)
      return true;
    return false;
  };

  /* read controller from file */
  scots::StaticController con;
  if(!read_from_file(con,"controller")) {
    std::cout << "Could not read controller from controller.scs\n";
    return 0;
  }
  
  std::cout << "\nSimulation:\n " << std::endl;

  state_type x={{.6, 0.6, 0}};

  scots::UniformGrid ss = con.get_ss_uniformgrid();
  scots::UniformGrid is = con.get_is_uniformgrid();
  state_type ss_int; 
  input_type is_int; 
  
  scots::abs_type ss_i;
  scots::abs_type is_i;

  std::ofstream myfile;
  std::ofstream myfile_abs_int; 
  std::ofstream myfile_abs_real; 
  
  myfile.open ("train_sequence_complete.txt");    
  myfile_abs_int.open ("train_sequence_integer_abstraction.txt");
  myfile_abs_real.open ("train_sequence_real.txt");
  
  while(1) {

    std::vector<input_type> u = con.get_control<state_type,input_type>(x);
    std::cout << x[0] <<  " "  << x[1] << " " << x[2] << " " << u.size() << " ";
    std::cout << u[0][0] <<  " "  << u[0][1] << "\n";
    
    ss.xtois(x, ss_int);
    is.xtois(x, is_int);
    ss_i = ss.xtoi(x);
    is_i = is.xtoi(u[0]);

    myfile << x[0] << " " << ss_int[0] << " ";
    myfile << x[1] << " " << ss_int[1] << " ";
    myfile << x[2] << " " << ss_int[2] << " ";
    myfile << u[0][0] << " " << is_int[0] << " ";  
    myfile << u[0][1] << " " << is_int[1] << " ";
    myfile << ss_i << " " << is_i << '\n';
    
    for(int i=0; i < state_dim; i++){
      myfile_abs_int << ss_int[i] << " "; 
      myfile_abs_real << x[i] << " ";      
    }

    for(int i=0; i < input_dim; i++){
      myfile_abs_int << is_int[i] << " ";
      myfile_abs_real << u[0][i] << " ";       
    }
    myfile_abs_int << "\n";
    myfile_abs_real << "\n";

    vehicle_post(x,u[0]);
    if(target(x)) {
      std::cout << "Arrived: " << x[0] <<  " "  << x[1] << " " << x[2] << std::endl;
      break;
    }
  }
  myfile.close();
  myfile_abs_int.close();
  myfile_abs_real.close()

  return 1;
}