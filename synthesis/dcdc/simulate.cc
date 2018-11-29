/*
 * simulate.cc
 *
 *  created: Jan 2017
 *   author: Matthias Rungger
 */

/*
 * information about this example is given in the readme file
 */
#include <iostream>
#include <array>

/* SCOTS header */
#include "scots.hh"
/* ode solver */
#include "RungeKutta4.hh"

/* state space dim */
const int state_dim=2;
/* input space dim */
const int input_dim=1;
/* sampling time */
const double tau = 0.5;

/*
 * data types for the elements of the state space 
 * and input space used by the ODE solver
 */
using state_type = std::array<double,state_dim>;
using input_type = std::array<double,input_dim>;

/* parameters for system dynamics */
const double xc=70;
const double xl=3;
const double rc=0.005;
const double rl=0.05;
const double ro=1;
const double vs=1;
/* parameters for radius calculation */
const double mu=std::sqrt(2);

/* we integrate the dcdc ode by 0.5 sec (the result is stored in x)  */
auto system_post = [](state_type &x, const input_type &u) noexcept {
  /* the ode describing the dcdc converter */
  auto rhs =[](state_type& xx,  const state_type &x, const input_type &u) noexcept {
    if(u[0]==1) {
      xx[0]=-rl/xl*x[0]+vs/xl;
      xx[1]=-1/(xc*(ro+rc))*x[1];
    } else {
      xx[0]=-(1/xl)*(rl+ro*rc/(ro+rc))*x[0]-(1/xl)*ro/(5*(ro+rc))*x[1]+vs/xl;
      xx[1]=(1/xc)*5*ro/(ro+rc)*x[0]-(1/xc)*(1/(ro+rc))*x[1];
    }
	};
  scots::runge_kutta_fixed4(rhs,x,u,state_dim,tau);
};


int main() {
  /* read controller from file */
  scots::StaticController con;
  if(!read_from_file(con,"controller")) {
    std::cout << "Could not read controller from controller.scs\n";
    return 0;
  }

  std::cout << "\nSimulation:\n ";
  /* initial state */
  state_type x={{1, 5.565}};
  // state_type x={{1.35, 5.755}};

  scots::UniformGrid ss = con.get_ss_uniformgrid();
  scots::UniformGrid is = con.get_is_uniformgrid();
  state_type ss_int; 
  input_type is_int;

  scots::abs_type ss_i;
  scots::abs_type is_i;

  std::ofstream myfile;
  std::ofstream myfile_int;

  int total_sequence = 50;

  myfile.open ("sequence_complete"+std::to_string(total_sequence)+".txt");
  myfile_int.open ("sequence_int"+std::to_string(total_sequence)+".txt");

  /* iterate */
  for(int i=0; i<total_sequence; i++) {
    std::vector<input_type> u = con.get_control<state_type,input_type>(x);
    
    ss.xtois(x, ss_int);
    is.xtois(u[0], is_int);

    ss_i = ss.xtoi(x);
    is_i = is.xtoi(u[0]);

    for(int i=0; i < state_dim; i++){
      std::cout << x[i] << " " << ss_int[i] << " ";       
      myfile << x[i] << " " << ss_int[i] << " ";
      myfile_int << ss_int[i] << " ";       
    }

    for(int i=0; i < input_dim; i++){
      std::cout << u[0][i] << " " << is_int[i] << " ";
      myfile << u[0][i] << " " << is_int[i] << " ";
      myfile_int << is_int[i] << "\n";          
    }

    myfile << ss_i << " " << is_i << '\n';
    std::cout << ss_i << " " << is_i << '\n';

    /*
    for(std::string::size_type i= 0; i<u.size(); i++)
    {
      std::cout << u[i][0] << "\n";
    } 
    */
    system_post(x,u[0]);
  }
  myfile.close();
  return 1;
}
