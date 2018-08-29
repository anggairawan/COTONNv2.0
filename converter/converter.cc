#include <array>
#include <cmath>
#include <fstream>

/* SCOTS header */
#include "scots.hh"
/* ode solver */
#include "RungeKutta4.hh"

// const std::string controller_filename = "../examples/dcdc/controller";
const std::string controller_filename = "../ext/COTONN/controllers/dcdc_small_bdd/controller";
const std::string controller_path = "../controllers/cotonn_converted/dcdc_small_bdd/";

/* state space dim */
const int state_dim=3;
/* input space dim */
const int input_dim=2;

/*
 * data types for the state space elements and input space
 * elements used in uniform grid and ode solvers
 */

using state_type = std::array<double,state_dim>;
using input_type = std::array<double,input_dim>;

int main() {
  /* read controller from file */
  scots::StaticController con;
  if(!read_from_file(con,controller_filename)) {
    std::cout << "Could not read controller from controller.scs\n";
    return 0;
  }

  state_type x{};
  
  scots::UniformGrid ss = con.get_ss_uniformgrid();
  scots::UniformGrid is = con.get_is_uniformgrid();
  auto wd = con.get_domain<state_type>();
  
  state_type ss_int; 
  input_type is_int; 
  
  scots::abs_type ss_i;
  scots::abs_type is_i;

  std::ofstream myfile;
  std::ofstream myfile_abs_int; 
  std::ofstream myfile_abs_real; 
  std::ofstream myfile_int_real;

  myfile.open (controller_path+"complete_form.scs");    
  myfile_abs_int.open (controller_path+"integer_abstraction.scs");
  myfile_abs_real.open (controller_path+"real_sampled.scs");
  myfile_int_real.open (controller_path+"combined_sampled.scs");

	for(scots::abs_type i=0; i<wd.size(); i++){
		x = wd[i];
		std::vector<input_type> u = con.get_control<state_type,input_type>(x);
		
		ss.xtois(x, ss_int);
		is.xtois(u[0], is_int);
		ss_i = ss.xtoi(x);
		is_i = is.xtoi(u[0]);

		for(int i=0; i < state_dim; i++){
			myfile << x[i] << " " << ss_int[i] << " ";
			myfile_abs_int << ss_int[i] << " "; 
			myfile_abs_real << x[i] << " ";
			myfile_int_real << ss_int[i] << " ";       
		}

		for(int i=0; i < input_dim; i++){
			myfile << u[0][i] << " " << is_int[i] << " ";  
			myfile_abs_int << is_int[i] << " ";
			myfile_abs_real << u[0][i] << " ";
			myfile_int_real << u[0][i] << " ";        
		}

		myfile << ss_i << " " << is_i << '\n';
		myfile_abs_int << "\n";
		myfile_abs_real << "\n";
		myfile_int_real << "\n";

	}

  myfile.close();
  myfile_abs_int.close();
  myfile_abs_real.close();

  return 1;
}