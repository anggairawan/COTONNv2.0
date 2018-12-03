#include <iostream>
#include <string>
#include <array>
#include <math.h> 
#include <typeinfo>

#include "scots.hh"

/*
	These parameters need to be changed in order to operate the converter
*/
const std::string controller_filename = "controller";
const std::string new_controller_filename = "../controller";
const std::string distinct_inputs_filename = "../nondet";

const int state_dim = 2;
const int input_dim = 1;

/*
	Actual program
*/
const bool debug_mode = 0;

using state_type = std::array<double,state_dim>;
using input_type = std::array<double,input_dim>;

struct ControllerPoint {
	unsigned int state; // state id
	// unsigned int input; // input id
	std::vector<int> input; // input id
	// ControllerPoint(unsigned int s, unsigned int i) : state(s), input(i) { }
	ControllerPoint(unsigned int s, std::vector<int> i) : state(s), input(i) { }
};

bool sort_winning_domain(ControllerPoint a, ControllerPoint b) {
	return (a.state < b.state);
}

void print(std::string text) {
	std::cout << text << std::endl;
}

// Get bounds and eta's
void get_bounds_and_etas(scots::SymbolicSet &controller, state_type &ss_lb, state_type &ss_ub, state_type &ss_eta, input_type &is_lb, input_type &is_ub, input_type &is_eta) {
	auto dim = controller.get_dim();
	auto eta = controller.get_eta();
	auto lower_bound = controller.get_lower_left();
	auto upper_bound = controller.get_upper_right();

	// state_space
	for(int i = 0; i < state_dim; i++) {
		ss_lb[i] = lower_bound[i];
		ss_ub[i] = upper_bound[i];
		ss_eta[i] = eta[i];
	}

	// input space
	for(int i = 0; i < (dim - state_dim); i++) {
		is_lb[i] = lower_bound[i + state_dim];
		is_ub[i] = upper_bound[i + state_dim];
		is_eta[i] = eta[i + state_dim];
	}
}

// Find and return the uniform grids that are associated with the controller
void get_uniform_grids(scots::SymbolicSet &controller, scots::UniformGrid &ss, scots::UniformGrid &is) {
	state_type ss_lb, ss_ub, ss_eta;
	input_type is_lb, is_ub, is_eta;
	get_bounds_and_etas(controller, ss_lb, ss_ub, ss_eta, is_lb, is_ub, is_eta);
	ss = scots::UniformGrid(state_dim, ss_lb, ss_ub, ss_eta);
	is = scots::UniformGrid(input_dim, is_lb, is_ub, is_eta);
	if(debug_mode)
	{
		print("\nState space:");
		ss.print_info();
		print("\nInput space:");
		is.print_info();
	} 	
}

// Find the winning domain from the controller, bdd, state space and input space
std::vector<ControllerPoint> find_winning_domain(scots::SymbolicSet &controller, Cudd &manager, BDD &bdd, scots::UniformGrid &ss, scots::UniformGrid &is) {
	std::vector<ControllerPoint> winning_domain;

	auto size = ss.size();

	for(unsigned int s = 0; s < size; s++) {
		// read x at state
		state_type x;
		ss.itox(s, x);

		// read input u at that state x
		auto u = controller.restriction(manager, bdd, x);

		// check if state is in winning domain
		if(!u.empty()) {
			size_t u_size = u.size();
			std::vector<int> I;

			for(unsigned j=0; j<(u_size/input_dim); j++)
			{
				// read input id from input x
				auto i = is.xtoi(u);
				// std::cout << s << " " << u.size() << " ";
				// for (unsigned k = 0; k < u.size(); ++k) std::cout << u[k] << ' ';
				// std::cout << "\n";
			  	I.push_back(i);
				// add controllerpoint to winning_domain
				u.erase (u.begin(),u.begin()+input_dim);
			}
			winning_domain.push_back(ControllerPoint(s, I));
		}
	}
	return winning_domain;
}

// Find unique subset of power set of inputs
bool write_unique_inputs(scots::SymbolicSet &controller, Cudd &manager, BDD &bdd, scots::UniformGrid &is, bool append_to_file = false) {
  scots::FileWriter writer(distinct_inputs_filename);
  if(append_to_file) {
   	if(!writer.open()) {
   	 	return false;
    }
	}else{
		if(!writer.create()){
      return false;
    }
	}

  std::vector<std::vector<double>> results = controller.get_distinct_inputs<int>(manager, bdd, state_dim);

	for(unsigned i=0; i < results.size(); i++){
		writer.add_PLAIN(std::to_string(i)+" ");
    size_t u_size = results[i].size();
    for(unsigned j = 0; j < (u_size/input_dim); j++){      
    	auto id_i = is.xtoi(results[i]);
    	results[i].erase (results[i].begin(),results[i].begin()+input_dim);
    	writer.add_PLAIN(std::to_string(id_i)+' ');
    }
    writer.add_PLAIN("\n");
  }

	/*
  for(unsigned i = 0; i < results.size(); i++){
    if(results[i].size()){
      writer.add_PLAIN(std::to_string(i)+"(");
    }
    
    for(unsigned j = 0; j < results[i].size(); j++){
      if(j%input_dim == 0)writer.add_PLAIN("(");
      writer.add_PLAIN(std::to_string(results[i][j])+' ');
      if((j+1)%input_dim == 0)writer.add_PLAIN(")");
    }
    if(results[i].size()){
      writer.add_PLAIN(")\n");
    }        
  }
  */
  writer.close();

  return true;
}

// Write a new controller using the old symbolic set controller and the newly formatted controller
bool write_new_controller(scots::SymbolicSet &controller, std::vector<ControllerPoint> &winning_domain, scots::UniformGrid &ss, scots::UniformGrid &is, bool append_to_file = false) {
	scots::FileWriter writer(new_controller_filename);
	if(append_to_file) {
        if(!writer.open()) {
            return false;
        }
    } else {
        if(!writer.create()) {
            return false;
        }
    }

    auto eta = controller.get_eta();
    auto lb = controller.get_lower_left();
    auto ub = controller.get_upper_right();

    std::vector<double> ss_eta(state_dim);
    std::vector<double> ss_lb(state_dim);
    std::vector<double> ss_ub(state_dim);

    std::vector<double> is_eta(input_dim);
    std::vector<double> is_lb(input_dim);
    std::vector<double> is_ub(input_dim);

    for(int i = 0; i < state_dim; i++) {
    	ss_eta[i] = eta[i];
    	ss_ub[i] = ub[i];
    	ss_lb[i] = lb[i];
    }

    for(int i = 0; i < input_dim; i++) {
    	is_eta[i] = eta[i+state_dim];
    	is_ub[i] = ub[i+state_dim];
    	is_lb[i] = lb[i+state_dim];
    }

    writer.add_VERSION();
    writer.add_TYPE(SCOTS_SC_TYPE);

    writer.add_TEXT("STATE_SPACE");
    writer.add_TYPE(SCOTS_UG_TYPE);
    writer.add_MEMBER(SCOTS_UG_DIM,state_dim);
    writer.add_VECTOR(SCOTS_UG_ETA,ss_eta);
    writer.add_VECTOR(SCOTS_UG_LOWER_LEFT,ss_lb);
    writer.add_VECTOR(SCOTS_UG_UPPER_RIGHT,ss_ub);

    writer.add_TEXT("INPUT_SPACE");
    writer.add_TYPE(SCOTS_UG_TYPE);
    writer.add_MEMBER(SCOTS_UG_DIM,input_dim);
    writer.add_VECTOR(SCOTS_UG_ETA,is_eta);
    writer.add_VECTOR(SCOTS_UG_LOWER_LEFT,is_lb);
    writer.add_VECTOR(SCOTS_UG_UPPER_RIGHT,is_ub);

    writer.add_PLAIN("#TYPE:WINNINGDOMAIN\n#SCOTS:i (state) j_0 ... j_n (valid inputs)\n#MATRIX:DATA\n");
    writer.add_PLAIN("#BEGIN:" + std::to_string(ss.size()) + " " + std::to_string(is.size()) + "\n");

    for(unsigned int i = 0; i < winning_domain.size(); i++) {
    	writer.add_PLAIN(std::to_string(winning_domain[i].state));

    	for(unsigned j=0; j < winning_domain[i].input.size(); j++)
    	writer.add_PLAIN(" " + std::to_string(winning_domain[i].input[j]));

    	writer.add_PLAIN("\n");
    }

    writer.add_PLAIN("#END");

    writer.close();

    return true;
}

int main(){
	print("Controller converter v1.0");

	// initialize CUDD and controller
	Cudd manager;
	BDD bdd;
	scots::SymbolicSet controller;

	// read controller using SCOTS
	if(!read_from_file(manager, controller, bdd, controller_filename)) {
		print("Could not read controller from: " + controller_filename);
		return 0;
	}

	// init uniform grid state and input space
	scots::UniformGrid ss;
	scots::UniformGrid is;
	get_uniform_grids(controller, ss, is);

	// Find winning domain
	std::vector<ControllerPoint> winning_domain = find_winning_domain(controller, manager, bdd, ss, is);

	// sort winning domain
	std::sort(begin(winning_domain), end(winning_domain), sort_winning_domain);

	// print if in debug mode
	if(debug_mode) {
		for(unsigned int i = 0; i < winning_domain.size(); i += 100) {
			// print("SS: " + std::to_string(winning_domain[i].state) + " IS: " + std::to_string(winning_domain[i].input));
		}
	}

	// write distinct inputs
	if(write_unique_inputs(controller, manager, bdd, is)) {
		print("Distinct inputs written to: " + distinct_inputs_filename + ".scs");
	} else {
		print("An error occured while writing the inputs.");
	}

	std::ofstream myfile;
	std::ofstream mylog;
	myfile.open ("../../../simulation/matlab/winning_dcdc_bdd.m");
	mylog.open("../../../controllers/nn/con"+std::to_string(winning_domain.size())+".m");


	auto size = ss.size();

	for(int i=0; i < state_dim; i++){
		myfile << "x{"+std::to_string(i+1)+"} = [";
		mylog << "x{"+std::to_string(i+1)+"} = [";
		for(unsigned int s = 0; s < size; s++) {
			// read x at state
			state_type x;
			ss.itox(s, x);

			// read input u at that state x
			auto u = controller.restriction(manager, bdd, x);

			// check if state is in winning domain
			if(u.empty()) {
				myfile << x[i] << ", ";  
			    mylog << x[i] << ", ";
			}
		}
		myfile << "]; \n";
		mylog << "]; \n";
	}

	for(int i=0; i < state_dim; i++){
		myfile << "x{"+std::to_string(i+1+state_dim)+"} = [";
		mylog << "x{"+std::to_string(i+1+state_dim)+"} = [";
		myfile << "]; \n";
		mylog << "]; \n";
	}
	/*
	state_type x;
	scots::abs_type state_size = winning_domain.size();
	// std::cout << state_size;
	for(int i=0; i < state_dim; i++){
		myfile << "x{"+std::to_string(i+1)+"} = [";
		mylog << "x{"+std::to_string(i+1)+"} = [";
		for(scots::abs_type j=0; j<state_size; j++) {
			ss.itox(winning_domain[j].state, x);
		    myfile << x[i] << ", ";  
		    mylog << x[i] << ", ";
		}
		myfile << "]; \n";
		mylog << "]; \n";
	}
	*/

	myfile.close();
	mylog.close();


	// write out new controller
	if(write_new_controller(controller, winning_domain, ss, is)) {
		print("Converter controller to static controller and written to: " + new_controller_filename + ".scs");
	} else {
		print("An error occured while writing the controller.");
	}

	return 1;
}