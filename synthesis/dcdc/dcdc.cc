/*
 * dcdc.cc
 *
 *  created: Oct 2015
 *   author: Matthias Rungger
 */

/*
 * information about this example is given in the readme file
 */

#include <iostream>
#include <array>
#include <cmath>

#include <vector>
#include "UpdateAvoid.hh"
#include <ctime>

/* SCOTS header */
#include "scots.hh"
/* ode solver */
#include "RungeKutta4.hh"

/* time profiling */
#include "TicToc.hh"
/* memory profiling */
#include <sys/time.h>
#include <sys/resource.h>
struct rusage usage;


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

/* abbrev of the type for abstract states and inputs */
using abs_type = scots::abs_type;

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
  scots::runge_kutta_fixed4(rhs,x,u,state_dim,tau,5);
};
/* we integrate the growth bound by 0.5 sec (the result is stored in r)  */
auto radius_post = [](state_type &r, const state_type &, const input_type &u) noexcept {
  /* the ode for the growth bound */
  auto rhs =[](state_type& rr,  const state_type &r, const input_type &u) noexcept {
    if(u[0]==1) {
      rr[0]=-rl/xl*r[0];
      rr[1]=-1/(xc*(ro+rc))*r[1];
    } else {
      rr[0]=-(1/xl)*(rl+ro*rc/(ro+rc))*r[0]+(1/xl)*ro/(5*(ro+rc))*r[1];
      rr[1]=5*(1/xc)*ro/(ro+rc)*r[0]-(1/xc)*(1/(ro+rc))*r[1];
    }
	};
  scots::runge_kutta_fixed4(rhs,r,u,state_dim,tau,5);
};

std::string get_time() {
    std::time_t t = std::time(0);   // get time now
    std::tm* now = std::localtime(&t);
    return std::to_string(now->tm_year + 1900) + std::to_string(now->tm_mon + 1) + std::to_string(now->tm_mday);
}

int main() {
  /* to measure time */
  TicToc tt;

  /* setup the workspace of the synthesis problem and the uniform grid */
   /* grid node distance diameter */
  // state_type eta={{2.0/4e2,2.0/4e2}};
  state_type eta={{0.005,0.005}};
  /* lower bounds of the hyper-rectangle */
  state_type lb={{0.5,5}};
  // state_type lb={{1.2,5.5}};
  /* upper bounds of the hyper-rectangle */
  state_type ub={{1.5,6}};
  // state_type ub={{1.5,5.8}};
  scots::UniformGrid ss(state_dim,lb,ub,eta);
  std::cout << "Uniform grid details:\n";
  ss.print_info();

  /* construct grid for the input alphabet */
  /* hyper-rectangle [1,2] with grid node distance 1 */
  scots::UniformGrid is(input_dim,input_type{{1}},input_type{{2}},input_type{{1}});
  is.print_info();

  extern std::vector<abs_type> H;

  /* avoid function returns 1 if x is in avoid set  */
  auto avoid = [ss,eta](const abs_type& idx) {
    //state_type x;
    //ss.itox(idx,x);
    //double c1= eta[0]/2.0+1e-10;
    //double c2= eta[1]/2.0+1e-10;
    for(size_t i=0; i<H.size(); i++) {
      if(idx == H[i])
        return true;
    }
    return false;
  };
  /* write obstacles to file */
  write_to_file(ss,avoid,"obstacles");

  /* compute transition function of symbolic model */
  std::cout << "Computing the transition function:\n";
  /* transition function of symbolic model */
  scots::TransitionFunction tf;
  scots::Abstraction<state_type,input_type> abs(ss,is);
  abs.verbose_off();

  tt.tic();
  // abs.compute_gb(tf,system_post,radius_post);
  abs.compute_gb(tf, system_post, radius_post, avoid);
  tt.toc();
  std::cout << "Number of transitions: " << tf.get_no_transitions() <<"\n";

  if(!getrusage(RUSAGE_SELF, &usage))
    std::cout << "Memory per transition: " << usage.ru_maxrss/(double)tf.get_no_transitions() << "\n";

  /* continue with synthesis */
  /* define function to check if the cell is in the safe set  */
  auto safeset = [&lb, &ub, &ss, &eta](const scots::abs_type& idx) noexcept {
    state_type x;
    ss.itox(idx,x);
    /* function returns 1 if cell associated with x is in target set  */
    if (lb[0] <= (x[0]-eta[0]/2.0) && (x[0]+eta[0]/2.0)<= ub[0] && 
        lb[1] <= (x[1]-eta[1]/2.0) &&  (x[1]+eta[1]/2.0) <= ub[1])
      return true;
    return false;
  };
  /* compute winning domain (contains also valid inputs) */

  // std::vector<abs_type> pre;
  // pre = tf.get_pre(1,1);

  // for( int i = 0; i < pre.size(); i++ )
  //  std::cout << pre[i] << std::endl;

  std::cout << "\nSynthesis: \n";
  tt.tic();
  scots::WinningDomain win = scots::solve_invariance_game(tf,safeset);
  tt.toc();
  std::cout << "Winning domain size: " << win.get_size() << "\n";
  std::cout << "Number of states: " << win.get_no_states() << "\n";
  std::cout << "Number of inputs: " << win.get_no_inputs() << "\n";
  // std::cout << "Winning domain vectors: " << win.get_winning_domain() << "\n";


  std::ofstream myfile;
  std::ofstream mylog;
  myfile.open ("../../simulation/matlab/winning_x.m");
  mylog.open("../../controllers/nn/con"+std::to_string(win.get_size())+".m");
  
  /* iterate 
  state_type x;
  std::vector<abs_type> win_i = win.get_winning_domain();
  for(int i=0; i < state_dim; i++){
    myfile << "x{"+std::to_string(i+1)+"} = [";
    for(abs_type j=0; j<win.get_size(); j++) {
      ss.itox(win_i[j], x);
      myfile << x[i] << " ";       
    }
    myfile << "] \n";
  }
  myfile.close();
  */

  state_type x;
  abs_type state_size = win.get_no_states();
  std::cout << state_size;
  for(int i=0; i < state_dim; i++){
    myfile << "x{"+std::to_string(i+1)+"} = [";
    mylog << "x{"+std::to_string(i+1)+"} = [";
    for(abs_type j=0; j<state_size; j++) {
      if(!(win.is_winning(j)))
      {
        ss.itox(j, x);
        myfile << x[i] << ", ";  
        mylog << x[i] << ", ";             
      }
    }
    myfile << "]; \n";
    mylog << "]; \n";
  }
  myfile.close();
  mylog.close();

  scots::StaticController sc(ss,is,std::move(win));
  std::cout << "\nWrite controller to controller.scs \n";
  // if(write_to_file(scots::StaticController(ss,is,std::move(win)),"controller"))
  if(write_to_file(sc,"controller"))  
    std::cout << "Done. \n";

  return 1;
}

   
 
