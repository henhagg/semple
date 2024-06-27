#include <Rcpp.h>
using namespace Rcpp;

Rcpp::NumericVector linspace(double from, double to, double by){
  int num_points = floor((to-from)/by) + 1;
  Rcpp::NumericVector output(num_points);
  output(0) = from;
  for(int i = 1; i < num_points; i++) {
    output(i) = from + i*by;
  }
  return output;
}

Rcpp::NumericVector interpolate_grid(NumericVector t_data, NumericVector data, NumericVector t_out, double dt, double offset){
  
  int num_points = t_out.length();
  double t_0 = t_data(0);
  int data_length = t_data.length();
  double t_data_max = t_data(data_length-1);
  
  Rcpp::NumericVector y_out(num_points);
  
  for(int i = 0; i < num_points; i++) {
   
    if(t_out(i) >= t_data_max){
      y_out(i) = data(data_length-1);
      continue;
      
    }else if(t_out(i) < t_0){
      y_out(i) = offset;
      continue;
      
    }else{
      double n_steps = (t_out(i)-t_data(0))/dt;
      int n_steps_lower = floor(n_steps);
      
      double t_lower = t_data(0) + n_steps_lower*dt;
      double t_upper = t_lower+dt;
      
      double a = data(n_steps_lower);
      double b = data(n_steps_lower+1);
      
      double interp = a + (b-a)*(t_out(i)-t_lower)/dt;
      y_out(i) = interp;
    }
  }
  return y_out;
}

Rcpp::NumericMatrix cpp_euler_maruyama(double delta, double gamma, double k, double m0, double t0, double dt, int num_points){
  if (t0 > 30){
    Rcpp::NumericMatrix xs(1, 2);
    xs(0,0) = m0;
    xs(0,1) = 0;
    return xs;
  }
  
  // int num_points = ceil(((tt+dt)-t0)/dt);
  Rcpp::NumericMatrix xs(num_points, 2);
  xs(0,0) = m0;
  xs(0,1) = 0;
    
  Rcpp::NumericVector dw = Rcpp::rnorm(2*(num_points-1), 0, sqrt(dt));
  
  for(int i = 1; i < num_points; i++) {
    double m_temp = xs(i-1,0) - (delta*xs(i-1,0) * dt) + sqrt(delta*xs(i-1,0)) * dw(2*(i-1));
    double p_temp = xs(i-1,1) + (k*xs(i-1,0)-gamma*xs(i-1,1))*dt + sqrt(k*xs(i-1,0)+gamma*xs(i-1,1)) * dw((2*i)-1);
    
    if(m_temp < 0){
      xs(i,0) = 0;
    } else {
      xs(i,0) = m_temp;
    }
    
    if(p_temp < 0){
      xs(i,1) = 0;
    } else {
      xs(i,1) = p_temp;
    }
  }
  
  return xs;
}

// [[Rcpp::export]]
Rcpp::NumericVector model(NumericVector logparam){
  NumericVector param = exp(logparam);
  
  double delta = param(0);
  double gamma = param(1);
  double k = param(2);
  double m0 = param(3);
  double scale = param(4);
  double t0 = param(5);
  double offset = param(6);
  double sigma = param(7);
  
  double tt = 30;
  double dt = 0.01;
  int dim_data = 60;
  
  NumericVector ts;
  int num_points;
  if(t0 > tt){
    ts = NumericVector::create(t0);
    num_points = 1;
  }else{
    ts = linspace(t0, tt+dt, dt);
    num_points = ts.length();
  }
  
  NumericMatrix sol_euler = cpp_euler_maruyama(delta, gamma, k, m0, t0, dt, num_points);
  NumericVector y = log(scale*sol_euler(_,1) + offset);
  NumericVector t_out = linspace(tt/dim_data, tt, tt/dim_data);

  NumericVector sol = interpolate_grid(ts, y, t_out, dt, log(offset));
  NumericVector sol_with_noise = sol + Rcpp::rnorm(dim_data, 0, sigma);
  
  return sol_with_noise;
}
 