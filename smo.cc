/*
 * Copyright 2010 Thomas Eastman
 *
 * This program is free software: you can redistribute it and/or modify it under the terms of the
 * GNU General Public License as published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program.  If
 * not, see <http://www.gnu.org/licenses/>.
 *
 *
 * This program implements Platt's Sequential Minimal Optimization (SMO) algorithm for Octave.
 *
 * For details, see the following papers:
 *
 *   J. C. Platt. Fast Training of Support Vector Machines using Sequential Minimal Optimization.
 *   In: B. Schölkopf, C. J. C. Burges and A. J. Smola, eds. Advances in Kernel Methods: Support
 *   Vector Learning. MIT Press, 1999, pp. 185-208.
 *
 *   J. C. Platt. Using Analytic QP and Sparseness to Speed Training of Support Vector Machines.
 *   In: M. S. Kearns, S. A. Solla and D. A. Cohn, eds. Advances in Neural Information Processing
 *   Systems 11. MIT Press, 1999, pp. 557-563.
 *
 *   S. S. Keerthi, S. K. Shevade, C. Bhattacharyya and K. R. K. Murthy. Improvements to Platt's
 *   SMO Algorithm for SVM Classifier Design. Neural Computation, 13(3):637-649, 2001.
 *
 * Usage:
 *
 *   >> [alpha b] = smo(X, y, c, kernel, epsilon, debug);
 *
 * Input arguments:
 *
 *   X       - instances in the data set, one per row (n by p)
 *   y       - instance class labels (n by 1)
 *   c       - SVM parameter c (default = 1)
 *   kernel  - reference to kernel function (see examples below)
 *   epsilon - tolerance for equality checking
 *   debug   - flag to output debug information
 *
 * Common kernel function examples:
 *
 *   >> kernel = @(x1, x2) x1 * x2';                       % linear (default)
 *   >> kernel = @(x1, x2) (a * x1 * x2' + c) .^ d;        % polynomial
 *   >> kernel = @(x1, x2) exp(-gamma * norm(x1 - x2)^2);  % radial basis
 *   >> kernel = @(x1, x2) tanh(a * x1 * x2' + c);         % sigmoid
 *
 * Output arguments:
 *
 *   alpha   - SVM Lagrange multiplier vector (n by 1)
 *   b       - SVM threshold value
 *
 * Bugs:
 *
 *  The kernel function reference is not checked to ensure that it refers to a kernel function that
 *  takes the correct arguments and returns a sensible value.  I don't know how to do this properly
 *  with the Octave C++ API.  The kernel function is assumed to have the signature
 *
 *   double scalar = kernel(double matrix (1 by p), double matrix (1 by p)).
 *
 *  This program may not behave properly if it does not.
 *
 */

#include <octave/oct.h>
#include <octave/parse.h>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>

const char *fname = "smo";
const double default_c = 1.0;
const double default_epsilon = 1e-10;
const bool default_debug = false;

Matrix X;
ColumnVector y;
octave_idx_type n; // instances
octave_idx_type p; // variables
ColumnVector E;
ColumnVector alpha;
double b;
Matrix kernel_cache;
boolMatrix kernel_valid;
octave_function *kernel;
double c;
double epsilon;
bool debug;

// evaluate a linear kernel function (default)
double kernel_linear(int i1, int i2)
{
  double val = 0.0;
  for(int j = 0; j < p; ++j)
    val += X(i1, j) * X(i2, j);

  return val;
}

// evaluate an external kernel function (if provided)
double kernel_external(int i1, int i2)
{
  octave_value_list kernel_args;
  kernel_args.append(octave_value(X.row(i1)));
  kernel_args.append(octave_value(X.row(i2)));

  octave_value_list kernel_return = feval(kernel, kernel_args, 1);

  return kernel_return(0).scalar_value();
}

// evaluate the kernel function and cache the result
double kernel_eval(int i1, int i2)
{
  if(kernel_valid(i1, i2))
    return kernel_cache(i1, i2);

  double val = kernel == NULL ? kernel_linear(i1, i2) : kernel_external(i1, i2);

  kernel_cache(i1, i2) = val;
  kernel_cache(i2, i1) = val;
  kernel_valid(i1, i2) = true;
  kernel_valid(i2, i1) = true;

  return val;
}

// optimize a pair of selected Lagrange multipliers
bool take_step(int i1, int i2)
{
  if(i1 == i2)
    return false;

  double alpha1 = alpha(i1);
  double alpha2 = alpha(i2);

  // compute bounds on alpha2
  double y1 = y(i1);
  double y2 = y(i2);
  double s = y1 * y2;
  double L = std::max(0.0, alpha2 + s * alpha1 - 0.5 * (s + 1.0) * c);
  double H = std::min(c, alpha2 + s * alpha1 - 0.5 * (s - 1.0) * c);

  if(L >= H)
    return false;

  double K11 = kernel_eval(i1, i1);
  double K22 = kernel_eval(i2, i2);
  double K12 = kernel_eval(i1, i2);

  // compute second derivative of the dual objective function projected onto the linear
  // equality constraint where all Lagrange multipliers are held constant exept for alpha1
  // and alpha2 following equation 12.5 (Platt 1999)
  double eta = 2.0 * K12 - K11 - K22;

  double E1 = E(i1);
  double E2 = E(i2);

  // compute optimal values of alpha1 and alpha2 on the feasible line segment illustrated
  // in figure 12.2 (Platt 1999)
  double alpha2_new;
  double alpha1_new;
  if(eta < 0.0){
    // compute optimal value of alpha2 and clip to satisfy the box constraints according
    // to equations 12.6 and 12.7 (Platt 1999)
    alpha2_new = alpha2 - y2 * (E1 - E2) / eta;
    alpha2_new = alpha2_new < L ? L : alpha2_new;
    alpha2_new = alpha2_new > H ? H : alpha2_new;
  }else{
    error("%s: eta = %.5f (i1 = %d, i2 = %d) is not negative", fname, eta, i1, i2);

    // evaluate the objective function at both ends of the feasible line segment following
    // equations 12.21, 12.22 and 12.23 (Platt 1999)
    double v1 = E1 + b + y1 - K11 * y1 * alpha1 - K12 * y2 * alpha2;
    double v2 = E2 + b + y2 - K12 * y1 * alpha1 - K22 * y2 * alpha2;

    double gamma = alpha1 + s * alpha2;
    double gamma_sL = gamma - s * L;
    double gamma_sH = gamma - s * H;
    double WL = gamma_sL + L - 0.5 * K11 * pow(gamma_sL, 2) - 0.5 * K22 * pow(L, 2) - s * K12 * gamma_sL * L - y1 * gamma_sL * v1 - y2 * L * v2;
    double WH = gamma_sH + H - 0.5 * K11 * pow(gamma_sH, 2) - 0.5 * K22 * pow(H, 2) - s * K12 * gamma_sH * H - y1 * gamma_sH * v1 - y2 * H * v2;

    if(WL > WH + epsilon)
      alpha2_new = L;
    else if(WH > WL + epsilon)
      alpha2_new = H;
    else{
      error("%s: unable to make progress with %d %d", fname, i1, i2);
      return false;
    }
  }

  // prevent precision problems
  if(alpha2_new < epsilon)
    alpha2_new = 0.0;
  else if(alpha2_new > c - epsilon)
    alpha2_new = c;

  if(std::abs(alpha2_new - alpha2) < epsilon)
    return false;

  // compute new value for the other Lagrange multiplier according to equation 12.8 (Platt 1999)
  alpha1_new = alpha1 + s * (alpha2 - alpha2_new);

  // prevent precision problems
  if(alpha1_new < epsilon)
    alpha1_new = 0.0;
  else if(alpha1_new > c - epsilon)
    alpha1_new = c;

  // update threshold according to section 12.2.3 (Platt 1999)
  bool valid1 = alpha1_new > 0.0 && alpha1_new < c;
  bool valid2 = alpha2_new > 0.0 && alpha2_new < c;
  double b1 = E1 + y1 * (alpha1_new - alpha1) * K11 + y2 * (alpha2_new - alpha2) * K12 + b;
  double b2 = E2 + y1 * (alpha1_new - alpha1) * K12 + y2 * (alpha2_new - alpha2) * K22 + b;
  double b_new;
  if(valid1 && valid2 && std::abs(b1 - b2) >= epsilon){
    error("%s: b1 %.15f b2 %.15f are not equal", fname, b1, b2);
    return false;
  }else if(valid1 && valid2)
    b_new = b1;
  else if(valid1)
    b_new = b1;
  else if(valid2)
    b_new = b2;
  else
    b_new = 0.5 * (b1 + b2);

  // update error cache via equation 12.11 (Platt 1999)
  // note that the error values are maintained regardless of wether the corresponding Lagrange
  // multipliers are at the bounds or not
  for(int i = 0; i < n; ++i)
    E(i) = E(i) + y1 * (alpha1_new - alpha1) * kernel_eval(i1, i) + y2 * (alpha2_new - alpha2) * kernel_eval(i2, i) + b - b_new;

  // update SVM parameters
  alpha(i1) = alpha1_new;
  alpha(i2) = alpha2_new;
  b = b_new;

  return true;
}

// select a second example to optimize via a series of heuristics
int examine_example(int i2)
{
  // verify that the selected example violates the KKT conditions
  // note that the definition of r2 = E(x_2) * y_2 = f(x_2) y_2 - 1 leads to the following simplification
  // of the KKT conditions given in equation 12.1 (Platt 1999)
  double r2 = E(i2) * y(i2);
  if((r2 < -epsilon && alpha(i2) < c) ||
     (r2 > epsilon && alpha(i2) > 0.0)){
    // first heuristic: approximately maximize the step size
    double E2 = E(i2);
    double step_max = -1.0;
    int step_i = -1;
    for(int i = 0; i < n; ++i){
      double step = std::abs(E(i) - E2);
      if(i != i2 && step > step_max){
        step_max = step;
        step_i = i;
      }
    }
    if(take_step(step_i, i2))
      return 1;

    // second heuristic: find a bound example that results in a positive change in the objective value
    int start1 = rand() % n;
    for(int i = 0; i < n; ++i){
      int i1 = (start1 + i) % n;
      if(alpha(i1) > 0.0 && alpha(i1) < c)
        if(take_step(i1, i2))
          return 1;
    }

    // third heuristic: find any example that results in a positive change in the objective value
    int start2 = rand() % n;
    for(int i = 0; i < n; ++i){
      int i1 = (start2 + i) % n;
      if(take_step(i1, i2))
        return 1;
    }
  }

  if(debug)
    std::cout << fname << ": example " << i2 << " failed to improve the objective function" << std::endl;

  return 0;
}

// validate input arguments
bool valid_arguments(const octave_value_list& args)
{
  octave_idx_type nargin = args.length();
  if(nargin < 2 || nargin > 6){
    print_usage();
    return false;
  }

  // data matrix should be a real-valued matrix
  octave_value X = args(0);
  if(!X.is_real_matrix() || X.is_string()){
    error("%s: data matrix (argument 1) is not a real-valued matrix", fname);
    return false;
  }else if(X.ndims() != 2){
    error("%s: data matrix (argument 1) does not have exaclty two dimensions", fname);
    return false;
  }else if(X.columns() == 0){
    error("%s: data matrix (argument 1) must contain one or more columns", fname);
    return false;
  }else if(X.rows() < 2){
    error("%s: data matrix (argument 1) must contain at least two instances", fname);
    return false;
  }

  // kernel matrix should not contain Inf or NaN
  Matrix MX(X.matrix_value());
  if(error_state){
    error("%s: data matrix (argument 1) is not a real-valued matrix", fname);
    return false;
  }else if(MX.any_element_is_inf_or_nan()){
    error("%s: data matrix (argument 1) contains Inf and/or NaN", fname);
    return false;
  }

  // class label vector should be a real-valued column vector
  octave_value y = args(1);
  if(!y.is_real_matrix() || y.is_string()){
    error("%s: class vector (argument 2) is not a real-valued vector", fname);
    return false;
  }else if(y.ndims() != 2){
    error("%s: class vector (argument 2) does not have exactly two dimensions", fname);
    return false;
  }else if(y.rows() == 0){
    error("%s: class vector (argument 2) is empty", fname);
    return false;
  }else if(y.columns() != 1){
    error("%s: class vector (argument 2) does not have exactly one column", fname);
    return false;
  }

  // data matrix and class vector must contain the same number of instances
  if(X.rows() != y.rows()){
    error("%s: data matrix and class vector (arguments 1 and 2) do not have the same number of instances", fname);
    return false;
  }

  // class label vector should not contain Inf or NaN
  Matrix My(y.matrix_value());
  if(error_state){
    error("%s: class vector (argument 2) is not a real-valued vector", fname);
    return false;
  }else if(My.any_element_is_inf_or_nan()){
    error("%s: class vector (argument 2) contains Inf and/or NaN", fname);
    return false;
  }

  // class labels should all be 1 or -1 and contain at least one of each
  octave_value pos(do_binary_op(octave_value::op_eq, y, 1.0));
  octave_value neg(do_binary_op(octave_value::op_eq, y, -1.0));
  octave_value valid(do_binary_op(octave_value::op_el_or, pos, neg));
  if(!valid.all().is_true()){
    error("%s: class vector (argument 2) contains value(s) not equal to +1 or -1", fname);
    return false;
  }else if(!pos.any().is_true()){
    error("%s: class vector (argument 2) must contain at least one positive instance", fname);
    return false;
  }else if(!neg.any().is_true()){
    error("%s: class vector (argument 2) must contain at least one negative instance", fname);
    return false;
  }

  // the remaining arguments are optional
  if(nargin > 2){
    octave_value c = args(2);
    if(!c.is_real_scalar()){
      error("%s: c (argument 3) is not a real-valued scalar", fname);
      return false;
    }else if(c.matrix_value().any_element_is_inf_or_nan()){
      error("%s: c (argument 3) is Inf or NaN", fname);
      return false;
    }else if(c.scalar_value() <= 0.0){
      error("%s: c (argument 3) is non-positive", fname);
      return false;
    }
  }
  if(nargin > 3){
    octave_value kernel = args(3);
    if(!kernel.is_function_handle()){
      error("%s: kernel (argument 4) is not a function", fname);
      return false;
    }
  }
  if(nargin > 4){
    octave_value epsilon = args(4);
    if(!epsilon.is_real_scalar()){
      error("%s: epsilon (argument 5) is not a real-valued scalar", fname);
      return false;
    }else if(epsilon.matrix_value().any_element_is_inf_or_nan()){
      error("%s: epsilon (argument 5) is Inf or NaN", fname);
      return false;
    }else if(epsilon.scalar_value() <= 0.0){
      error("%s: epsilon (argument 5) is non-positive", fname);
      return false;
    }
  }
  if(nargin > 5){
    octave_value debug = args(5);
    if(!debug.is_real_scalar()){
      error("%s: debug (argument 6) is not a real-valued scalar", fname);
      return false;
    }else if(debug.matrix_value().any_element_is_inf_or_nan()){
      error("%s: debug (argument 6) is Inf or NaN", fname);
      return false;
    }
  }

  return true;
}

DEFUN_DLD(smo, args, nargout,
"-*- texinfo -*-\n\
@deftypefn{Loadable Function} {[@var{alpha}, @var{b}] =}\
smo(@var{X}, @var{y}, @var{c}, @var{kernel}, @var{epsilon}, @var{debug})\n\
\n\
@cindex Support Vector Machine (SVM) learning via Sequential Minimal Optimization (SMO)\n\
\n\
Implements Platt's SMO algorithm for SVM learning.  The approach is based on iteratively\n\
solving the smallest possible sub-problems involving just two Lagrange multipliers.  These problems\n\
have an analytical solution thus the SMO algorithm is both simple and efficient.\n\
\n\
The instances are given in the matrix @var{X} one instance per row and the corresponding class\n\
labels are given in the vector @var{y}.  The argument @var{c} specifies the SVM parameter c (default\n\
is 1).  For non-linear SVMs, a kernel function reference can be provided via @var{kernel}.\n\
\n\
Numbers are considered equal if and only if they are within @var{epsilon} of each other.  The\n\
optional @var{debug} flag controls the printing of debugging information.\
@end deftypefn")
{
  octave_value_list rvalues;

  if(!valid_arguments(args))
    return rvalues;

  // required arguments
  X = Matrix(args(0).matrix_value());
  y = ColumnVector(args(1).column_vector_value());
  n = X.rows();
  p = X.columns();

  c = default_c;
  kernel = NULL;
  epsilon = default_epsilon;
  debug = default_debug;

  // optional arguments
  octave_idx_type nargin = args.length();
  if(nargin > 2)
    c = args(2).scalar_value();
  if(nargin > 3)
    kernel = args(3).function_value();
  if(nargin > 4)
    epsilon = args(4).scalar_value();
  if(nargin > 5)
    debug = args(5).is_true();

  // initialize SVM parameters and error cache
  // note that the initial parameters cause f(x) = 0 for all x
  alpha = ColumnVector(n, 0.0);
  b = 0.0;
  E = ColumnVector(-y);

  // initialize kernel cache
  kernel_cache = Matrix(n, n, 0.0);
  kernel_valid = boolMatrix(n, n, false);

  int num_changed = 0;
  bool examine_all = true;
  while(num_changed > 0 || examine_all){
    OCTAVE_QUIT; // allow user to stop execution via ^C

    num_changed = 0;
    for(int i = 0; i < n; ++i)
      if(examine_all)
        num_changed += examine_example(i);
      else if(alpha(i) > 0.0 && alpha(i) < c) // unbound subset only
        num_changed += examine_example(i);

    if(debug)
      std::cout << fname << ": " << " number changed (examine_all? " << examine_all << "): " << num_changed << std::endl;

    if(examine_all)
      examine_all = false;
    else if(num_changed == 0)
      examine_all = true;
  }
  rvalues.append(octave_value(alpha));
  rvalues.append(octave_value(b));

  return rvalues;
}
