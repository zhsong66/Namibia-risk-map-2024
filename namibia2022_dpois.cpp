#include <TMB.hpp>
template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace R_inla;
  using namespace density;
  using namespace Eigen;
  
  // Input Data
  
  DATA_INTEGER(bigN); // number of pixels in image plane
  DATA_MATRIX(pca_covariates);
  DATA_STRUCT(spde,spde_t); // INLA SPDE object (components of precision matrix)
  DATA_SPARSE_MATRIX(A); // INLA SPDE projection matrix: mesh to pixels [dim: bigN x nMesh]
  DATA_INTEGER(nHFs); // number of (aggregated) health facilities
  DATA_VECTOR(population); // [dim: bigN]
  DATA_VECTOR(HFcases); // [dim: nHFs]  
  DATA_VECTOR(validHFs); // [dim: nHFs]  
  DATA_SPARSE_MATRIX(invdistsparse);
  
  // Parameters
  
  PARAMETER(intercept);
  PARAMETER_VECTOR(log_masses); // [length: nHFs]
  PARAMETER(log_range);
  PARAMETER(log_sd);
  PARAMETER_VECTOR(field); // [dim: nMesh]
  PARAMETER(log_rangexcov);
  PARAMETER(log_sdxcov);
  PARAMETER_ARRAY(fieldxcov); // [dim: nMesh x 2]
  PARAMETER_VECTOR(log_offsetHF); //[length:nHFs]

  // Parameter Transforms
  
  Type range = exp(log_range);
  Type kappa = 2.8284/range;
  Type sd = exp(log_sd);
  Type rangexcov = exp(log_rangexcov);
  Type kappaxcov = 2.8284/rangexcov;
  Type sdxcov = exp(log_sdxcov);
  vector<Type> masses(nHFs);
  masses = exp(log_masses);
  vector<Type> offsetHF(nHFs);
  offsetHF = exp(log_offsetHF);

  // Priors
  
  Type nll = 0.0;
  
  nll -= (dnorm(log_masses,Type(0.0),Type(0.25),true)).sum();
  nll -= (dnorm(log_offsetHF,Type(0.0),Type(0.5),true)).sum();
  nll -= dnorm(log_sd,Type(-1),Type(0.5),true);
  nll -= dnorm(log_range,Type(1),Type(0.5),true);
  SparseMatrix<Type> Q = Q_spde(spde,kappa);
  nll += SCALE(GMRF(Q),sd)(field);
  nll -= dnorm(log_sdxcov,Type(-1),Type(0.5),true);
  nll -= dnorm(log_rangexcov,Type(1),Type(0.5),true);
  SparseMatrix<Type> Qxcov = Q_spde(spde,kappaxcov);
  nll += SCALE(GMRF(Qxcov),sdxcov)(fieldxcov.matrix().col(0));
  nll += SCALE(GMRF(Qxcov),sdxcov)(fieldxcov.matrix().col(1));
  
  // Algebra: Construct baseline field 
  
  vector<Type> baseline_field(bigN);
  baseline_field= A*field;
  vector<Type> static_field(bigN);
  static_field = (pca_covariates.array()*(A*fieldxcov.matrix()).array()).matrix().rowwise().sum();
  
  vector<Type> predicted_surface_malaria(bigN);
  predicted_surface_malaria = intercept + static_field + baseline_field;
  
  vector<Type> cbuffer(bigN); // penalty against pixels with very high incidence rates
  cbuffer = exp(predicted_surface_malaria);
  cbuffer = cbuffer*Type(2.0); // x 2 to 'centre' penalty at >500 cases per 1000 PYO
  nll += pow(cbuffer,Type(4.0)).sum(); // penalty must be nicely differentiable

  vector<Type> predicted_surface_malaria_effrate(bigN);
  predicted_surface_malaria_effrate = exp(predicted_surface_malaria)*population;
  
  // Algebra: Catchments
  
  typedef typename Eigen::SparseMatrix<Type>::InnerIterator Iterator;
  Eigen::SparseMatrix<Type> catchments(bigN,nHFs);
  catchments = invdistsparse;
  for (int k=0; k<catchments.outerSize(); ++k) {
    for (Iterator it(catchments,k); it; ++it) {
      it.valueRef() = masses(k)*it.value(); // unnormalised treatment propensity is mass / distance^2
    }
  }
  
  vector<Type> catchments_total(bigN);
  for (int i=0; i<bigN; i++) {catchments_total[i] = 0.00001;} // avoid any zero/zero errors
  for (int i=0; i<bigN; i++) {catchments_total[i] = catchments_total[i] + catchments.row(i).sum();}
  catchments = catchments.transpose();
  for (int i=0; i<bigN; i++) {catchments.col(i) = (Type(1.0)/catchments_total[i])*catchments.col(i);}  //normalising step
  
  // Algebra: Expected Cases
  
  vector<Type> predicted_cases(nHFs);
  predicted_cases = catchments* predicted_surface_malaria_effrate*offsetHF;
  
  // Likelihood
  
  for (int i=0; i<nHFs; i++) {
    if (validHFs[i]==1) {
      nll -= dpois(HFcases[i],predicted_cases[i],true);
    }
  }
  
  // Reporting
  REPORT(predicted_surface_malaria);
  REPORT(predicted_cases);
  REPORT(catchments);
  
  return nll;
}
