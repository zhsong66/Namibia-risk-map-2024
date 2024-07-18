### Basic R script for spato-temporal analysis of Namibia dataset

### Stage 1 modelling ###
library(INLA)
library(TMB)

# Build INLA mesh
namibia.mesh <- inla.mesh.2d(cbind(fac.data.matched$long,fac.data.matched$lat),loc.domain=admin0@polygons[[1]]@Polygons[[1]]@coords,max.edge=c(0.75,2),cut=0.05,offset=0)
namibia.spde <- (inla.spde2.matern(namibia.mesh,alpha=2)$param.inla)[c("M0","M1","M2")] 
namibia.A <- inla.mesh.project(namibia.mesh,in.country.coords)$A 

# Use mean annual cases from 2018 to 2021
namean <- function(x) {mean(x,na.rm=TRUE)}
allages <- apply(cbind(allages_2021,allages_2020,allages_2019,allages_2018),1,namean)

# Fit catchment + incidence model
compile("namibia2022_dpois.cpp",flags = "-Ofast")
dyn.load(dynlib("namibia2022_dpois"))

input.data <- list(
  'bigN'=bigN,
  'pca_covariates'=xcovs,
  'spde'=namibia.spde,
  'A'=namibia.A,
  'nHFs'=Nfacilities_matched,
  'population'=population,
  'HFcases'=allages,
  'validHFs'=validHFs,
  'invdistsparse'=invdistsparse
)

parameters <- list(
  'intercept'=-5.321551,
  'log_masses'=rep(0,Nfacilities_matched),
  'log_range'=0,
  'log_sd'=2,
  'field'=numeric(namibia.mesh$n),
  'log_rangexcov'=0,
  'log_sdxcov'=2,
  'fieldxcov'=matrix(0,nrow=namibia.mesh$n,ncol=2),
  'log_offsetHF'=rep(0,Nfacilities_matched)
)

obj <- MakeADFun(input.data,parameters,DLL="namibia2022_dpois",random=c('field','log_masses','fieldxcov','log_offsetHF'))
obj$fn()

opt <- nlminb(obj$par,obj$fn,obj$gr,control=list(iter.max=100,eval.max=100)) 
rep <- sdreport(obj,getJointPrecision = TRUE)

parnames <- unique(names(rep$jointPrecision[1,]))
for (i in 1:length(parnames)) {
  eval(parse(text=(paste("parameters$",parnames[i]," <- c(rep$par.fixed,rep$par.random)[names(c(rep$par.fixed,rep$par.random))==\"",parnames[i],"\"]",sep=""))))}

r.draws <- rmvn.sparse(300,unlist(parameters),Cholesky(rep$jointPrecision),prec=TRUE)
output.list <- list()
for (i in 1:300) {output.list[[i]] <- obj$report(r.draws[i,])}

# Outputs
post_HF_cases <- matrix(NA,nrow=300,ncol=Nfacilities_matched)
for (i in 1:300) {post_HF_cases[i,] <- output.list[[i]]$predicted_cases}  
post_HF <- apply(post_HF_cases,2,mean)/52


### Stage 1 modelling ###
# Use GAM to predict weekly expected cases at health facility level
library(mgcv)
bam.model <- bam(case ~ baseHF +
                   te(long,lat,week, bs=c("tp","cr"), k=c(50,50), d=c(2,1)) +
                   s(HF_ID, bs="re"),
                 family=quasipoisson,
                 data=baseHF.data)
bam.expect <- predict(bam.model,baseHF.exp,type="response")     


