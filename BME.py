from __future__ import print_function

import numpy as np
from scipy import optimize #version >0.13.0 for dogleg optimizer
import sys
import warnings
import BME_tools as bt

#warnings.simplefilter("error", "RuntimeWarning")
#warnings.filterwarnings("error")


# reweight class. 
class Reweight:

    
    # initialize
    def __init__(self,name):

        self.name = name
        self.w0 = []
        self.wopt = []
        self.log = open("%s.log" % name, "w")
        self.labels = []
        self.experiment =  []
        self.calculated =  []
                        
        
    def ReadFile(self,exp_file,calc_file,averaging="auto",fit='no',use_samples=[],use_data=[]):

        # read file
        
        label,exp,calc,log = bt.parse(exp_file,calc_file,averaging="auto")
        self.log.write(log)
        
        # remove datapoints if use_samples or use_data is not empty
        exp, calc, log = bt.subsample(label,exp,calc,use_samples,use_data)
        self.log.write(log)

        if(len(self.w0)==0):
            self.w0 = np.ones(calc.shape[0])/calc.shape[0]
            self.log.write("Initialized uniform weights %d\n" % calc.shape[0])

        # fit/scale
        calc_avg,log = bt.fit_and_scale(exp,calc,self.w0,fit=fit)
        self.log.write(log)

        # do sanity checks
        log  = bt.check_data(label,exp,calc,self.w0)
        self.log.write(log)

        log = bt.standardize(exp,calc,self.w0)
        self.log.write(log)
        
        return label,exp,calc

    def LoadFile(self,exp_file,calc_file,averaging="auto",fit='no',use_samples=[],use_data=[],weight=1):
        
        label,exp, calc = self.ReadFile(exp_file,calc_file,averaging=averaging,fit=fit,use_samples=use_samples,use_data=use_data)

        if(len(self.experiment)==0):
            self.experiment = exp
            self.calculated = calc
            self.labels = label
        else:
            self.experiment = np.vstack([self.experiment,exp])
            self.calculated = np.hstack([self.calculated,calc])
            self.labels = np.hstack([self.labels,label])
        
        # note to self: implement weight

    # add data from external array
    def LoadData(self,label,exp,calc,weight=1):
        return 0

    # optimize
    def Optimize(self,theta,lambdas_init=[],method="BME"):

        def maxent(lambdas):
            # weights
            arg = -np.sum(lambdas[np.newaxis,:]*self.calculated,axis=1) - tmax
            #arg -= tmax
            
            #########
            ww = (self.w0*np.exp(arg))
            zz = np.sum(ww)
            assert np.isfinite(zz), "# Error. sum of weights is infinite. Use higher theta"
            
            ww /= zz
            avg = np.sum(ww[:,np.newaxis]*self.calculated, axis=0)
            
            # gaussian integral
            eps2 = 0.5*np.sum((lambdas*lambdas)*theta_sigma2) 
            
            # experimental value 
            sum1 = np.dot(lambdas,self.experiment[:,0])
            fun = sum1 + eps2 + np.log(zz)
            
            # gradient
            #jac = self.experiment["avg"].values + lambdas*err - avg
            jac = self.experiment[:,0] + lambdas*theta_sigma2 - avg
            

            # divide by theta to avoid numerical problems
            #print(fun)
            return  fun/theta,jac/theta
    
        def maxent_hess(lambdas):
            arg = -np.sum(lambdas[np.newaxis,:]*self.calculated,axis=1) -tmax
            #arg -= tmax
            
            #########
            ww = (self.w0*np.exp(arg))
            zz = np.sum(ww)
            assert np.isfinite(zz), "# Error. sum of weights is infinite. Use higher theta"
            ww /= zz

            q_w = np.dot(ww,self.calculated)
            hess = np.einsum('k, ki, kj->ij',ww,self.calculated,self.calculated) - np.outer(q_w,q_w) + np.diag(theta_sigma2)

            return  hess/theta


        
        if(len(lambdas_init)==0):
            lambdas=np.zeros(self.experiment.shape[0])
            self.log.write("Lagrange multipliers initialized from zero\n")
        else:
            lambdas = np.array(lambdas_init)
            self.log.write("Custom initial lagrange multipliers\n")
            
        if(method=="BME"):
            opt={'maxiter':50000,'disp':False}
            
            tmax = np.log((sys.float_info.max)/5.)
            bounds = []
            for j in range(self.experiment.shape[0]):
                if(self.experiment[j,2]==0):
                    bounds.append([None,None])
                elif(self.experiment[j,2]==-1):
                    bounds.append([None,0.0])
                else:
                    bounds.append([0.0,None])

            theta_sigma2 = theta*self.experiment[:,1]**2

            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            
            #if(all(self.experiment[:,2]==5)):
            #    mini_method="trust-constr"
            #    result = optimize.minimize(maxent,lambdas,\
            #                               options=opt,method=mini_method,\
            #                               jac=True,hess=maxent_hess)
            mini_method = "L-BFGS-B"            
            result = optimize.minimize(maxent,lambdas,\
                                       options=opt,method=mini_method,\
                                       jac=True,bounds=bounds)
        
        
            arg = -np.sum(result.x[np.newaxis,:]*self.calculated,axis=1) -tmax
            w_opt = self.w0*np.exp(arg)
            w_opt /= np.sum(w_opt)
            self.wopt = np.copy(w_opt)
            self.lambdas = np.copy(result.x)
            chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
            phi = np.exp(-bt.srel(self.w0,w_opt))
            
            #print(chi2_before,chi2_after,neff)
            #fhw=open("crap.dat","w")
            #for k in w_opt:
            #    fhw.write("%8.4e\n" % k)
            #f##hw.close()
            
            return chi2_before,chi2_after,phi
        
        
        
