from __future__ import print_function

import numpy as np
import pandas as pd
from math import fsum
from scipy import optimize #version >0.13.0 for dogleg optimizer
from scipy.special import logsumexp
from sklearn.linear_model import LinearRegression
import sys
import warnings
#warnings.simplefilter("error", "RuntimeWarning")
#warnings.filterwarnings("error")


# reweight class. 
class Reweight:

    
    # initialize
    def __init__(self,name):

        self.exp_types = ["NOE","JCOUPLINGS","CS","SAXS","RDC"]
        self.bound_types = ["UPPER","LOWER"]
        self.averaging_types = ["linear","power_3","power_6","power_4"]
        self.fit_types = ["scale","scale+offset","none"]

        self.use_samples = []
        self.name = name
        self.w0 = []
        self.log = open("%s.log" % name, "w")
    
        self.experiment =  []
        self.calculated =  []
        
    def preprocess_data():
        self.log.write("# Using %s averaging \n" % (averaging))
        self.log.write("# Using %s scaling \n" % (scaling))
        
        
        
    def load_file(self,exp_file,calc_file,averaging="auto",fit="auto",use_samples=[],use_exp=[]):

        # read experimental data 
        fh = open(exp_file)
        first = fh.readline()
        assert first[0] == "#", "Error. First line of exp file %s must be in the format \# DATA=[%s] [BOUND=UPPER/LOWER]" % (exp_file,self.exp_types)

        # read data type
        data_string = (first.split("DATA=")[-1].split()[0]).strip()
        assert data_string in self.exp_types , "Error. DATA in %s must be one of the following: %s " % (exp_file,self.exp_types)
        self.log.write("###################\n")
        self.log.write("# Reading %s data \n" % (data_string))
        
        # If it is not an average but a boundary it can be specified
        bound_string=None
        if(len(first.split("BOUND="))==2):
            bound_string = (first.split("BOUND=")[-1].split()[0]).strip()
            assert bound_string in self.bound_types , "Error. %s is not known. BOUND in %s must be one of the following: %s " % (bound_string,exp_file,self.bound_types)
            self.log.write("# %s-bound data \n" % (bound_string))
        
        df_exp = pd.read_csv(exp_file,sep="\s+",header=None,comment="#")

        assert df_exp.shape[1]==3, "Error. Experimental datafile must be in the format LABEL VALUE ERROR"
        df_exp = df_exp.rename(columns={0: "label", 1: "val",2:"sigma"})

        # read calculated datafile
        df_calc = pd.read_csv(calc_file,sep="\s+",header=None,comment="#")
        #print(df_calc)
        df_calc = df_calc.drop(columns=[0])
        #print(df_calc)
        assert (df_calc.shape[1])==df_exp.shape[0],\
            "Error: Number of experimental data in %s (%d) must match the calculated data in %s (%d)" % (exp_file,df_exp.shape[0],calc_file,df_calc.shape[1])

        # write to log file
        self.log.write("# Loaded %d experimental data from %s \n" % (df_exp.shape[0],exp_file))
        self.log.write("# Loaded %d calculated samples from %s \n" % (df_calc.shape[0],calc_file))
        


        # set initial uniform weight if not initialized
        if(len(self.w0)==0):
            self.w0 = np.ones(df_calc.shape[0])/df_calc.shape[0]

            
        # determine averaging
        if(averaging=="auto"):
            if(data_string=="NOE"):
                averaging = "power_6"
            else:
                averaging = "linear"
        else:
            assert averaging in self.averaging_types, "averaging type must be in %s " % (self.averaging_types)
        self.log.write("# Using %s averaging \n" % (averaging))

        if(averaging.split("_")[0]=="power"):
            noe_power = int(averaging.split("_")[-1])
            df_exp["avg"] = np.power(df_exp["val"], -noe_power)
            df_exp["sigma2"] = (noe_power*df_exp["avg"]*df_exp["sigma"]/(df_exp["val"]))**2
            df_calc = np.power(df_calc, -noe_power)
            
            # if bound constraints, swap lower and upper
            if(bound_string=="LOWER"):
                bound_string="UPPER"
            elif(bound_string=="UPPER"):
                bound_string="LOWER"

        else:
            df_exp = df_exp.rename(columns={"val": "avg"})
            df_exp["sigma2"] = np.square(df_exp["sigma"])

            
        # determine rescaling 
        if(fit=="auto"):
            fit = "none"
            if(data_string=="RDC"):
                fit = "scale"
            if(data_string=="SAXS"):
                fit = "scale+offset"
        else:
            assert fit in self.fit_types, "fit type must be in %s " % (self.fit_types)
        self.log.write("# Using %s scaling \n" % (fit))

        # perform a linear fit (no offset)
        df_calc_stats = pd.DataFrame(np.sum(df_calc.values*self.w0[:,np.newaxis],axis=0),columns=["avg"])
        ytmp = df_exp["avg"].values.reshape(-1,1)
        xtmp = df_calc_stats["avg"].values.reshape(-1,1)
        oversigma = 1/df_exp["sigma2"].values
        
        if(fit=="scale"):
            reg = LinearRegression(fit_intercept=False).fit(xtmp,ytmp,sample_weight=oversigma)
            slope = reg.coef_[0][0]
            df_calc *= slope
            df_calc_stats["avg"] = np.sum(df_calc.values*self.w0[:,np.newaxis],axis=0)
            self.log.write("# Slope=%8.4f; r2=%5.3f \n" % (slope,reg.score(xtmp,ytmp,oversigma)))

        # perform a linear fit (with offset)
        elif(fit=="scale+offset"):
            reg = LinearRegression(fit_intercept=True).fit(xtmp,ytmp,sample_weight=oversigma)
            slope = reg.coef_[0][0]
            df_calc = slope*df_calc+reg.intercept_[0]

            df_calc_stats["avg"] = np.sum(df_calc.values*self.w0[:,np.newaxis],axis=0)
            score = reg.score(xtmp,ytmp,oversigma)
            self.log.write("# Slope=%8.4f; Offset=%8.4f; r2=%8.4f \n" % (slope,reg.intercept_,score))

        # no scaling
        else:
            if(bound_string==None):
                reg = LinearRegression(fit_intercept=True).fit(xtmp,ytmp,sample_weight=oversigma)
                slope = reg.coef_[0][0]            
                self.log.write("# Slope=%8.4f; Offset=%8.4f; r2=%5.3f \n" % (slope,reg.intercept_[0],reg.score(xtmp,ytmp,oversigma)))
                
        # preliminary data check. 
        if(bound_string==None):

            df_calc_stats["min"] = np.min(df_calc.values,axis=0)
            df_calc_stats["max"] = np.max(df_calc.values,axis=0)
            # check if data are outside the range (suspicious)
            ii_min = df_calc_stats["min"]>df_exp["avg"]+df_exp["sigma"]
            if(ii_min.sum()>0):
                self.log.write("##### WARNING ########## \n")
                self.log.write("# The minimum value of the following data is higher than the experimental average: \n")
                self.log.write("# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","min_calc"))
                for j in range(len(ii_min)):
                    if(ii_min[j]==True):
                        ss = "%15s %8.4f %8.4f %8.4f\n" % (df_exp["label"].iloc[j],df_exp["avg"].iloc[j],df_exp["sigma"].iloc[j],df_calc_stats["min"].iloc[j])
                        self.log.write(ss)
                self.log.write("##### WARNING ########## \n")
            ii_max = df_calc_stats["max"]<df_exp["avg"]-df_exp["sigma"]
            if(ii_max.sum()>0):
                self.log.write("##### WARNING ########## \n")
                self.log.write("# The maximum value of the following data is lower than the experimental average:\n")
                self.log.write("# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","min_calc"))
                for j in range(len(ii_max)):
                    if(ii_max[j]==True):
                        ss = "%15s %8.4f %8.4f %8.4f \n" % (df_exp["label"].iloc[j],df_exp["avg"].iloc[j],df_exp["sigma"].iloc[j],df_calc_stats["max"].iloc[j])
                        self.log.write(ss)
                self.log.write("##### WARNING ########## \n")
                
            # calculate RMSD and CHI2
            diff = (df_calc_stats["avg"]-df_exp["avg"])            
            chi2 = np.average((diff**2/df_exp["sigma2"]))
            rmsd = np.sqrt(np.average((diff**2)))
            self.log.write("RMSD: %8.4f \n" % rmsd)
            self.log.write("CHI2: %8.4f \n" % chi2)
            
        # different treatment for boundaries
        else:
            diff = (df_calc_stats["avg"]-df_exp["avg"])            
            if(bound_string=="LOWER"):
                ii=np.where(diff.values<0)[0]
            if(bound_string=="UPPER"):
                ii=np.where(diff.values>0)[0]
                
            if(len(ii)>0):
                self.log.write("# The ensemble violates the following boundaries: \n")
                self.log.write("# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","calc_avg"))
                for j in ii:
                    expv = df_exp["avg"].iloc[j]
                    calcv = df_calc_stats["avg"].iloc[j]
                    if(averaging!="linear"):
                        expv = np.power(expv,-1/noe_power)
                        calcv = np.power(calcv,-1/noe_power)
                    ss = "%15s %8.4f %8.4f %8.4f \n" % (df_exp["label"].iloc[j],expv,df_exp["sigma"].iloc[j],calcv)
                    self.log.write(ss)
                
            
        # standardize
        normalize = "zscore"
        if(normalize=="zscore"):
            df_calc_stats["std"] = np.sqrt(np.average((df_calc_stats["avg"].values-df_calc.values)**2, weights=self.w0,axis=0))
            df_exp["avg"] = ((df_exp["avg"]-df_calc_stats["avg"])/df_calc_stats["std"]).values
            df_calc = (df_calc-df_calc_stats["avg"].values)/df_calc_stats["std"].values
            df_exp["sigma2"] /= (df_calc_stats["std"]*df_calc_stats["std"])
            
        elif(normalize=="minmax"):
            mmin = df_calc.min(axis=0).values
            mmax = df_calc.max(axis=0).values
            delta = mmax-mmin
            df_exp["avg"] = (df_exp["avg"]-mmin)/delta
            df_calc = (df_calc-mmin)/delta
            df_exp["sigma2"] /= (delta*delta)

            
        # define bounds 
        df_exp["bound1"] = None
        df_exp["bound2"] = None
        if(bound_string=="UPPER"): df_exp["bound1"] = 0.0
        if(bound_string=="LOWER"): df_exp["bound2"] = 0.0
        df_exp["tag"] = exp_file
        
        if(len(self.experiment)==0):
            self.experiment = pd.DataFrame(df_exp[["label","avg","sigma2","bound1","bound2","tag"]])
            self.calculated = df_calc.values
        else:
            self.experiment = self.experiment.append(df_exp[["label","avg","sigma2","bound1","bound2","tag"]],ignore_index=True)
            self.calculated = np.hstack([self.calculated,df_calc.values])
            
        
        self.log.write("###################\n")


    
    def optimize(self,theta=1,lambdas_init=[]):

        def fun(lambdas):
            # weights
            arg = -np.sum(lambdas[np.newaxis,:]*self.calculated,axis=1)
            arg -= tmax
            
            #########
            ww = (self.w0*np.exp(arg))
            zz = np.sum(ww)
            assert np.isfinite(zz), "# Error. sum of weights is infinite. Use higher theta"
            #logz = logsumexp(arg,b=self.w0)
            #zz = np.exp(logz)
            
            #ww /= np.exp(logz)
            logz = np.log(zz)
            ww /= zz
            avg = np.sum(ww[:,np.newaxis]*self.calculated, axis=0)
            
            # gaussian integral
            eps2 = 0.5*np.sum((lambdas*lambdas)*err) 
            
            # experimental value 
            sum1 = np.dot(lambdas,my_avg)
            fun = sum1 + eps2+ logz
            
            # gradient
            #jac = self.experiment["avg"].values + lambdas*err - avg
            jac = my_avg + lambdas*err - avg
            

            # divide by theta to avoid numerical problems
            return  fun/theta,jac/theta
    
        def hess(lambdas):
            arg = -np.sum(lambdas[np.newaxis,:]*self.calculated,axis=1)
            arg -= tmax
            
            #########
            ww = (self.w0*np.exp(arg))
            zz = np.sum(ww)
            assert np.isfinite(zz), "# Error. sum of weights is infinite. Use higher theta"
            ww /= zz

            q_w = np.dot(ww,self.calculated)
            hess = np.einsum('k, ki, kj->ij',ww,self.calculated,self.calculated) - np.outer(q_w,q_w) + np.diag(err)

            return  hess/theta
        
        if(len(lambdas_init)==0):
            lambdas=np.zeros(self.experiment.shape[0])
        else:
            lambdas = np.array(lambdas_init)
        #opt={'maxiter':50000,'disp':False,'ftol': 1e-10 }
        opt={'maxiter':50000,'disp':True}
        # errors are rescaled by factor theta
        err = (theta)*(self.experiment["sigma2"].values) # this can be done outside this loop

        tmax = np.log((sys.float_info.max)/2.)
        my_avg = self.experiment["avg"].values



        meth = "L-BFGS-B"
        result = optimize.minimize(fun,lambdas,options=opt,method=meth,jac=True,bounds=self.experiment[["bound1","bound2"]].values)
        
        meth="trust-constr"
        result = optimize.minimize(fun,lambdas,options=opt,method=meth,jac=True,hess=hess,bounds=self.experiment[["bound1","bound2"]].values)
        #result = optimize.minimize(fun,lambdas,options=opt,method=meth,jac=True)
        
        arg = -np.sum(result.x[np.newaxis,:]*self.calculated,axis=1)
        arg -= tmax
        w_opt = self.w0*np.exp(arg)
        w_opt /= np.sum(w_opt)

        fhw=open("crap.dat","w")
        for k in w_opt:
            fhw.write("%8.4e\n" % k)


        avg1 = np.sum(self.calculated*self.w0[:,np.newaxis],axis=0)
        avg2 = np.sum(self.calculated*w_opt[:,np.newaxis],axis=0)
        diff1 = (avg1-self.experiment["avg"].values)
        diff2 = (avg2-self.experiment["avg"].values)
        i_tozero1 = np.where(((diff1<=0) & (self.experiment["bound1"]==0)) | ((diff1>=0) & (self.experiment["bound2"]==0) ))
        i_tozero2 = np.where(((diff2<=0) & (self.experiment["bound1"]==0)) | ((diff2>=0) & (self.experiment["bound2"]==0) ))
        diff1[i_tozero1] = 0.0
        diff2[i_tozero2] = 0.0
        #print(self.experiment["sigma2"])
        chi1 = np.average((diff1**2/self.experiment["sigma2"].values))
        chi2 = np.average((diff2**2/self.experiment["sigma2"].values))
        print(chi1,chi2)
        #for name,group in self.experiment.groupby('tag'):
        #    avg  = 
        #    print(name,group)
                                  
