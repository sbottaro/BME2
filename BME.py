from __future__ import print_function
import time
import numpy as np
from scipy import optimize #version >0.13.0 for dogleg optimizer
import sys
import warnings
import BME_tools as bt

#warnings.simplefilter("error", "RuntimeWarning")
#warnings.filterwarnings("error")

known_methods = ["BME","BER","CHI2_L2","CHI1_L1"]

# reweight class. 
class Reweight:

    
    # initialize
    def __init__(self,name,w0=[]):

        self.name = name

        if(len(w0)==0):
            self.w0 = []
        else:
            self.w0 = w0/np.sum(w0)
        self.w_opt = []
        self.lambdas = []
        
        self.log = open("%s.log" % name, "w")
        
        self.labels = []
        self.experiment =  []
        self.calculated =  []
                        
        
    def read_file(self,exp_file,calc_file,averaging="auto",fit='no',use_samples=[],use_data=[]):

        # read file
        log = ""
        label,exp,calc,log,averaging = bt.parse(exp_file,calc_file,averaging=averaging)
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

    def load(self,exp_file,calc_file,averaging="auto",fit='no',use_samples=[],use_data=[],weight=1):
        
        label,exp, calc = self.read_file(exp_file,calc_file,averaging=averaging,fit=fit,use_samples=use_samples,use_data=use_data)

        if(len(self.experiment)==0):
            self.experiment = exp
            self.calculated = calc
            self.labels = label
            self.weights = np.ones(exp.shape[0])*weight
        else:
            self.experiment = np.vstack([self.experiment,exp])
            self.calculated = np.hstack([self.calculated,calc])
            self.labels = np.hstack([self.labels,label])
            self.weights = np.hstack([self.weights,np.ones(exp.shape[0])*weight])
        
        # note to self: implement weight

    # add data from external array
    def load_array(self,label,exp,calc,weight=1):
        
        if(len(self.experiment)==0):
            self.experiment = exp
            self.calculated = calc
            self.labels = label
            self.w0 = np.ones(calc.shape[0])/calc.shape[0]
            self.weights = np.ones(exp.shape[0])*weight
        else:
            self.experiment = np.vstack([self.experiment,exp])
            self.calculated = np.hstack([self.calculated,calc])
            self.labels = np.hstack([self.labels,label])
            self.weights = np.hstack([self.weights,np.ones(exp.shape[0])*weight])
            
    def predict(self,exp_file,calc_file,outfile=None,averaging="auto",fit='no',use_samples=[],use_data=[]):

        label,exp,calc,log,averaging = bt.parse(exp_file,calc_file,averaging=averaging)
        
        self.log.write(log)
        
        # remove datapoints if use_samples or use_data is not empty
        exp, calc, log = bt.subsample(label,exp,calc,use_samples,use_data)
        self.log.write(log)

        # do sanity checks
        stats  = bt.calc_stats(label,exp,calc,self.w0,self.w_opt,averaging=averaging,fit=fit,outfile=outfile)

        return stats
    
    def predict_array(self,label,exp,calc,outfile=None):

        return bt.calc_stats(label,exp,calc,self.w0,self.w_opt,averaging="linear",outfile=outfile)
        
        # do sanity checks
        #log  = bt.check_data(label,exp,calc,self.w_opt)
        #self.log.write(log)

    def get_lambdas(self):
        return self.lambdas

    def get_weights(self):
        return self.w_opt

    # optimize
    def fit(self,theta,lambdas_init=True,method="BME"):

        assert method in known_methods, "method %s not in known methods:" % (method,known_methods)
        
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
            #return  fun,jac
    
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

        def func_ber_gauss(w):

            bcalc = np.sum(w[:,np.newaxis]*self.calculated,axis=0)
            diff = bcalc-self.experiment[:,0]
            #print(diff)
            #ii = np.where(((diff<0) & (self.experiment[:,2]<0)) | ((diff>0) & (self.experiment[:,2]>0)) )[0#]
 #           ff = [1 if (self.experiment[j,2]==0 or j in ii) else 0 for j in range(self.experiment.shape[0])]
            #diff *= ff
            
            chi2_half =  0.5*np.sum(((diff/self.experiment[:,1])**2))

            
            idxs = np.where(w>1.0E-50)
            log_div = np.zeros(w.shape[0])
            log_div[idxs] = np.log(w[idxs]/self.w0[idxs])
            srel = theta*np.sum(w*log_div)

            jac = np.sum(diff*self.calculated,axis=1) + theta*(1.+log_div)
            return chi2_half+srel#, jac
    

        
        def func_chi2_L2(w):

            bcalc = np.sum(w[:,np.newaxis]*self.calculated,axis=0)
            diff = (bcalc-self.experiment[:,0])/self.experiment[:,1]
            
            ii = np.where(((diff<0) & (self.experiment[:,2]<0)) | ((diff>0) & (self.experiment[:,2]>0)) )[0]
            ff = [1 if (self.experiment[j,2]==0 or j in ii) else 0 for j in range(self.experiment.shape[0])]

            diff *= ff
            
            chi2_half =  0.5*np.sum(diff**2)

            jac = np.sum(diff*self.calculated,axis=1)
            #idxs = np.where(w>1.0E-50)
                                                                                                                                              #srel = theta*np.sum(w[idxs]*np.log(w[idxs]/self.w0[idxs]))
                                                                                                                                              #jac = 
            return chi2_half
        
        def func_chi2_L1(w):

            bcalc = np.sum(w[:,np.newaxis]*self.calculated,axis=0)
            diff = (bcalc-self.experiment[:,0])/self.experiment[:,1]
            
            ii = np.where(((diff<0) & (self.experiment[:,2]<0)) | ((diff>0) & (self.experiment[:,2]>0)) )[0]
            ff = [1 if (self.experiment[j,2]==0 or j in ii) else 0 for j in range(self.experiment.shape[0])]

            diff *= ff
            
            chi2_half =  0.5*np.sum(diff**2)

            jac = np.sum(diff*self.calculated,axis=1)
            #idxs = np.where(w>1.0E-50)
                                                                                                                                              #srel = theta*np.sum(w[idxs]*np.log(w[idxs]/self.w0[idxs]))
                                                                                                                                              #jac = 
            return chi2_half,jac

        if(lambdas_init==True):
            lambdas=np.zeros(self.experiment.shape[0],dtype=np.longdouble)
            self.log.write("Lagrange multipliers initialized from zero\n")
        else:
            assert(len(self.lambdas)==self.experiment.shape[0])
            lambdas = np.copy(self.lambdas)
            #np.array(lambdas_init)
            self.log.write("Warm start\n")
        #print(lambdas)
            
        bounds = []
        for j in range(self.experiment.shape[0]):
            if(self.experiment[j,2]==0):
                bounds.append([None,None])
            elif(self.experiment[j,2]==-1):
                bounds.append([None,0.0])
            else:
                bounds.append([0.0,None])

        if(method=="BME"):
            opt={'maxiter':50000,'disp':False}
            
            tmax = np.log((sys.float_info.max)/5.)

            theta_sigma2 = theta*self.weights*self.experiment[:,1]**2

            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            self.log.write("Optimizing %d data and %d samples. Theta=%f \n" % (self.experiment.shape[0],self.calculated.shape[0],theta))
            self.log.write("CHI2 before optimization: %8.4f \n" % (chi2_before))
            mini_method = "L-BFGS-B"
            start_time = time.time()
            #if(all(self.experiment[:,2]==0)):
            #    mini_method="trust-constr"
            #    result = optimize.minimize(maxent,lambdas,\
            #                               options=opt,method=mini_method,\
            #                               jac=True,hess=maxent_hess)
#
            #else:
            result = optimize.minimize(maxent,lambdas,\
                                       options=opt,method=mini_method,\
                                       jac=True,bounds=bounds)
            self.log.write("Execution time: %.2f seconds\n" % (time.time() - start_time))
            
            if(result.success):
                self.log.write("Minimization using %s successful (iterations:%d)\n" % (mini_method,result.nit))
                arg = -np.sum(result.x[np.newaxis,:]*self.calculated,axis=1) -tmax
                w_opt = self.w0*np.exp(arg)
                w_opt /= np.sum(w_opt)
                self.lambdas = np.copy(result.x)
                self.w_opt = np.copy(w_opt)

                chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
                phi = np.exp(-bt.srel(self.w0,w_opt))
                
                self.log.write("CHI2 after optimization: %8.4f \n" % (chi2_after))
                self.log.write("Fraction of effective frames: %8.4f \n" % (phi))
                return chi2_before,chi2_after,phi
            
            else:
                self.log.write("Minimization using %s failed\n" % (mini_method))
                self.log.write("Message: %s\n" % (result.message))
                return np.NaN, np.NaN, np.NaN
            
            

        # please check 
        if(method=="BER"):
            
            opt={'maxiter':2000,'disp': True,'ftol':1.0e-20}
            cons = {'type': 'eq', 'fun':lambda x: np.sum(x)-1.0}
            bounds = [(0.,None)]*len(self.w0)
            mini_method = "SLSQP"
            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            self.log.write("CHI2 before optimization: %8.4f \n" % (chi2_before))

                                                                                                                                            
            w0 = np.copy(self.w0)
            start_time = time.time()
            #print(func_ber_gauss(w0))
            #result = optimize.minimize(func_ber_gauss,w0,constraints=cons,options=opt,method=mini_method,bounds=bounds,jac=True)
            result = optimize.minimize(func_ber_gauss,w0,constraints=cons,options=opt,method=mini_method,bounds=bounds,jac=False)
            self.log.write("Execution time: %.2f seconds\n" % (time.time() - start_time))
            if(result.success):
                self.log.write("Minimization using %s successful (iterations:%d)\n" % (mini_method,result.nit))
                w_opt = np.copy(result.x)
                self.w_opt = w_opt
                chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
                phi = np.exp(-bt.srel(self.w0,w_opt))
                self.log.write("CHI2 after optimization: %8.4f \n" % (chi2_after))
                self.log.write("Fraction of effective frames: %8.4f \n" % (phi))
                return chi2_before,chi2_after,phi
            
            else:
                self.log.write("Minimization using %s failed\n" % (mini_method))
                self.log.write("Message: %s\n" % (result.message))
                return np.NaN, np.NaN, np.NaN
   
        # please check 
        if(method=="CHI2_L2"):
            
            opt={'maxiter':2000,'disp': True,'ftol':1.0e-20}
            cons = {'type': 'eq', 'fun':lambda x: np.sum(x)-1.0}
            bounds = [(0.,None)]*len(self.w0)
            meth = "SLSQP"
            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            self.log.write("CHI2 before optimization: %8.4f \n" % (chi2_before))
            start_time = time.time()
            result = optimize.minimize(func_chi2_L2,self.w0,constraints=cons,options=opt,method=meth,jac=True,bounds=bounds)

            w_opt = np.copy(result.x)
            self.w_opt = w_opt
            chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
            phi = np.exp(-bt.srel(self.w0,w_opt))
            self.log.write("Execution time: %.2f seconds\n" % (time.time() - start_time))
            self.log.write("CHI2 after optimization: %8.4f \n" % (chi2_after))

        # please check 
        if(method=="CHI2_L1"):
            
            opt={'maxiter':2000,'disp': True,'ftol':1.0e-20}
            cons = {'type': 'eq', 'fun':lambda x: np.sum(x)-1.0}
            bounds = [(0.,None)]*len(self.w0)
            meth = "SLSQP"
            chi2_before  = bt.calc_chi(self.experiment,self.calculated,self.w0)
            self.log.write("CHI2 before optimization: %8.4f \n" % (chi2_before))
            start_time = time.time()
            result = optimize.minimize(func_chi2_L2,self.w0,constraints=cons,options=opt,method=meth,jac=True,bounds=bounds)

            w_opt = np.copy(result.x)
            self.w_opt = w_opt
            chi2_after  = bt.calc_chi(self.experiment,self.calculated,w_opt)
            phi = np.exp(-bt.srel(self.w0,w_opt))
            self.log.write("Execution time: %.2f seconds\n" % (time.time() - start_time))
            self.log.write("CHI2 after optimization: %8.4f \n" % (chi2_after))


    def theta_scan(self,thetas=[],train_fraction_data=0.75,nfold=5,train_fraction_samples=0.8):

        np.random.seed(42)
        if(len(thetas)==0):
            thetas = [1000,100,10,1]
            
        nsamples = self.calculated.shape[0]
        ndata =   self.experiment.shape[0]
        train_samples =  int(nsamples*train_fraction_samples)
        train_data =  int(ndata*train_fraction_data)

        fhx = open(self.name + ".xval.dat","w")
        fhx.write("# %d fold cross-validation \n" % nfold)
        fhx.write("# %f training fraction (data)\n" % train_fraction_data)
        fhx.write("# %f training fraction (samples)\n" % train_fraction_samples)
        fhx.write("# index theta chi2_0_train RMSD_0_train violations_0_train chi2_0_train RMSD_0_train violations_0_train chi2_0_test RMSD_0_test violations_0_test chi2_0_test RMSD_0_test violations_0_test \n")

        results = np.zeros((nfold,len(thetas),13))
        for i in range(nfold):
            shuffle_samples = np.arange(nsamples)
            shuffle_data = np.arange(ndata)
            np.random.shuffle(shuffle_samples)
            np.random.shuffle(shuffle_data)

            train_idx_data = shuffle_data[:train_data]
            train_idx_samples = shuffle_samples[:train_samples]
            test_idx_data = shuffle_data[train_data:]
            test_idx_samples = shuffle_samples[:train_samples]  # test samples are the same as train!
            
            r1 = Reweight("theta_scan_%s_%d" % (self.name,i))
            labels_train = [self.labels[k] for k in train_idx_data]
            labels_test = [self.labels[k] for k in test_idx_data]
            
            exp_train = self.experiment[train_idx_data,:]
            calc_train = self.calculated[:,train_idx_data]
            
            exp_test = self.experiment[test_idx_data,:]
            calc_test = self.calculated[:,test_idx_data]
            
            r1.load_array(labels_train,exp_train,calc_train[train_idx_samples,:])
            for j,t in enumerate(thetas):
                l_init = False
                if(j==0):
                    l_init=True
                c1,c2,phi = r1.fit(t,lambdas_init=l_init)
                train_stats = r1.predict_array(labels_train,exp_train,calc_train[train_idx_samples,:])
                test_stats = r1.predict_array(labels_test,exp_test,calc_test[test_idx_samples,:]) + [phi]
                train_string = ["%.3f" % x for x in train_stats]
                test_string = ["%.3f" % x for x in test_stats] 
                ss = "%d %.2f %s %s \n"  % (i,t," ".join(train_string)," ".join(test_string))
                fhx.write(ss)
                results[i,j,:6] = train_stats
                results[i,j,6:] = test_stats
                
            fhx.write("\n")
            fhx.write("\n")


        for j,t in enumerate(thetas):
            avgs = np.average(results[:,j,:],axis=0)
            avgs_string = ["%.3f" % el  for el in avgs]
            fhx.write("AVG %.2f %s \n"  % (t," ".join(avgs_string)))
            
