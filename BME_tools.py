from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

exp_types = ["NOE","JCOUPLINGS","CS","SAXS","RDC"]
bound_types = ["UPPER","LOWER"]
averaging_types = ["linear","power_3","power_6","power_4"]
fit_types = ["scale","scale+offset","no"]


def srel(w0,w1):
    idxs = np.where(w1>1.0E-50)
    return np.sum(w1[idxs]*np.log(w1[idxs]/w0[idxs]))

def parse(exp_file,calc_file,averaging="auto"):

    log = ""
    
    # read experimental data 
    fh = open(exp_file)
    first = fh.readline()
    assert first[0] == "#", "Error. First line of exp file %s must be in the format \# DATA=[%s] [BOUND=UPPER/LOWER]" % (exp_file,self.exp_types)

    # read data type
    data_string = (first.split("DATA=")[-1].split()[0]).strip()
    assert data_string in exp_types , "Error. DATA in %s must be one of the following: %s " % (exp_file,exp_types)


    log += "# Reading %s data \n" % (data_string)
        
    # If it is not an average but a boundary it can be specified
    bound_string=None
    if(len(first.split("BOUND="))==2):
        bound_string = (first.split("BOUND=")[-1].split()[0]).strip()
        assert bound_string in bound_types , "Error. %s is not known. BOUND in %s must be one of the following: %s " % (bound_string,exp_file,bound_types)
        log += "# %s-bound data \n" % (bound_string)
        
    df_exp = pd.read_csv(exp_file,sep="\s+",header=None,comment="#")
        
    assert df_exp.shape[1]==3, "Error. Experimental datafile must be in the format LABEL VALUE ERROR"
    df_exp = df_exp.rename(columns={0: "label", 1: "val",2:"sigma"})
        
    # read calculated datafile
    df_calc = pd.read_csv(calc_file,sep="\s+",header=None,comment="#")
    # Drop frame
    df_calc = df_calc.drop(columns=[0])
    #print(df_calc)
    assert (df_calc.shape[1])==df_exp.shape[0],\
        "Error: Number of experimental data in %s (%d) must match the calculated data in %s (%d)" % (exp_file,df_exp.shape[0],calc_file,df_calc.shape[1])

    # write to log file
    log += "# Reading %d experimental data from %s \n" % (df_exp.shape[0],exp_file)
    log += "# Reading %d calculated samples from %s \n" % (df_calc.shape[0],calc_file)
        
            
    # determine averaging
    if(averaging=="auto"):
        if(data_string=="NOE"):
            averaging = "power_6"
        else:
            averaging = "linear"
    else:
        assert averaging in averaging_types, "averaging type must be in %s " % (averaging_types)
    log += "# Using %s averaging \n" % (averaging)

    if(averaging.split("_")[0]=="power"):
        noe_power = int(averaging.split("_")[-1])
        df_exp["avg"] = np.power(df_exp["val"], -noe_power)
        df_exp["sigma2"] = (noe_power*df_exp["avg"]*df_exp["sigma"]/(df_exp["val"]))
        df_calc = np.power(df_calc, -noe_power)
            
        # if bound constraints, swap lower and upper
        if(bound_string=="LOWER"):
            bound_string="UPPER"
        elif(bound_string=="UPPER"):
            bound_string="LOWER"
    else:
        df_exp = df_exp.rename(columns={"val": "avg"})
        df_exp = df_exp.rename(columns={"sigma":"sigma2"})

    # define bounds 
    df_exp["bound"] = 0
    if(bound_string=="UPPER"): df_exp["bound"] = 1.0
    if(bound_string=="LOWER"): df_exp["bound"] = -1.0
    #df_exp["tag"] = exp_file

    labels = df_exp["label"].values
    exp = np.array(df_exp[["avg","sigma2","bound"]].values)
    calc = np.array(df_calc.values)
    return labels,exp,calc,log

    
def fit_and_scale(exp,calc,sample_weights,fit):

    assert fit in fit_types, "fit type must be in %s " % (fit_types)

    log = "# Using %s scaling \n" % (fit)
    # perform a linear fit 
    calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0).reshape(-1,1)
    exp_avg = exp[:,0].reshape(-1,1)
    oversigma = (1./exp[:,1]**2)

    #print(calc_avg)
    print(np.sqrt(np.average((calc_avg-exp_avg)**2)))
    if(fit=="scale"): 
        reg = LinearRegression(fit_intercept=False).fit(calc_avg,exp_avg,sample_weight=oversigma)

        slope = reg.coef_[0]
        calc *= slope
        calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0).reshape(-1,1)
        score = reg.score(calc_avg,exp_avg,oversigma)
        print(np.sqrt(np.average((calc_avg-exp_avg)**2)))
        log += "# Slope=%8.4f; r2=%5.3f \n" % (slope,score)
        
    elif(fit=="scale+offset"):
        reg = LinearRegression(fit_intercept=True).fit(calc_avg,exp_avg,sample_weight=oversigma)
        slope = reg.coef_[0][0]
        intercept = reg.intercept_[0]
        calc *= slope
        calc += intercept
    
        calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0).reshape(-1,1)
        print(np.sqrt(np.average((calc_avg-exp_avg)**2)))
        score = reg.score(calc_avg,exp_avg,oversigma)
        log = "# Slope=%8.4f; Offset=%8.4f; r2=%8.4f \n" % (slope,reg.intercept_[0],score)
        
    return calc_avg,log


def subsample(label,exp,calc,use_samples,use_data):
    
    log = ""
    if(len(use_samples)!=0):
        calc = calc[use_samples,:]
        log += "# Using a subset of samples (%d) \n" % (calc.shape[0])
    if(len(use_data)!=0):
        label = label[use_data]
        exp = exp[use_data,:]
        calc = calc[:,use_data]
        log += "# Using a subset of datapoints (%d) \n" % (exp.shape[0])
    
    return exp, calc,log

def calc_chi(exp,calc,sample_weights):
    
    calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0)

    diff = (calc_avg-exp[:,0])    
    ii = np.where(((diff<0) & (exp[:,2]<0)) | ((diff>0) & (exp[:,2]>0)) )[0]
    ff = [1 if (exp[j,2]==0 or j in ii) else 0 for j in range(exp.shape[0])]
    #ff[ii] = 1
    diff *= ff #to_zero
    return  np.average((diff/exp[:,1])**2)

    
def check_data(label,exp,calc,sample_weights):

    calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0)

    log = ""

    diff = (calc_avg-exp[:,0])
    ii = np.where(((diff<0) & (exp[:,2]<0)) | ((diff>0) & (exp[:,2]>0)) )[0]
    ff = [1 if (exp[j,2]==0) else 0 for j in range(exp.shape[0])]
    #to_zero = np.zeros(len(diff))
    if(len(ii)>0):
        log += "# The ensemble violates the following %d boundaries: \n" % (len(ii))
        log += "# %14s %8s %8s \n" % ("label","exp_avg","calc_avg")
        for j in ii:
            log  += "%15s %8.4f %8.4f \n" % (label[j],exp[j,0],calc_avg[j])
            ff[j] = 1
    diff *= ff #to_zero
    chi2 = np.average((diff/exp[:,1])**2)
    log += "CHI2: %.5f \n" % chi2

    
    m_min = np.min(calc,axis=0)
    m_max = np.max(calc,axis=0)


    diff_min = ff*(m_min-exp[:,0])/exp[:,1]
    ii_min = np.where(diff_min>1.)[0]
    if(len(ii_min)>0):
        log += "##### WARNING ########## \n"
        log += "# The minimum value of the following data is higher than expt range: \n"
        log += "# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","min_calc")
        for j in ii_min:
            log += "%15s %8.4f %8.4f %8.4f\n" % (label[j],exp[j,0],exp[j,1],m_min[j])

    diff_max = ff*(exp[:,0]-m_max)/exp[:,1]
    ii_max = np.where(diff_max>1.)[0]
    if(len(ii_max)>0):
        log += "##### WARNING ########## \n"
        log += "# The maximum value of the following data is lower than expt range: \n"
        log += "# %14s %8s %8s %8s \n" % ("label","exp_avg","exp_sigma","max_calc")
        for j in ii_max:
            log += "%15s %8.4f %8.4f %8.4f\n" % (label[j],exp[j,0],exp[j,1],m_max[j])
                
                
        
    return log

def standardize(exp,calc,sample_weights,normalize="zscore"):
    
    if(normalize=="zscore"):
        calc_avg = np.sum(calc*sample_weights[:,np.newaxis],axis=0)
        std = np.sqrt(np.average((calc_avg-calc)**2, weights=sample_weights,axis=0))
        #exp[:,0] = (exp[:,0]-calc_avg)/std # does not modify in-place
        #calc = (calc-calc_avg)/std
        
        exp[:,0] -= calc_avg
        exp[:,0] /= std
        calc -= calc_avg
        calc /= std
        exp[:,1] /= std
        log = "# Z-score normalization \n"
        
    elif(normalize=="minmax"):
        mmin = calc.min(axis=0)
        mmax = calc.max(axis=0)
        delta = mmax-mmin
        #exp[:,0] = (exp[:,0]-mmin)/delta
        #calc = (calc-mmin)/delta
        exp[:,0] -= mmin
        exp[:,0] /= std
        calc -= mmin
        calc /= delta
        exp[:,1] /= delta
        log = "# MinMax normalization \n"

    return log
