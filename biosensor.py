import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import differential_evolution
from functools import lru_cache
from IPython.display import display
import seaborn as sns
import pymc3 as pm
from tqdm.autonotebook import tqdm
import logging


#Turn of PyMC3 reporting
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
        


class biosensor(object):
    def __init__(self,X,Y,beta=1.2):
        self.model_fit = False
        self.X = X
        self.Y = Y
        self.Ymax = beta*max(Y)
        self.Xmax = beta*max(X)
        self.fit()
        
    def fit(self,bounds=[(0.0,20000),(0.1,2),(0,100000),(0,100)]):
        cost = lambda params: -characterization_likelihood(self.X,self.Y,*params)
        params = differential_evolution(cost,bounds,tol=0.001)
        
        #Assign Variables to Fit
        self.sigma, self.hill_coef, self.Vmax, self.Kd = params.x
        
    
    def fit_with_error(self,verbose=True,samples=2000,seed=None):
        with pm.Model() as model:

            sigma = pm.Uniform('sigma', lower=0, upper=20000)
            hill_coef =  pm.Uniform('hill_coef', lower=0.1, upper=2)
            Vmax = pm.Uniform('Vmax', lower=0, upper=100000)
            Kd = pm.Uniform('Kd', lower=0, upper=100)
            
            #likelihood
            likelihood = pm.DensityDist('likelihood', 
                                        lambda sigma,hill_coef,Vmax,Kd: fit_likelihood(self.X,self.Y,sigma,hill_coef,
                                                                        Vmax,Kd),  observed={'sigma': sigma,'hill_coef':hill_coef,'Vmax':Vmax,'Kd':Kd})
            #Calculate Parameter Ensemble
            trace = pm.sample(samples, tune=1500,progressbar=verbose,random_seed=seed)
        
        for var,name in zip([self.sigma, self.hill_coef, self.Vmax, self.Kd],['sigma','hill_coef','Vmax','Kd']):
            
            mu = np.mean(trace[name])
            print(name,'mu:',mu,'sigma:',np.std(trace[name]))
            var = mu        
        
        
    def report_params(self):
        cols = ['Sigma','Hill Coefficient','Max Output','Kd']
        data = [[self.sigma,self.hill_coef,self.Vmax,self.Kd]]
        display(pd.DataFrame(columns=cols,data=data))
        
        
    def predict(self,Ym,pdf=False,verbose=True,samples=2000,seed=None):
        '''From a collection of Y Measurements predict compatable Xs'''
        
        with pm.Model() as model:
            x = pm.Uniform('x', lower=0, upper=100)

            #likelihood
            likelihood = pm.DensityDist('likelihood', 
                                        lambda x: prediction_likelihood(Ym,x,self.sigma,self.hill_coef,
                                                                        self.Vmax,self.Kd), observed={'x': x})

            #Calculate Parameter Ensemble
            trace = pm.sample(samples, tune=1500,progressbar=verbose,random_seed=seed)
        
        if pdf:
            return trace['x']
        
        else:
            mu = np.mean(trace['x'])
            sigma = np.std(trace['x'])
            return mu,sigma
        
        return trace
        
    
    def plot_fit(self):
        plt.figure()
        ax = plt.gca()
        
        #Overlay The Data
        plt.scatter(self.X,self.Y,color='g')

        #Plot Line Fit
        Xp = np.linspace(0,self.Xmax,1000)
        Yp = hill_fcn(Xp,self.hill_coef,self.Vmax,self.Kd)
        plt.plot(Xp,Yp)

        #Plot Variance as Fill
        ax.fill_between(Xp, Yp - 2*self.sigma, Yp + 2*self.sigma, facecolor='blue', alpha=0.1)

        #Label Plot
        plt.xlabel('Inducer Concentration $(mM)$')
        
        #Cleanup Formatting
        sns.despine(top=True,right=True)
        plt.ylim(bottom=0)
        
        
    def plot_resolution(self,Y,n=1,sigma=0):
        X = np.zeros(len(Y))
        Z = np.zeros(len(Y))
        for i,y in enumerate(tqdm(Y)):
            y = np.random.normal(y,sigma,n)
            X[i],Z[i] = self.predict(y,verbose=False)
        plt.plot(X,Z)
        
        return X,Z
        
        

#Hill Func
def hill_fcn(x, n, a, k):
        return a*x**n / (k**n + x**n)

hill_params = lambda bioS: [bioS.hill_coef,bioS.Vmax,bioS.Kd]

def hill_inverse(ics,n,a,k):
    '''return inducer concentration'''
    rfp = []
    for ic in ics:
        #define Cost Function
        cost = lambda x: (x - hill_fcn(ic,n,a,k))**2
        
        #Find Optimal Value
        res = differential_evolution(cost,[(0,10000)])
        x_hat = res.x[0]
        #Append to inducer concentration list
        rfp.append(x_hat)
        
    return rfp


def distribution_range_plot(biosensor,points,samples=20000,replicates=4,sigma=0):
    rfp = hill_inverse(points,*hill_params(biosensor))
    for mu,c in zip(rfp,sns.color_palette('nipy_spectral',len(points))):   
        
        #Generate Replicates with noise defined by the model
        Ym = np.random.normal(mu,sigma,3)

        #Create distribution of Compatable Inducer Concentraitons
        samples = biosensor.predict(Ym,pdf=True,samples=20000)

        #Plot Distribution
        sns.distplot(samples,color=c,norm_hist=True)
        
        plt.xlabel('Inducer Concentration [mM]')
        plt.ylabel('Density')
        
    return samples
        
        
#Likelihood used for model fitting
def characterization_likelihood(X,Y,sigma,n,a,k):
    Mu = hill_fcn(X,n,a,k)
    
    log_likelihood = 0
    for y,mu in zip(Y,Mu):
        #Calculate Log Probability of the Gaussian for each sample
        logp = norm.logpdf(y,mu,sigma)
        if np.isnan(logp):
            logp = np.finfo(float).eps    
        log_likelihood += logp
        
    return log_likelihood


#Likelihood used for x prediction
def prediction_likelihood(Ym,X,sigma,n,a,k):
    mu = hill_fcn(X,n,a,k)
    lpdf = lambda x: np.log(1/(np.sqrt(2*math.pi)*sigma)) - (x-mu)**2/(2*sigma**2)
    ll = sum([lpdf(y) for y in Ym])
    return ll


#Likelihood used fit
def fit_likelihood(Ym,X,sigma,n,a,k):
    ll = 0
    for x,y in zip(Ym,X):
        mu = hill_fcn(x,n,a,k)
        lpdf = lambda x: np.log(1/(np.sqrt(2*math.pi)*sigma)) - (x-mu)**2/(2*sigma**2)
        ll += lpdf(y)
    return ll


#Function to Calculate the Range of detection from a biosensor
opt_fun = differential_evolution
def detection_range(n,biosensor,seed=42,alpha=0.95,tol=1,verbose=False):
    
    np.random.seed(seed)
    Ym_base = np.random.normal(0,1,n)
    
    #Calculate Credible Interval for 0
    Ym = Ym_base*biosensor.sigma
    samples = biosensor.predict(Ym,pdf=True,samples=10000,seed=seed,verbose=verbose)
    interval = prediction_interval(samples,alpha,seed=seed)
    c0_max = interval[1]
    
    #Calculate Credible Interval for Max flouresence
    ymax = hill_fcn(biosensor.Xmax,*hill_params(biosensor))
    Ym = Ym_base*biosensor.sigma + ymax
    samples = biosensor.predict(Ym,pdf=True,samples=10000,seed=seed,verbose=verbose)
    interval = prediction_interval(samples,alpha,seed=seed)
    cmax_min = interval[0] 
    
    if verbose: print('FL Range:',0,ymax)
    if verbose: print('C interval:',interval)
    if verbose: print('Xmax:',biosensor.Xmax)
                    
    #Cost Function
    @lru_cache()
    def cost(f,objective):
        Ym = Ym_base*biosensor.sigma + f
        
        #Inducer Concentration Samples
        samples = biosensor.predict(Ym,pdf=True,samples=3000,seed=seed,verbose=False)
        c = np.mean(samples)
        
        #Calculate Prediction Interval
        interval = prediction_interval(samples,alpha,seed=seed)
        
        if verbose: print('fl:',f[0],'c:',c)
        if objective == 'min':
            cost = (interval[0] - c0_max)**2
            if verbose: print('min',interval[0],c0_max,cost)

        elif objective == 'max':
            cost = (interval[1] - cmax_min)**2
            if verbose: print('max',interval[1],cmax_min,cost)

        if verbose: print('')
        
        return cost
         
    #minimize cost function for the high end
    opt = opt_fun(lambda x: cost(tuple(x),'max'),[[0,biosensor.Ymax]],tol=tol)
    
    #Calculate max C
    Ym = Ym_base*biosensor.sigma + opt.x
    samples = biosensor.predict(Ym,pdf=True,samples=10000,seed=seed,verbose=False)
    max_c = np.mean(samples)
    
    #minimize cost function for the low end
    opt = opt_fun(lambda x: cost(tuple(x),'min'),[[0,biosensor.Ymax]],tol=tol)
    if verbose: print(opt)
    
    #Calculate min C
    Ym = Ym_base*biosensor.sigma + opt.x
    samples = biosensor.predict(Ym,pdf=True,samples=10000,seed=seed,verbose=False)
    min_c = np.mean(samples)

    return min_c,max_c

    #def max_cost(f):
        

def prediction_interval(samples,alpha=0.95,seed=None,bootstrap=100000):
    if alpha == 0.95:
        n=39
    else:
        raise NotImplementedError
    np.random.seed(seed)
    subsample = np.random.choice(samples,[n,bootstrap])
    return [np.mean(subsample.min(axis=0)),np.mean(subsample.max(axis=0))]
