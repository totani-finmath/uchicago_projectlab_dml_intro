#%% --------------------------------------------------
# import packages
# ----------------------------------------------------
import numpy as np
import bisect as bisect
from scipy import interpolate
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
# local library
import importlib
import lib_dml as ldml
importlib.reload(ldml)

#%% --------------------------------------------------
# Class definitions
# ----------------------------------------------------
# Dynamics class
class Dynamics:
    def __init__(self,r0=0.04,theta=2.0,mu=0.05,sigma=0.08):
        self.r0 = r0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
# Contract class
class Contract:
    def __init__(self,T=1.0):
        self.T = T
# Finite Difference class
class FD:
    def __init__(self,r_max=0.35,r_min=-0.20,T=1.0,sigma=0.1):
        self.r_max = r_max
        self.r_min = r_min
        self.deltat = T/10000.0
        self.deltar = sigma*np.sqrt(self.deltat) 

#%% --------------------------------------------------
# Vasicek analytical
# ----------------------------------------------------

def stoc_bond_price(dynamics, contract):
    """
    Function that computes the analytical zero-coupon bond price given vasicek interest rate model 
    """    
    theta, mu, sigma, r0, t = dynamics.theta, dynamics.mu, dynamics.sigma, dynamics.r0, contract.T   
    A = (1-np.exp(-theta*t))/theta
    D = (mu - (sigma**2)/(2*(theta)**2))*(A-t) - ((sigma**2)*A**2)/(4*theta)
    return np.exp(-A*r0+D)

def diff_bond_price(dynamics, contract):
    """
    Function that computes the analytical zero-coupon bond price sensitivity wrt r_0 given vasicek interest rate model 
    """
    theta, mu, sigma, r0, T = dynamics.theta, dynamics.mu, dynamics.sigma, dynamics.r0, contract.T    
    A = (1-np.exp(-theta*T))/theta
    D = (mu - (sigma**2)/(2*(theta)**2))*(A-T) - ((sigma**2)*A**2)/(4*theta)
    return -A*np.exp(D-A*r0)

def forward_exchange_rate(X0, dynamics_f, dynamics_d, contract):
    """
    Function that computes the analytical price of a forward exchange rate given the parameters of 2 short rate following ornstein-uhlenbeck SDE 
    """    
    bond_f = stoc_bond_price(dynamics_f, contract)
    bond_d = stoc_bond_price(dynamics_d, contract)
    return X0*bond_f/bond_d

#%% --------------------------------------------------
# FD helpers
# ----------------------------------------------------
def two_closest(lst, K): 
    """
    Function that find the closest upper and lower element in an array
    """    
    pos_arr = lst[(lst-K) > 0]
    neg_arr = lst[(K-lst) > 0]
    min_val = neg_arr[(np.abs(neg_arr - K)).argmin()]
    max_val = pos_arr[(np.abs(pos_arr - K)).argmin()]
    return min_val, max_val

def interpolate_rate(r_arr, bondprice, r0):
    """
    Find the interpolate value of a bond for a given rate of interest based on grid of values 
    """
    r_low, r_high = two_closest(r_arr, r0)
    bond_low = bondprice[np.where(r_arr == r_low)[0][0]]
    bond_high = bondprice[np.where(r_arr == r_high)[0][0]]
    x = [r_low, r_high]
    y = [bond_low, bond_high]
    f = interpolate.interp1d(x,y)
    return f(r0)

#%% --------------------------------------------------
# Clank Nicolson PDE
# ----------------------------------------------------
def pricer_bond_vasicek_CrankNicolson(contract,dynamics,FD):
    """
    returns array of all initial short rates,
    and the corresponding array of zero-coupon
    T-maturity bond prices
    """ 
    volcoeff, mu,  theta, r0 = dynamics.sigma, dynamics.mu, dynamics.theta, dynamics.r0
    T = contract.T
    # settings
    r_max = FD.r_max
    r_min = FD.r_min
    deltar = FD.deltar
    deltat = FD.deltat
    N = round(T/deltat)
    numr = int(round((r_max-r_min)/deltar)+1)
    # The FIRST indices in this array are for HIGH levels of r
    r = np.linspace(r_max,r_min,numr)
    # pricer
    bondprice = np.ones(np.size(r)) #Payoff
    ratio  = deltat/deltar
    ratio2 = deltat/deltar**2
    # parameters for PDE    
    f = 0.5*(volcoeff**2)*np.ones(np.size(r))  
    g = theta*(mu-r)
    h = -r 
    # conversion
    F = 0.5*ratio2*f+0.25*ratio*g
    G = ratio2*f-0.5*deltat*h
    H = 0.5*ratio2*f-0.25*ratio*g
    # formulation
    RHSmatrix = diags([H[:-1], 1-G, F[1:]], [1,0,-1], shape=(numr,numr), format="csr")
    LHSmatrix = diags([-H[:-1], 1+G, -F[1:]], [1,0,-1], shape=(numr,numr), format="csr")
    # looing solver
    for t in np.arange(N-1,-1,-1)*deltat:
        rhs = RHSmatrix * bondprice 
        # boundary condition vectors.
        rhs[-1]=rhs[-1]+2*H[-1]*(2*bondprice[-1]-bondprice[-2]) # boundary 
        rhs[0]=rhs[0]+2*F[0]*(2*bondprice[1]-bondprice[2]) # boundary
        bondprice = spsolve(LHSmatrix, rhs)
    return(r, bondprice)

#%% --------------------------------------------------
# Vasicek Monte Carlo simulation
# ----------------------------------------------------
def vasicek(dynamics,contract, N=1000, seed=767, simulations=10000):    
    """ 
    Produces the stochastic paths of a vasicek SDE
    """
    #np.random.seed(seed)
    r0,theta,mu,sigma,T = dynamics.r0,dynamics.theta,dynamics.mu,dynamics.sigma,contract.T
    dt = T/float(N)
    mean_vasicek = lambda t:np.exp(-theta*t)*(r0+mu*(np.exp(theta*t)-1))
    var_vasicek  = lambda t:((1-np.exp(-2*theta*t))/2*theta)*sigma**2
    normal_matrix = np.random.standard_normal(size=(simulations,N+1))
    rates_paths = np.add(np.array([mean_vasicek(dt*t_) for t_ in range(N+1)]),np.sqrt([var_vasicek(dt*t_) for t_ in range(N+1)])*normal_matrix)        
    return [x*dt for x in range(N+1)], rates_paths

def pricer_bond_vasicek_MonteCarlo(dynamics,contract,N=1000, seed=767, simulations=10000):
    """
    Returns the price of a Bond Contract using Monte-Carlo simulation
    """ 
    t,rates = vasicek(dynamics,contract,N=N, seed=seed, simulations=simulations)
    num_paths = len(rates)
    dt = t[1]
    exp_integrals = [np.exp(-(np.array(rates[i])*dt).sum()) for i in range(num_paths)]
    return np.mean(exp_integrals)

def price_diff_bond_vasicek_MonteCarlo(dynamics,contract,N=100, seed=767, simulations=1):
    """
    Returns the price of a Bond Contract using Monte-Carlo simulation and it's sensitivity w.r.t r_0
    """ 
    t,rates = vasicek(dynamics,contract,N=N, seed=seed, simulations=simulations)
    num_paths = len(rates)
    dt = t[1]
    exp_integrals = np.array([np.exp(-(np.array(rates[i])*dt).sum()) for i in range(num_paths)])
    return np.mean(exp_integrals), -exp_integrals*((1-np.exp(-dynamics.theta*contract.T))/dynamics.theta)

#%% --------------------------------------------------
# generator
# ----------------------------------------------------
# %%
# main class
class VasicekBond:
    
    def __init__(self, 
                 dynamics,
                 contract,
                 T1=0.5):
        
        self.dynamics = Dynamics()
        self.dynamics.r0 = dynamics.r0
        self.dynamics.theta = dynamics.theta
        self.dynamics.mu = dynamics.mu
        self.dynamics.sigma = dynamics.sigma
        self.contract_T1 = Contract()
        self.contract_T1.T = T1
        self.contract_T2= Contract()
        self.contract_T2.T = contract.T + T1
        self.contract_diff = Contract()
        self.contract_diff.T = contract.T
    
    # training set: returns r_1 (mx1), B_T (mx1) and dC2/dS1 (mx1)
    def trainingSet(self, m, seed=None):
    
        np.random.seed(seed)    
        # rates at (T-t) - inputs FF
        t, paths = vasicek(self.dynamics,self.contract_T1,N=1,seed=seed,simulations=m)
        X = np.squeeze(np.delete(paths, 0, 1))
        # payoff - labels
        Y = np.zeros(m)
        #differentials - inputs BP 
        Z = np.zeros(m)
        for i in range(m):
            self.dynamics.r0 = X[i]
            Y[i], Z[i] = price_diff_bond_vasicek_MonteCarlo(self.dynamics,self.contract_diff,N=100, seed=seed, simulations=1)
        return X.reshape([-1,1]), Y.reshape([-1,1]), Z.reshape([-1,1])
    
    # test set: returns a grid of uniform spots with corresponding ground true prices
    def testSet(self, lower=-0.35, upper=0.5, num=100, seed=None):
        
        spots = np.linspace(lower, upper, num).reshape((-1, 1))
        # compute prices, diffs
        prices = np.zeros(num)
        diffs = np.zeros(num)
        for i in range(num):
            self.dynamics.r0 = spots[i]
            prices[i] = stoc_bond_price(self.dynamics,self.contract_diff)
            diffs[i]  = diff_bond_price(self.dynamics,self.contract_diff)
        return spots.reshape((-1, 1)), spots.reshape((-1, 1)), prices.reshape((-1, 1)), diffs.reshape((-1, 1))  


#%% --------------------------------------------------
# Training neural networks
# ----------------------------------------------------
# simulation set sizes to perform
# sizes = [1024, 8192, 65536]
sizes = [1024, 4096]
# show delta?
showDeltas = True
# seed
simulSeed = np.random.randint(0, 10000) 
# simulSeed = 1234
print("using seed %d" % simulSeed)
weightSeed = None
# number of test scenarios
nTest = 1000
# domestic bond
dom_bond = Dynamics(r0=0.04,theta=3.0,mu=0.05,sigma=0.08)
fgn_bond = Dynamics(r0=0.10,theta=4.0,mu=0.08,sigma=0.12)
contract = Contract()
# training domestic
generator_dom = VasicekBond(dom_bond,contract)
xAxis_dom, yTest_dom, dydxTest_dom, values_dom, deltas_dom, time_dom = \
    ldml.test(generator_dom, sizes, nTest, simulSeed, None, weightSeed)
# training foreign
generator_fgn = VasicekBond(fgn_bond,contract)
xAxis_fgn, yTest_fgn, dydxTest_fgn, values_fgn, deltas_fgn, time_fgn = \
    ldml.test(generator_fgn, sizes, nTest, simulSeed, None, weightSeed)
# show predicitions
ldml.graph("Vasicek", values_dom, xAxis_dom, "", "values %", yTest_dom, sizes, True)
# show deltas
if showDeltas:
    ldml.graph("Vasicek", deltas_dom, xAxis_dom, "", "deltas%", dydxTest_dom, sizes, True)

#%% --------------------------------------------------
# generaet true price set
# ----------------------------------------------------
test_rs_dom, test_rs_damm, test_prices_dom, test_diffs_dom = generator_dom.testSet()
test_rs_fgn, test_rs_damm, test_prices_fgn, test_diffs_fgn = generator_fgn.testSet()

#%% --------------------------------------------------
# bond price comparison
# ----------------------------------------------------
# init rates for plots
sim_size = 100
X0 = 1.20
sim_rs_dom = np.linspace(0.00,0.25,num=sim_size)
sim_rs_fgn = np.linspace(0.05,0.05,num=sim_size)
# Clank Nicolson
fd_cn = FD()
grids_cn_dom, prices_cn_dom = pricer_bond_vasicek_CrankNicolson(contract,dom_bond,fd_cn)
grids_cn_fgn, prices_cn_fgn = pricer_bond_vasicek_CrankNicolson(contract,fgn_bond,fd_cn)
sim_price_cn_dom = [interpolate_rate(grids_cn_dom,prices_cn_dom,r) for r in sim_rs_dom]
sim_price_cn_fgn = [interpolate_rate(grids_cn_fgn,prices_cn_fgn,r) for r in sim_rs_fgn]
# SML
trg_set = ('standard',sizes[1])
sim_price_sml_dom = [interpolate_rate(xAxis_dom.flatten(),values_dom[trg_set].flatten(),r) for r in sim_rs_dom]
sim_price_sml_fgn = [interpolate_rate(xAxis_fgn.flatten(),values_fgn[trg_set].flatten(),r) for r in sim_rs_fgn]
# DML
trg_set = ('differential',sizes[1])
sim_price_dml_dom = [interpolate_rate(xAxis_dom.flatten(),values_dom[trg_set].flatten(),r) for r in sim_rs_dom]
sim_price_dml_fgn = [interpolate_rate(xAxis_fgn.flatten(),values_fgn[trg_set].flatten(),r) for r in sim_rs_fgn]
# true price
sim_price_tru_dom = [interpolate_rate(test_rs_dom.flatten(),test_prices_dom.flatten(),r) for r in sim_rs_dom]
sim_price_tru_fgn = [interpolate_rate(test_rs_fgn.flatten(),test_prices_fgn.flatten(),r) for r in sim_rs_fgn]
# figure domestic
plt.figure()
plt.plot(sim_rs_dom,sim_price_cn_dom,label='CN')
plt.plot(sim_rs_dom,sim_price_sml_dom,label='SML')
plt.plot(sim_rs_dom,sim_price_dml_dom,label='DML')
plt.plot(sim_rs_dom,sim_price_tru_dom,label='Analytical',ls=':')
plt.title('Domestic ZCB prices')
plt.xlabel('domestic interest spot rate')
plt.xlabel('bond price')
plt.legend()
plt.show()

#%% --------------------------------------------------
# forward price comparison
# ----------------------------------------------------
fwd_cn   = np.zeros(sim_size)
fwd_sml  = np.zeros(sim_size)
fwd_dml  = np.zeros(sim_size)
fwd_anal = np.zeros(sim_size)
for i in range(sim_size):
    fwd_cn[i]   = X0 * sim_price_cn_fgn[i] / sim_price_cn_dom[i]
    fwd_sml[i]  = X0 * sim_price_sml_fgn[i] / sim_price_sml_dom[i]
    fwd_dml[i]  = X0 * sim_price_dml_fgn[i] / sim_price_dml_dom[i]
    fgn_bond.r0 = sim_rs_fgn[i]
    dom_bond.r0 = sim_rs_dom[i]
    fwd_anal[i] = forward_exchange_rate(X0,fgn_bond,dom_bond,contract)
# figure domestic
plt.figure()
plt.plot(sim_rs_dom,fwd_cn,label='CN')
plt.plot(sim_rs_dom,fwd_dml,label='SML')
plt.plot(sim_rs_dom,fwd_dml,label='DML')
plt.plot(sim_rs_dom,fwd_anal,label='Analytical',ls=':',lw=2.5)
plt.title('FX forward prices')
plt.xlabel('domestic interest spot rate')
plt.xlabel('forward price')
plt.legend()
plt.show()

# %%
