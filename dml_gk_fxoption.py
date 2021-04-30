#%% --------------------------------------------------
# import packages
# ----------------------------------------------------
import numpy as np
import bisect as bisect
from scipy import interpolate
from scipy.stats import norm
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
    def __init__(self,rd=0.04,rf=0.02,sigma=0.10,S0=1.0):
        self.rd = rd
        self.rf = rf
        self.sigma = sigma
        self.S0 = S0
# Contract class
class Contract:
    def __init__(self,T=1.0,K=1.05):
        self.T = T
        self.K = K
# Finite Difference class
class FD:
    def __init__(self,SMax=1.35,SMin=0.20,T=1.0,sigma=0.1):
        self.SMax = SMax
        self.SMin = SMin
        self.deltat = T/10000.0
        self.deltas = sigma*np.sqrt(self.deltat) 

#%% --------------------------------------------------
# Garman-Kohlhagan analytical
# ----------------------------------------------------
def bsFXPrice(dynamics,contract):
    S0, vol, rd, rf= dynamics.S0, dynamics.sigma, dynamics.rd, dynamics.rf
    K, T = contract.K, contract.T 
    F=S0*np.exp((rd-rf)*T)
    d1 = (np.log(F/K) + (0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = (np.log(F/K) - (0.5 * vol**2) * T) / (vol * np.sqrt(T))
    return S0*np.exp(-rf*T) * norm.cdf(d1) - K*np.exp(-rd*T) * norm.cdf(d2)

def bsFXDelta(dynamics,contract):
    S0, vol, rd, rf= dynamics.S0, dynamics.sigma, dynamics.rd, dynamics.rf
    K, T = contract.K, contract.T
    F=S0*np.exp((rd-rf)*T)
    d1 = (np.log(F/K) + 0.5 * vol * vol * T) / (vol * np.sqrt(T))
    return np.exp((-rf)*T)*norm.cdf(d1)

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
def call_FX_GK_CrankNicolson(contract,dynamics,FD):
    """
    returns array of all initial short rates,
    and the corresponding array of zero-coupon
    T-maturity bond prices
    """ 
    volcoeff, S0,rf,rd=dynamics.sigma, dynamics.S0, dynamics.rf, dynamics.rd    
    T,K=contract.T,contract.K
    SMax=FD.SMax
    SMin=FD.SMin
    deltaS=FD.deltas
    deltat=FD.deltat
    # mesh
    N=round(T/deltat)    
    if abs(N-T/deltat)>1e-12:
        raise ValueError('Bad time step')
    numS=round((SMax-SMin)/deltaS)+1
    if abs(numS-(SMax-SMin)/deltaS-1)>1e-12:
        raise ValueError('Bad time step')
    S=np.linspace(SMax,SMin,numS)    #The FIRST indices in this array are for HIGH levels of S
    S_highboundary=SMax+deltaS
    callprice=np.maximum(S-K,0)
    ratio=deltat/deltaS
    ratio2=deltat/deltaS**2
    # coefficient    
    f = 0.5*(volcoeff**2)*S**2  
    g = (rd-rf)*S
    h = -rd 
    # conversion
    F = 0.5*ratio2*f+0.25*ratio*g
    G = ratio2*f-0.5*deltat*h
    H = 0.5*ratio2*f-0.25*ratio*g
    # PDE    
    RHSmatrix = diags([H[:-1], 1-G, F[1:]], [1,0,-1], shape=(numS,numS), format="csr")
    LHSmatrix = diags([-H[:-1], 1+G, -F[1:]], [1,0,-1], shape=(numS,numS), format="csr")
    # diags creates SPARSE matrices
    for t in np.arange(N-1,-1,-1)*deltat:
        rhs = RHSmatrix * callprice
        #boundary condition vectors.They are nonzero only in the first component:
        rhs[0]=rhs[0]+2*H[0]*(S_highboundary-K) #Boundary
        callprice = spsolve(LHSmatrix, rhs)    
    return(S, callprice)

#%% --------------------------------------------------
# Garman-Kohlhagan Monte Carlo simulation
# ----------------------------------------------------
def call_FX_GK_MC(contract,dynamics,MC):

    np.random.seed(MC.seed)  #seed the random number generator
    # You complete the coding of this function        
    S0, sigma, rd, rf= dynamics.S0, dynamics.sigma, dynamics.rd, dynamics.rf
    K, T = contract.K, contract.T
    N, M, epsilon = MC.N, MC.M, MC.epsilon
    # setting
    deltat = T/N
    C = []
    SE = []
    z = np.random.randn(N,M)
    for S in [S0, S0+epsilon]:
        x = S
        for t in range(N):
            zm = z[t]
            x = x + (rd-rf)*deltat + sigma*zm*np.sqrt(deltat)
        s = x
        # discounted payoff
        payoff = np.maximum(s-K,0)*np.exp(-rd*T)
        C.append(np.mean(payoff))
        SE.append(np.std(payoff, ddof=1)/np.sqrt(M))    
    call_price = C[0]
    standard_error = SE[0]
    call_delta = (C[1]-C[0])/epsilon
    return(call_price, standard_error, call_delta)

#%% --------------------------------------------------
# generator
# ----------------------------------------------------
# main class
class BlackScholesFX:
    
    def __init__(self, 
                 dynamics,
                 contract,
                 T1=0.5,
                 volMult=1.0):

        # dynamics        
        self.dynamics = Dynamics()
        self.dynamics.S0 = dynamics.S0
        self.dynamics.rd = dynamics.rd
        self.dynamics.rf = dynamics.rf
        self.dynamics.sigma = dynamics.sigma
        # contract
        self.contract_T2=Contract()
        self.contract_T2.T = contract.T + T1 # take difference
        self.contract_T2.K=contract.K
        self.contract_T1= Contract()
        self.contract_T1.T = T1
        self.contract_T1.K=contract.K
        self.contract_diff= Contract()
        self.contract_diff.T = contract.T
        self.contract_diff.K=contract.K
        # simulation
        self.spot = self.dynamics.S0
        self.vol =  self.dynamics.sigma
        self.rf = self.dynamics.rf
        self.rd = self.dynamics.rd
        self.T1 = T1
        self.T2 = self.contract_T2.T
        self.K = self.contract_T2.K
        self.volMult = volMult
                        
    # training set: returns S1 (mx1), C2 (mx1) and dC2/dS1 (mx1)
    def trainingSet(self, m, anti=True, seed=None):
    
        np.random.seed(seed)
        # 2 sets of normal returns
        returns = np.random.normal(size=[m, 2])
        # SDE
        vol0 = self.vol * self.volMult
        R1 = np.exp((self.rd-self.rf-0.5*vol0*vol0)*self.T1 + vol0*np.sqrt(self.T1)*returns[:,0])
        R2 = np.exp((self.rd-self.rf-0.5*self.vol*self.vol)*(self.T2-self.T1) \
                    + self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
        S1 = self.spot * R1
        S2 = S1 * R2 
        # payoff
        pay = np.maximum(0, np.exp(-self.rd*self.T2)*(S2 - self.K))
        # two antithetic paths
        if anti:
            # anti paths
            R2a = np.exp((self.rd-self.rf-0.5*self.vol*self.vol)*(self.T2-self.T1) \
                    - self.vol*np.sqrt(self.T2-self.T1)*returns[:,1])
            S2a = S1 * R2a             
            paya = np.maximum(0, np.exp(-self.rd*self.T2)*(S2a - self.K))
            # init            
            X = S1
            Y = 0.5 * (pay + paya)
            # differentials
            Z1 =  np.where(np.exp(-self.rd*self.T2)*S2 > np.exp(-self.rd*self.T2)*self.K, R2, 0.0).reshape((-1,1)) 
            Z2 =  np.where(np.exp(-self.rd*self.T2)*S2a > np.exp(-self.rd*self.T2)*self.K, R2a, 0.0).reshape((-1,1)) 
            Z = 0.5 * (Z1 + Z2)
        # standard
        else:
            # init
            X = S1
            Y = pay
            # differentials
            Z =  np.where(np.exp(-self.rd*self.T2)*S2 > np.exp(-self.rd*self.T2)*self.K, R2, 0.0).reshape((-1,1))         
        return X.reshape([-1,1]), Y.reshape([-1,1]), Z.reshape([-1,1])
    
    # test set: returns a grid of uniform spots with corresponding ground true prices, deltas
    def testSet(self, lower=0.8, upper=1.15, num=1000, seed=None):
        
        spots = np.linspace(lower, upper, num).reshape((-1, 1))
        # compute prices, deltas 
        prices = np.zeros(num)
        deltas = np.zeros(num)
        for i in range(num):
            self.dynamics.S0 = spots[i]
            dammy_cont = Contract()
            dammy_cont.T = self.contract_T2.T - self.contract_T1.T
            dammy_cont.K = self.contract_T2.K
            prices[i] = bsFXPrice(self.dynamics,dammy_cont)
            deltas[i] = bsFXDelta(self.dynamics,dammy_cont)
            #prices[i] = bsFXPrice(self.dynamics,self.contract_T2)
            #deltas[i] = bsFXDelta(self.dynamics,self.contract_T2)        
        prices = prices.reshape((-1, 1))
        deltas = deltas.reshape((-1, 1))
        return spots, spots, prices, deltas  

#%% --------------------------------------------------
# Training neural networks
# ----------------------------------------------------
# simulation set sizes to perform
sizes = [1024, 8192]
# show delta?
showDeltas = True
# seed
# simulSeed = 1234
simulSeed = np.random.randint(0, 10000) 
print("using seed %d" % simulSeed)
weightSeed = None
# number of test scenarios
nTest = 1000    
# training
fxrate = Dynamics()
contract = Contract()
generator = BlackScholesFX(fxrate,contract)
xAxis, yTest, dydxTest, values, deltas,exec_time = \
    ldml.test(generator, sizes, nTest, simulSeed, None, weightSeed)
# show predicitions
ldml.graph("Call-GK", values, xAxis, "", "values %", yTest, sizes, True)
# show deltas
if showDeltas:
    ldml.graph("Call-GK", deltas, xAxis, "", "deltas%", dydxTest, sizes, True)

#%% --------------------------------------------------
# PDE prices with Clank-Nicolson
# ----------------------------------------------------
# calculate CN prices
fd_cn = FD()
start = time.time()
(fxspot, callprice) = call_FX_GK_CrankNicolson(contract,fxrate,fd_cn)
end = time.time()
cn_fd_time = round(end - start,2)
print("PDE Clank-Nicolson: ",cn_fd_time,'[sec]')

#%% --------------------------------------------------
# prices to be compared
# ----------------------------------------------------
# simulation setting
N = 1000
s = 8192
# initialization
spot_charts = np.linspace(0.95,1.07, N)
call_fx_cn = np.zeros(N)
call_fx_dml= np.zeros(N)
call_fx_snn= np.zeros(N)
call_fx_analytical = np.zeros(N)
# get nn results
callprice_dml = [c[0] for c in values[("differential", s)].tolist()]
callprice_snn = [c[0] for c in values[("standard", s)].tolist()]

for i, S_ in enumerate(tqdm(spot_charts)):
    fxrate.S0 = S_
    call_fx_analytical[i] = bsFXPrice(fxrate,contract)
    call_fx_cn[i]  = interpolate_rate(fxspot,callprice,S_)
    call_fx_dml[i] = interpolate_rate(xAxis,callprice_dml,S_)
    call_fx_snn[i] = interpolate_rate(xAxis,callprice_snn,S_)
    
#%% --------------------------------------------------
# visualization
# ----------------------------------------------------
# figure
fig = plt.figure(figsize = (8,4))
# settings
X = spot_charts
Zcn  = call_fx_cn
#Zsml = call_fx_snn
Zdml = call_fx_dml
Zanl = call_fx_analytical
# CN result
ax1 = fig.add_subplot(121)
plt.subplots_adjust(hspace = 0.3,wspace = 0.3)
for c,m, Z in [('r','o', Zcn), ('b','^', Zanl)]:
    ax1.scatter(X, Z, c = c,marker = m,s=2)
ax1.legend(["CN","A"],loc="upper left")
ax1.set_xlabel('Spot Price')
ax1.set_ylabel('Price FX Call')
ax1.title.set_text('R-F Price:Crank-Nicolson vs Exact Sol.')
# DML result
ax2 = fig.add_subplot(122)
plt.subplots_adjust(hspace = 0.3,wspace = 0.3)
for c, m, Z in [('r','o', Zdml), ('b','^', Zanl)]:
    ax2.scatter(X, Z, c = c, marker = m,s=2)
ax2.legend(["DML","A"],loc="upper left")
ax2.set_xlabel('Spot Price')
ax2.set_ylabel('Price FX Call')
ax2.title.set_text('R-F Price:DML vs Exact Sol.')
# finalize
plt.show()


# %%
