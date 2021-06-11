# %% --------------------------------------------------
# Import packages
# -----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
import importlib
from copy import deepcopy as copy
from tqdm import tqdm

# %% --------------------------------------------------
# Helper function
# -----------------------------------------------------
# generate time grids list
def time_grids(t1,t2,freq=4):
    num_grid = int(t2 * freq) + 1
    ts = np.linspace(t1, t1+t2, num=num_grid)
    return ts

# %% --------------------------------------------------
# Vasicek class definitions
# -----------------------------------------------------
# Dynamics class
class Vasicek:

    def __init__(self, r0=0.01, theta=0.85, mu=0.04, sigma=0.05):
        # set up
        self.r0 = r0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma

    def __proceed_time(self, dt):
        self.rs = self.rs + self.theta*(self.mu - self.rs)*dt \
                  + self.sigma*normal(size=self.num_mc) * np.sqrt(dt)
        return self.rs

    def monte_carlo_simlation(self, dt=0.01, num_mc=1000, horizon=10.0, seed=1234):
        # initialize state
        self.num_mc = num_mc
        self.rs = np.ones(num_mc) * self.r0
        self.hist_rs = None
        np.random.seed(seed)
        num_grids = int(horizon/dt) + 1
        self.ts = np.linspace(0.0, horizon, num=num_grids)
        # simulation
        hist_rs = np.zeros((num_grids, self.num_mc), np.float)
        hist_rs[0, :] = self.rs
        for i in tqdm(range(num_grids-1)):
            hist_rs[i+1, :] = self.__proceed_time(dt)
        self.hist_rs = hist_rs

    def get_zcb(self, s, t):
        # corresponding index to time s
        idx = np.argmin(np.abs(self.ts - s))
        tau = t - s
        # extract parameters
        theta, mu, sigma = self.theta, self.mu, self.sigma
        # coefficients
        B = 1/theta*(1 - np.exp(-theta*tau))
        A = np.exp((mu - sigma**2/theta**2/2)*(B - tau) - sigma**2/theta/4*B**2)
        # discount bond
        zcb = A.reshape(1, -1)*np.exp(-B.reshape(1, -1)*self.hist_rs[idx,:].reshape(-1, 1))
        return zcb.flatten()

    def get_fwd(self, s, t1, t2):
        # corresponding index to time s
        fwd = (self.get_zcb(s,t1)/self.get_zcb(s,t2) - 1.0) / (t2 - t1)
        return fwd.flatten()

# %% --------------------------------------------------
# IRS class definitions
# -----------------------------------------------------
class IRS:

    def __init__(self, tenor=7.0, X=0.05, side=1):
        self.T = tenor  # contract lifespan
        self.X = X      # fixed interest rate
        self.side = side  # 1 for payer, -1 for receiver
        self.ts_fixed = time_grids(0.0, tenor, freq=2)  # semi-annually
        self.ts_float = time_grids(0.0, tenor, freq=4)  # quarterly

    def pricing(self, s, dynamics):
        # copy params
        ts_fixed = self.ts_fixed
        ts_float = self.ts_float
        # error handle
        if np.max(ts_float)<=s or np.max(ts_fixed)<=s:
            return np.zeros(dynamics.num_mc)
        # new grids
        ts_fixed_s, ts_float_s = [], []
        is_fixed1st = True
        for i in range(len(ts_fixed)):
            if ts_fixed[i]-s>0.0 and is_fixed1st:
                ts_fixed_s.append(ts_fixed[i-1]-s)
                is_fixed1st = False
            if ts_fixed[i]-s>0.0:
                ts_fixed_s.append(ts_fixed[i]-s)
        is_float1st = True
        for i in range(len(ts_float)):
            if ts_float[i]-s>0.0 and is_float1st:
                ts_float_s.append(ts_float[i-1]-s)
                is_float1st = False
            if ts_float[i]-s>0.0:
                ts_float_s.append(ts_float[i]-s)
        tau_fixed_s = [ts_fixed_s[i+1] - ts_fixed_s[i] for i in range(len(ts_fixed_s)-1)]
        tau_float_s = [ts_float_s[i+1] - ts_float_s[i] for i in range(len(ts_float_s)-1)]
        # fixed leg pv
        s_prev = s + ts_float_s[0]
        pv_fixed = np.array([tau_fixed_s[i] * dynamics.get_zcb(s, s+ts_fixed_s[i+1]) * self.X \
                                for i in range(len(tau_fixed_s))]).sum(axis=0).flatten()
        # float leg pv: 1st CF being fixed
        pv_float = np.array(tau_float_s[0] * dynamics.get_zcb(s, s+ts_float_s[1]) \
                            * dynamics.get_fwd(s_prev, s_prev, s+ts_float_s[1]))
        # float CF after 1st CF
        if len(tau_float_s) > 1: # if still exist floating CF
            pv_float2 = np.array([tau_float_s[i] * dynamics.get_zcb(s, s+ts_float_s[i+1]) \
                                 * dynamics.get_fwd(s, s+ts_float_s[i], s+ts_float_s[i+1]) \
                                 for i in range(1,len(tau_float_s))]).sum(axis=0).flatten()
            pv_float = pv_float + pv_float2
        return self.side * (pv_float - pv_fixed)
        #return pv_float
        #return pv_fixed

# %% --------------------------------------------------
# test
# -----------------------------------------------------
# simulation setting
num_sim = 40000
dt = 0.01
horizon = 10.0
# build instances
dynamics = Vasicek(r0=0.01, theta=0.85, mu=0.04, sigma=0.08)
contract = IRS(X=0.03, tenor=7.0)

# simulation
dynamics.monte_carlo_simlation(dt=0.01, num_mc=num_sim, horizon=horizon)
ts_sim = np.linspace(0.0, 8.0, num=500)
price = np.array([contract.pricing(t, dynamics) for t in ts_sim])
# figure
plt.plot(ts_sim, np.maximum(price,0.0).mean(axis=1), label='EPE')
plt.plot(ts_sim, np.minimum(price,0.0).mean(axis=1), label='ENE')
plt.plot(ts_sim, price.mean(axis=1), label='EE')
plt.show()



# %%
