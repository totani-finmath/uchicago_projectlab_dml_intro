# %% --------------------------------------------------
# Import packages
# -----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
import importlib
from copy import deepcopy as copy


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

    def __init__(self, r0=0.04, theta=0.30, mu=0.05, 
                 sigma=0.08, num_mc=1000, seed=1234):
        # set up
        self.r0 = r0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.num_mc = num_mc
        # initialize state
        self.rs = np.ones(num_mc) * r0
        self.hist_rs = None
        np.random.seed(seed)

    def proceed_time(self, dt):
        self.rs = self.rs + self.theta*(self.mu - self.rs)*dt \
                  + self.sigma*normal(size=self.num_mc) * np.sqrt(dt)
        return self.rs

    def monte_carlo_simlation(self, dt=0.01, horizon=10.0):
        num_grids = int(horizon/dt) + 1
        self.ts = np.linspace(0.0, horizon, num=num_grids)
        hist_rs = np.zeros((num_grids, self.num_mc), np.float)
        hist_rs[0, :] = self.rs
        for i in range(num_grids-1):
            hist_rs[i+1, :] = self.proceed_time(dt)
        self.hist_rs = hist_rs

    def get_zcb(self, s, t):
        # corresponding index to time s
        idx = np.argmin(np.abs(self.ts - s))
        # extract parameters
        theta, mu, sigma = self.theta, self.mu, self.sigma
        # coefficients
        B = 1 / theta * (1 - np.exp(-theta * (t - s)))
        A = np.exp((mu - sigma**2 / theta**2 / 2) * (B - t + s) - sigma**2 / theta / 4 * B**2)
        # discount bond
        zcb = A.reshape(1, -1) * np.exp(-B.reshape(1, -1) * self.hist_rs[idx, :].reshape(-1, 1))
        return zcb.flatten()

    def get_fwd(self, s, t1, t2):
        # corresponding index to time s
        idx = np.argmin(np.abs(self.ts - s))
        fwd = (self.get_zcb(s,t1)/self.get_zcb(s,t2) - 1.0) / (t2 - t1)
        return fwd.flatten()

# %% --------------------------------------------------
# IRS class definitions
# -----------------------------------------------------
class IRS:

    def __init__(self, tenor=7.0, X=0.03, side=1):
        self.T = tenor  # contract lifespan
        self.X = X      # fixed interest rate
        self.side = side  # 1 for payer, -1 for receiver
        self.ts_fixed = time_grids(0.0, tenor, freq=2)  # semi-annually
        self.ts_float = time_grids(0.0, tenor, freq=4)  # quarterly

    def pricing(self, s, dynamics):
        # copy params
        ts_fixed = self.ts_fixed
        ts_float = self.ts_float
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
        pv_fixed = np.array([tau_fixed_s[i] * dynamics.get_zcb(s, ts_fixed_s[i+1]) * self.X \
                                for i in range(len(tau_fixed_s))]).sum(axis=0).flatten()
        # float leg pv: 1st CF being fixed
        pv_float = np.array(tau_float_s[0] * dynamics.get_zcb(s, ts_float_s[1]) \
                            * dynamics.get_fwd(s, ts_float_s[0], ts_float_s[1]))
        # float CF after 1st CF
        if len(tau_float_s) > 1: # if still exist floating CF
            pv_float2 = np.array([tau_float_s[i] * dynamics.get_zcb(s, ts_float_s[i+1]) \
                                 * dynamics.get_fwd(s, ts_float_s[i], ts_float_s[i+1]) \
                               for i in range(1,len(tau_float_s))]).sum(axis=0).flatten()
            pv_float += pv_float
        return self.side * (pv_float - pv_fixed)

# %% --------------------------------------------------
# test
# -----------------------------------------------------
dynamics = Vasicek()
dynamics.monte_carlo_simlation()
plt.plot(dynamics.hist_rs[:,0])

contract = IRS()
test = contract.pricing(0.3333, dynamics)

# %%


