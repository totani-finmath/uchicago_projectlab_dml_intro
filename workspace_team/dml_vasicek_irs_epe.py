import importlib
from copy import deepcopy as copy
import matplotlib.pyplot as plt
import numpy as np

import lib_dml as ldml

importlib.reload(ldml)


# %% --------------------------------------------------
# Class definitions
# -----------------------------------------------------
# Dynamics class
class Dynamics:
    """
    Vasicek short rate dynamics.

    Governing equations:
        1. dr(t) = theta * (mu - r(t)) * dt + sigma * dW
        2. r(0) = r0
    """

    def __init__(self, r0=0.04, theta=3.0, mu=0.05, sigma=0.08):
        """
        Parameters:
            r0 : float
                initial value of rate process
            theta : float
                speed of mean reversion (constant in t)
            mu : float
                mean of rate process (constant in t)
            sigma : float
                volatility of rate process (constant in t)
        """

        self.r0 = r0
        self.theta = theta
        self.mu = mu
        self.sigma = sigma


# Contract class
class Contract:
    """
    Interest rate swap.

    References
    ----------
        [1] Brigo & Mercurio (2006), sec. 1.5
    """

    def __init__(self, T1=0.5, T=7.0, t=0., K=0.03, side=1, horizon_=10):
        self.T1 = T1  # time until contract initiation
        self.T = T  # contract lifespan
        self.T2 = T1 + T  # time until contract expiration
        self.t = t  # evaluation date
        self.K = K  # fixed interest rate
        self.side = side  # 1 for payer, -1 for receiver
        self.ts_fixed = time_grids(0.0, T, freq=2) - t  # semi-annually
        self.ts_float = time_grids(0.0, T, freq=4) - t  # quarterly
        self.horizon = horizon_


# generator
class VasicekIRS:
    def __init__(self,
                 dynamics,
                 contract):
        self.dynamics = copy(dynamics)
        self.contract = copy(contract)

        self.__X = None
        self.__Y = None
        self.__Z = None

    def trainingSet(self, m, seed=None):
        r0, theta, mu, sigma = self.dynamics.r0, self.dynamics.theta, self.dynamics.mu, self.dynamics.sigma
        dt = self.contract.T1 + self.contract.t
        np.random.seed(seed)

        # compute initial interest rates by Monte Carlo
        rt = r0 + theta * (mu - r0) * dt + sigma * np.random.normal(size=m) * np.sqrt(dt)

        # exposures, diff_exposures = [], []
        # for i in range(m):
        #     self.dynamics.r0 = rt[i]
        e, de = exposure_irs_vasicek_MonteCarlo(self.dynamics, self.contract, rt)
        #     exposures.append(e[0])
        #     diff_exposures.append(de[0])

        positive_exposures = np.maximum(e, 0)
        diff_exposures = np.where(positive_exposures > 0, de, 0)

        X = rt
        Y = positive_exposures
        Z = np.array(diff_exposures)

        self.__X = X
        self.__Y = Y
        self.__Z = Z

        return X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)

    def testSet(self, num=0, seed=None):
        return self.__X.reshape(-1, 1), self.__X.reshape(-1, 1), self.__Y.reshape(-1, 1), self.__Z.reshape(-1, 1)


# %% --------------------------------------------------
# Functions
# -----------------------------------------------------
def get_zcb_vasicek(dynamics, t, T, r):
    """
    Price of zero-coupon bond under Vasicek dynamics
    """
    theta, mu, sigma = dynamics.theta, dynamics.mu, dynamics.sigma

    B = 1 / theta * (1 - np.exp(-theta * (T - t)))
    A = np.exp((mu - sigma ** 2 / theta ** 2 / 2) * (B - T + t) - sigma ** 2 / theta / 4 * B ** 2)

    zcb = A.reshape(1, -1) * np.exp(-B.reshape(1, -1) * r.reshape(-1, 1))

    return zcb


def get_zcb_diff_vasicek(dynamics, t, T, r):
    """
    Price of zero-coupon bond under Vasicek dynamics
    """
    theta, mu, sigma = dynamics.theta, dynamics.mu, dynamics.sigma

    B = 1 / theta * (1 - np.exp(-theta * (T - t)))
    A = np.exp((mu - sigma ** 2 / theta ** 2 / 2) * (B - T + t) - sigma ** 2 / theta / 4 * B ** 2)

    zcb_diff = - A.reshape(1, -1) * B.reshape(1, -1) * np.exp(-B.reshape(1, -1) * r.reshape(-1, 1))

    return zcb_diff


# def get_zcb_diff_vasicek(dynamics, t, T, r):
#     """
#     Price of zero-coupon bond under Vasicek dynamics
#     """
#     theta, mu, sigma = dynamics.theta, dynamics.mu, dynamics.sigma
#
#     B = 1 / theta * (1 - np.exp(-theta * (T - t)))
#     A = np.exp((mu - sigma ** 2 / theta ** 2 / 2) * (B - T + t) - sigma ** 2 / theta / 4 * B ** 2)
#
#     dB_dt = -np.exp(-theta * (T - t))
#     dA_dt = ((dB_dt + 1) * (mu - sigma**2 / (2 * theta**2)) - sigma**2 * B * dB_dt / (2 * theta)) * A
#
#     zcb_diff = dA_dt * np.exp(r * B) + r * A * np.exp(r * B) * dB_dt
#
#     return zcb_diff


def get_fwd_vasicek(dynamics, T, S, r):
    """
    Forward interest rate under Vasicek model
    """
    tau = S - T
    fwd = (get_zcb_vasicek(dynamics, 0, T, r) / get_zcb_vasicek(dynamics, 0, S, r) - 1) / tau
    return fwd


def get_fwd_diff_vasicek(dynamics, T, S, r):
    """
    Forward interest rate under Vasicek model
    """
    tau = S - T
    fwd_diff = ((get_zcb_diff_vasicek(dynamics, 0, T, r) * get_zcb_vasicek(dynamics, 0, S, r) -
                 get_zcb_vasicek(dynamics, 0, T, r) * get_zcb_diff_vasicek(dynamics, 0, S, r)) /
                get_zcb_vasicek(dynamics, 0, S, r)**2) / tau
    return fwd_diff


def time_grids(t1, t2, freq=4):
    num_grid = int(t2 * freq) + 1
    ts = np.linspace(t1, t1 + t2, num=num_grid)
    return ts


def monte_carlo_vasicek(dynamics, contract, N=1000, seed=1234, simulations=100000):
    """
    Produces the stochastic paths of a vasicek SDE
    """
    ts_pre = np.linspace(0.0, contract.T, N + 1) - contract.t
    ts_agg = sorted(set(ts_pre) | set(contract.ts_fixed) | set(contract.ts_float))

    t = np.array(ts_agg)
    dt = np.diff(t)[0]

    # Vasicek parameters
    r0, theta, mu, sigma = dynamics.r0, dynamics.theta, dynamics.mu, dynamics.sigma

    # initialization
    np.random.seed(seed)  # reset random seed for reproducibility

    # generate Vasicek paths
    rates = np.zeros((simulations, N + 1))
    rates[:, 0] = r0
    for i in range(1, N + 1):
        rates[:, i] = rates[:, i - 1] + theta * (mu - rates[:, i - 1]) * dt \
                      + sigma * np.random.normal(size=simulations) * np.sqrt(dt)

    # store into np.array
    return t, rates


def exposure_irs_vasicek_MonteCarlo(dynamics, contract, r=None):
    """
    computes mark-to-market value of single IRS for all times from contract initiation to expiry assuming Vasicek
    dynamics for the floating rate
    """
    T, K, omega, ts_fixed, ts_float = contract.T, contract.K, contract.side, contract.ts_fixed, contract.ts_float
    if r is None:
        r = dynamics.r0

    # fixed leg computations
    ts_fixed_t = ts_fixed[ts_fixed > 0.]  # time until future payment dates
    tau_fixed_t = np.diff(ts_fixed)[0]  # fraction of annualized payment amount
    zcb_fixed_t = get_zcb_vasicek(dynamics, 0, ts_fixed_t, r)  # discount factor for fixed leg
    zcb_fixed_diff_t = get_zcb_diff_vasicek(dynamics, 0, ts_fixed_t, r)

    pv_fixed = (zcb_fixed_t * tau_fixed_t * K).sum(1)  # sum of discounted future CFs
    pv_fixed_diff = (zcb_fixed_diff_t * tau_fixed_t * K).sum(1)

    # floating leg computations
    ts_float_t = ts_float[ts_float > 0.]  # time until future payment dates
    tau_float_t = np.diff(ts_float)[0]  # fraction of annualized payment amount
    zcb_float_t = get_zcb_vasicek(dynamics, 0, ts_float_t[0], r)  # discount factor for floating leg
    zcb_float_diff_t = get_zcb_diff_vasicek(dynamics, 0, ts_float_t[0], r)
    fwd_float_t = get_fwd_vasicek(dynamics, ts_float_t[0] - tau_float_t, ts_float_t[0], r)  # implied forward rates
    fwd_float_diff_t = get_fwd_diff_vasicek(dynamics, ts_float_t[0] - tau_float_t, ts_float_t[0], r)

    pv_float = (tau_float_t * zcb_float_t * fwd_float_t).flatten()  # most recent CF
    pv_float_diff = tau_float_t * (zcb_float_diff_t * fwd_float_t + zcb_float_t * fwd_float_diff_t).flatten()

    # sum of PVs of future CFs (if any)
    if len(ts_float_t) > 1:
        zcb_float_t = get_zcb_vasicek(dynamics, 0, ts_float_t[1:], r)
        zcb_float_diff_t = get_zcb_diff_vasicek(dynamics, 0, ts_float_t[1:], r)
        fwd_float_t = get_fwd_vasicek(dynamics, ts_float_t[:-1], ts_float_t[1:], r)
        fwd_float_diff_t = get_fwd_diff_vasicek(dynamics, ts_float_t[:-1], ts_float_t[1:], r)

        pv_float += (tau_float_t * zcb_float_t * fwd_float_t).sum(1)
        pv_float_diff += tau_float_t * (zcb_float_diff_t * fwd_float_t + zcb_float_t * fwd_float_diff_t).sum(1)

    pv = omega * (pv_float - pv_fixed)
    pv_diff = omega * (pv_float_diff - pv_fixed_diff)

    return pv, pv_diff


if __name__ == '__main__':
    dynamics_vasicek = Dynamics(r0=0.01, sigma=0.05, theta=0.5, mu=0.03)
    contract_irs = Contract(T=7.0, t=0., K=0.015, side=1)

    generator = VasicekIRS(dynamics_vasicek, contract_irs)
    # generator.trainingSet(1000)

    # simulation set sizes to perform
    # sizes = [1024, 8192]
    sizes = [1024, 16384]

    # show delta?
    showDeltas = True

    # seed
    # simulSeed = 1234
    simulSeed = np.random.randint(0, 10000)
    print("using seed %d" % simulSeed)

    # number of test scenarios
    nTest = 100

    xAxis, yTest, dydxTest, values, deltas, _ = ldml.test(generator, sizes, nTest, simulSeed, None, None)

    ldml.graph("EPE", values, xAxis, "", "values", yTest, sizes, False)

    if showDeltas:
        ldml.graph("EPE", deltas, xAxis, "", "deltas", dydxTest, sizes, False)
