#%% --------------------------------------------------
# import packages
# ----------------------------------------------------
# numerical
import numpy as np
import pandas as pd
from scipy import integrate as itg
from scipy import optimize as opt
from scipy.stats import norm
# helper
import time as tm
import pickle as pk
# visualization
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

#%% --------------------------------------------------
# constants
# ----------------------------------------------------
upr_itg = np.inf
lwr_itg = -np.inf
upr_itg = 100.0
lwr_itg = -100.0
y_bar_init = 0.001
lst_bounds = [(1.0e-5,5.0), (1.0e-5,5.0), (1.0e-5,5.0), (1.0e-5,5.0), (-1.0,1.0)]

#%% --------------------------------------------------
# parameters initial
# ----------------------------------------------------
alpha1_init = 0.5
alpha2_init = 0.1
sigma1_init = 0.05
sigma2_init = 0.02
rho_init    = -0.5
# aggregate
params_init = [alpha1_init, alpha2_init, sigma1_init, sigma2_init, rho_init]

#%% --------------------------------------------------
# contracts sample
# ----------------------------------------------------
def build_contract_swpn(t_mat,t_tnr,strike,omega):
    # assuming frequencies are same for both legs
    num_grid = int(t_tnr * 4) + 1
    ts = np.linspace(t_mat, t_mat+t_tnr, num=num_grid)
    contract = {'grids':ts,'strike':strike,'side': omega}
    return contract

#%% --------------------------------------------------
# sample market data
# ----------------------------------------------------
# import zcb data
df_zcb = pd.read_excel('zcb_usd.xlsx',index_col=0,header=0)
key_lib3m = 'USD#LIBOR3M_Curve'
key_sofr  = 'USD#SOFR_Curve'
# generate instantaneous forward rates
numer = (df_zcb/df_zcb.shift(-1)-1.0)
denom = df_zcb.index[1:] - df_zcb.index[:-1]
sr_fwd_lib3m = numer[key_lib3m].iloc[:-1]/denom
sr_fwd_sofr  = numer[key_sofr].iloc[:-1]/denom
# linear interpolated value
def get_lininterp(sr,x):
    return np.interp(x,sr.index,sr.values)
# get zcb for libor3m at time s>=0.0 where t=0.0 for now
def get_zcb_lib3m(s,t=0.0):
    return get_lininterp(df_zcb[key_lib3m],s)/get_lininterp(df_zcb[key_lib3m],t)
# get zcb for sofr at time s>=0.0 where t=0.0 for now
def get_zcb_sofr(s,t=0.0):
    return get_lininterp(df_zcb[key_sofr],s)/get_lininterp(df_zcb[key_sofr],t)
# generate time grids list
def time_grids(t1,t2,freq=4):
    num_grid = int(t2 * freq) + 1
    ts = np.linspace(t1, t1+t2, num=num_grid)
    return ts
# swap rate
def swap_rate_ts(ts,t=0.0):
    t_stt = ts[0]
    t_end = ts[-1]
    taus = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
    denom = np.sum([taus[i]*get_zcb_lib3m(ts[i+1],t) for i in range(len(taus))])
    numer = get_zcb_lib3m(t_stt,t) - get_zcb_lib3m(t_end,t)
    return numer/denom

#%% ##################################################
# G2++ implementation
# ####################################################
#%% --------------------------------------------------
# functions: convert parameters
# ----------------------------------------------------
# Brigo & Mercurio (4.19)
def M_x_g2pp(s,t,T,alpha1,alpha2,sigma1,sigma2,rho):
    trm1 = (sigma1**2/alpha1**2 + rho*(sigma1*sigma2)/alpha1/alpha2) * (1.0 - np.exp(-alpha1*(t-s)))
    trm2 = (-1.0) * sigma1**2*0.5/alpha1**2 * (np.exp(-alpha1*(T-t)) - np.exp(-alpha1*(T+t-2*s)))
    trm3 = (-1.0) * rho*sigma1*sigma2/alpha2/(alpha1+alpha2) * (np.exp(-alpha2*(T-t)) - np.exp(-alpha2*T - alpha1*t + (alpha1+alpha2)*s))
    return trm1 + trm2 + trm3
# Brigo & Mercurio (4.19)
def M_y_g2pp(s,t,T,alpha1,alpha2,sigma1,sigma2,rho):
    return M_x_g2pp(s,t,T,alpha2,alpha1,sigma2,sigma1,rho)
# Brigo & Mercurio (4.31)
def sigma_xy_g2pp(t0,alpha,sigma):
    return sigma * np.sqrt((1.0-np.exp(-2*alpha*t0))*0.5/alpha)
# Brigo & Mercurio (4.31)
def rho_xy_g2pp(t0,alpha1,alpha2,sigma1,sigma2,rho,sigma_x,sigma_y):
    return rho*sigma1*sigma2/(alpha1 + alpha2)/sigma_x/sigma_y * (1.0 - np.exp(-(alpha1+alpha2)*t0))
# Brigo & Mercurio (4.31)
def ci_g2pp(i,X,tau): # 1<=i<=n
    if i is not len(tau)-1:
        return X*tau[i]
    else:
        return 1.0 + X*tau[i]

#%% --------------------------------------------------
# functions: bond pricings
# ----------------------------------------------------
# Brigo & Mercurio (4.10)
def V_g2pp(t,T,alpha1,alpha2,sigma1,sigma2,rho):
    trm1 = sigma1**2/alpha1**2 * (T-t + 2.0/alpha1*np.exp(-alpha1*(T-t)) -0.5/alpha1*np.exp(-2*alpha1*(T-t)) - 1.5/alpha1)
    trm2 = sigma2**2/alpha2**2 * (T-t + 2.0/alpha2*np.exp(-alpha2*(T-t)) -0.5/alpha2*np.exp(-2*alpha2*(T-t)) - 1.5/alpha2)
    trm3 = 2.0*rho*sigma1*sigma2/alpha1/alpha2 * (T-t + (np.exp(-alpha1*(T-t))-1.0)/alpha1 + (np.exp(-alpha2*(T-t))-1.0)/alpha2 - (np.exp(-(alpha1+alpha2)*(T-t))-1.0)/(alpha1+alpha2))
    return trm1 + trm2 + trm3
# Brigo & Mercurio (4.15)
def A_g2pp(t,T,alpha1,alpha2,sigma1,sigma2,rho):
    return get_zcb_lib3m(T,0.0)/get_zcb_lib3m(t,0.0)*np.exp(0.5*(V_g2pp(t,T,alpha1,alpha2,sigma1,sigma2,rho)\
                                                            - V_g2pp(0,T,alpha1,alpha2,sigma1,sigma2,rho)\
                                                            + V_g2pp(0,t,alpha1,alpha2,sigma1,sigma2,rho)))
# Brigo & Mercurio (4.15)
def B_g2pp(z,t,T):
    return (1.0-np.exp(-z*(T-t)))/z

#%% --------------------------------------------------
# functions: input functionals
# ----------------------------------------------------
# Brigo & Mercurio (4.31)
def h1_g2pp(x,y_bar,mu_x,mu_y,sigma_x,sigma_y,rho_xy):
    return (y_bar - mu_y)/sigma_y/np.sqrt(1.0-rho_xy**2) - rho_xy*(x-mu_x)/sigma_x/np.sqrt(1.0-rho_xy**2)
# Brigo & Mercurio (4.31)
def h2i_g2pp(sigma_y,rho_xy,h1,B_bi):
    return h1 + B_bi*sigma_y*np.sqrt(1.0-rho_xy**2)
# Brigo & Mercurio (4.31)
def lamb_i_g2pp(x,i,A_i,B_ai,ci):
    return ci*A_i*np.exp(-B_ai*x)
# Brigo & Mercurio (4.31)
def kappa_i_g2pp(x,A_i,B_bi,mu_x,mu_y,sigma_x,sigma_y,rho_xy):
    trm = mu_y - 0.5*(1.0-rho_xy**2)*sigma_y**2*B_bi + rho_xy*sigma_y*(x-mu_x)/sigma_x
    return (-1.0)*B_bi*trm

#%% --------------------------------------------------
# optimization
# ----------------------------------------------------
# constraint of swaption formula, Brigo & Mercurio (4.31)
def constraint_ybar(y,x,cs,As,B1s,B2s):
    trms = [cs[i]*As[i]*np.exp(-B1s[i]*x-B2s[i]*y) for i in range(len(cs))]
    return np.sum(trms) - 1.0
# swaption formula integrand, Brigo & Mercurio (4.31)
def swpn_integrand_g2pp(x,params,contract):
    # unpack parameters
    alpha1, alpha2 = params[0], params[1]
    sigma1, sigma2 = params[2], params[3]
    rho = params[4]
    ts  = contract['grids']
    tau = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
    K   = contract['strike']
    omega = contract['side']
    num_grid = len(ts)
    # preparation #1
    mu_x    = -M_x_g2pp(0,ts[0],ts[0],alpha1,alpha2,sigma1,sigma2,rho)
    mu_y    = -M_y_g2pp(0,ts[0],ts[0],alpha1,alpha2,sigma1,sigma2,rho)
    sigma_x = sigma_xy_g2pp(ts[0],alpha1,sigma1)
    sigma_y = sigma_xy_g2pp(ts[0],alpha2,sigma2)
    rho_xy  = rho_xy_g2pp(ts[0],alpha1,alpha2,sigma1,sigma2,rho,sigma_x,sigma_y)
    # preparation #2
    As = [A_g2pp(ts[0],ts[i+1],alpha1,alpha2,sigma1,sigma2,rho) for i in range(num_grid-1)]
    B1s = [B_g2pp(alpha1,ts[0],ts[i+1]) for i in range(num_grid-1)]
    B2s = [B_g2pp(alpha2,ts[0],ts[i+1]) for i in range(num_grid-1)]
    cs  = [ci_g2pp(i,K,tau) for i in range(num_grid-1)]
    # root finding
    y_bar = opt.root(constraint_ybar,y_bar_init,\
                args=(x,cs,As,B1s,B2s),method='hybr').x[0]
    # preparation #3
    h1  = h1_g2pp(x,y_bar,mu_x,mu_y,sigma_x,sigma_y,rho_xy)
    h2s = [h2i_g2pp(sigma_y,rho_xy,h1,B2s[i]) for i in range(num_grid-1)]
    lambs  = [lamb_i_g2pp(x,i+1,As[i],B1s[i],cs[i]) for i in range(num_grid-1)]
    kappas = [kappa_i_g2pp(x,As[i],B2s[i],mu_x,mu_y,sigma_x,sigma_y,rho_xy) for i in range(num_grid-1)]
    # integrand
    trm1 = np.exp(-0.5*((x-mu_x)/sigma_x)**2) / sigma_x/np.sqrt(2.0*np.pi)
    trm2 = norm.cdf(-omega*h1)
    trm3 = np.sum([lambs[i]*np.exp(kappas[i])*norm.cdf(-omega*h2s[i]) for i in range(num_grid-1)])
    return trm1 * (trm2 - trm3)
# swaption price, Brigo & Mercurio (4.31)
def swpn_price_g2pp(params,contract):
    ts = contract['grids']
    omega = contract['side']
    integral = itg.quad(swpn_integrand_g2pp,lwr_itg,upr_itg,args=(params,contract))[0] # discard error
    return integral * omega * get_zcb_lib3m(ts[0],0.0)

#%% --------------------------------------------------
# G2++ bond price given Gaussian factors
# ----------------------------------------------------
# Brigo & Mercurio (4.14)
def get_zcb_g2pp(t,T,params,x1t,x2t):
    alpha1, sigma1 = params[0], params[2]
    alpha2, sigma2 = params[1], params[3]
    rho = params[4]
    # Vs
    v1 = V_g2pp(t,T,alpha1,alpha2,sigma1,sigma2,rho)
    v2 = V_g2pp(0,T,alpha1,alpha2,sigma1,sigma2,rho)
    v3 = V_g2pp(0,t,alpha1,alpha2,sigma1,sigma2,rho)
    # coef factors
    b1 = B_g2pp(alpha1,t,T)
    b2 = B_g2pp(alpha2,t,T)
    # aggregation
    At = 0.5*(v1 - v2 + v3) - b1*x1t - b2*x2t
    return get_zcb_lib3m(T)/get_zcb_lib3m(t) * np.exp(At)
def get_fwd_g2pp(t1,t2,params,x1t,x2t):
    tau = t2 - t1
    fwd = (get_zcb_g2pp(0.0,t1,params,x1t,x2t)/get_zcb_g2pp(0.0,t2,params,x1t,x2t) - 1.0)/tau
    return fwd

#%% --------------------------------------------------
# test: pricing
# ----------------------------------------------------
# test setting
t_mat_test = 1.0
t_tnr_test = 3.0
omega_test = 1
# test preparation
num_grid_test = int(t_tnr_test * 4) + 1
ts_test = np.linspace(t_mat_test, t_mat_test+t_tnr_test, num=num_grid_test)
X_test = swap_rate_ts(ts_test)
cont_test = build_contract_swpn(t_mat_test,t_tnr_test,X_test,omega_test)
# calculation
price_test = swpn_price_g2pp(params_init,cont_test)
print('test price',t_mat_test,'x',t_tnr_test,':',price_test)

#%% ##################################################
# optimization part
# ####################################################
#%% --------------------------------------------------
# iv/price conversions
# ----------------------------------------------------
# black model
def black(S,K,T,sigma,omega):
    d1 = (np.log(S/K) + 0.5*sigma**2*T)/sigma/np.sqrt(T)
    d2 = (np.log(S/K) - 0.5*sigma**2*T)/sigma/np.sqrt(T)
    return omega * (S*norm.cdf(omega*d1) - K*norm.cdf(omega*d2))
# swaption by log-normal implied vol
def vol_to_price_black(S,K,sigma,omega,ts):
    tau = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
    ann = np.sum([tau[i] * get_zcb_lib3m(ts[i+1],0.0) for i in range(len(tau))])
    return black(S,K,ts[0],sigma,omega) * ann
# log-normal implied vol by swaption
def price_to_vol_black(price,S,K,omega,ts,init_iv=0.01):
    err = lambda sigma: (price - vol_to_price_black(S,K,sigma,omega,ts))/price
    iv = opt.root(err,init_iv,method='hybr').x[0]
    return iv
# bachelier model
def bachelier(S,K,T,sigma,omega):
    d1 = (S-K)/sigma/np.sqrt(T)
    return omega*(S-K)*norm.cdf(omega*d1) + sigma*np.sqrt(T)*norm.pdf(omega*d1)
# swaption by normal implied vol
def vol_to_price_normal(S,K,sigma,omega,ts):
    tau = [t2 - t1 for t1, t2 in zip(ts[:-1], ts[1:])]
    ann = np.sum([tau[i] * get_zcb_lib3m(ts[i+1],0.0) for i in range(len(tau))])
    return bachelier(S,K,ts[0],sigma,omega) * ann
# normal implied vol by swaption
def price_to_vol_normal(price,S,K,omega,ts,init_iv=0.01):
    err = lambda sigma: (price - vol_to_price_normal(S,K,sigma,omega,ts))/price
    iv = opt.root(err,init_iv,method='hybr').x[0]
    return iv

#%% --------------------------------------------------
# import sample data
# ----------------------------------------------------
filename = 'matrix_normal.xlsx'
df_vols = pd.read_excel(filename,sheet_name='vols_normal',index_col=0)
df_wgts = pd.read_excel(filename,sheet_name='weights_normal',index_col=0)

# generating data
lst_data_opt = []
lst_data_all = []
for idx in df_vols.index:
    for col in df_vols.columns:
        lst_data_all.append([float(idx),float(col),df_vols[col][idx]/10000])
        if df_wgts[col][idx] > 0:
            lst_data_opt.append([float(idx),float(col),df_vols[col][idx]/10000])
# generate contract list for optimization
lst_contracts_opt = []
for i in range(len(lst_data_opt)):
    mem = lst_data_opt[i]
    # this contract
    ts = time_grids(mem[0],mem[1])
    swp_atm = swap_rate_ts(ts)
    omega = 1
    # swaption price
    price = vol_to_price_normal(swp_atm,swp_atm,mem[2],omega,ts)
    contract = {'tenor'  : (mem[0],mem[1]),
                'grids'  : ts,
                'strike' : swp_atm,
                'side'   : omega,
                'price'  : price}
    lst_contracts_opt.append(contract)
# generate contract list all
lst_contracts_all = []
for i in range(len(lst_data_all)):
    mem = lst_data_all[i]
    # this contract
    ts = time_grids(mem[0],mem[1])
    swp_atm = swap_rate_ts(ts)
    omega = 1
    # swaption price
    price = vol_to_price_normal(swp_atm,swp_atm,mem[2],omega,ts)
    contract = {'tenor'  : (mem[0],mem[1]),
                'grids'  : ts,
                'strike' : swp_atm,
                'side'   : omega,
                'price'  : price}
    lst_contracts_all.append(contract)

#%% --------------------------------------------------
# objective function
# ----------------------------------------------------
# objective function
def opt_objective_g2pp(params, lst_contracts):
    err_square = []
    for contract in lst_contracts:
        err = contract['price'] - swpn_price_g2pp(params,contract)
        err_square.append(err**2)
    return np.sqrt(np.mean(err_square))

#%% --------------------------------------------------
# test
# ----------------------------------------------------
#rerr = opt_objective_g2pp(params,lst_contracts)
if False:
    for cont in lst_contracts_opt:
        price = swpn_price_g2pp(params,cont)
        print(cont['tenor'])
        print(price)

#%% --------------------------------------------------
# calibration of G2++ model
# ----------------------------------------------------
# optimizer
# method: 'Powell', 'TNC', 'SLSQP'
algo = 'SLSQP'
is_calib = False
if is_calib:
    time_stt = tm.time()
    res = opt.minimize(opt_objective_g2pp,params_init,method=algo,args=(lst_contracts_opt),bounds=lst_bounds)
    time_end = tm.time()
    time_exe = time_end - time_stt
    print(time_exe/60.0,'[min]')
    pk.dump(res,open('res_calib_org.pkl','wb'))
else:
    res = pk.load(open('res_calib_org.pkl','rb'))
# store params
params_new = list(res.x)

#%% --------------------------------------------------
# review results
# ----------------------------------------------------
'''
method='SLSQP'
upr_itg = np.inf
lwr_itg = -np.inf
params_init = [0.5, 0.1, 0.05, 0.02, -0.5]
174.80168081124623 [min]

     fun: 0.007100906657011682
     jac: array([-0.00368487,  0.00014465,  0.05815335,  0.05066651,  0.00305027])
 message: 'Optimization terminated successfully'
    nfev: 80
     nit: 10
    njev: 10
  status: 0
 success: True
       x: array([ 0.50048295,  0.0948413 ,  0.04117085,  0.01517368, -0.50044063])
'''
if False:
    for cont in lst_contracts_opt:
        price = swpn_price_g2pp(params_new,cont)
        print('Tenor >> ', cont['tenor'])
        print('G2++ price   :', price)
        print('Market price :', cont['price'])

#%% --------------------------------------------------
# rooting full surface of model vols
# ----------------------------------------------------
def model_vols_surface(params,lst_contracts):
    # preparation
    set_idx = set()
    set_col = set()
    for cont in lst_contracts:
        set_idx.add(cont['tenor'][0])
        set_col.add(cont['tenor'][1])
    lst_idx = list(set_idx)
    lst_col = list(set_col)
    lst_idx.sort()
    lst_col.sort()
    # build dataframe
    df_vols_model = pd.DataFrame(index=lst_idx,columns=lst_col)
    for cont in lst_contracts_all:
        ts = cont['grids']
        X = cont['strike']
        omega = cont['side']
        price = swpn_price_g2pp(params,cont)
        vol = price_to_vol_normal(price,X,X,omega,ts)
        df_vols_model[cont['tenor'][1]][cont['tenor'][0]] = vol*10000
        print('Tenor >> ', cont['tenor'],'\t: Completed')
    df_vols_model = df_vols_model.astype(float)
    return df_vols_model

#%% --------------------------------------------------
# visualization of vols surface
# ----------------------------------------------------
title_fmt = 'Swaption Vols Surface: '
def show_vols3d(df,label='Market'):
    x = df.columns
    y = df.index
    x, y = np.meshgrid(x,y)
    # figure
    fig = plt.figure(figsize=(7,4))
    ax = Axes3D(fig)
    surf = ax.plot_surface(x,y,df.values,cmap=cm.jet)
    fig.colorbar(surf)
    plt.title(title_fmt+label)
    plt.xlabel('Swap tenor')
    plt.ylabel('Option maturity')
    plt.show()

#%% --------------------------------------------------
# review results: vols surface
# ----------------------------------------------------
if False:
    # calculate model vols
    df_vols_model = model_vols_surface(params_new,lst_contracts_all)
    # target
    show_vols3d(df_vols,'Market')
    show_vols3d(df_vols_model,'G2++')

#%% ##################################################
# Monte Carlo simulation
# ####################################################
#%% --------------------------------------------------
# irs pricer
# ----------------------------------------------------
def pv_irs_g2pp(s,cont_irs,params,x1,x2,x1last,x2last):
    ts_fixed = cont_irs['grids_fixed']
    ts_float = cont_irs['grids_float']
    X, omega = cont_irs['strike'], cont_irs['side']
    try:
        num_mc = x1.shape[0] # assuming columns vector
    except:
        num_mc = 1
    # error handle
    if np.max(ts_float)<=s or np.max(ts_fixed)<=s:
        return np.zeros(num_mc)
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
    pv_fixed = np.array([tau_fixed_s[i] * get_zcb_g2pp(s,s+ts_fixed_s[i+1],params,x1,x2) * X \
                            for i in range(len(tau_fixed_s))]).T.sum(axis=1)
    # float leg pv
    # fixed 1st CF
    pv_float = np.array(tau_float_s[0] * get_zcb_g2pp(s,s+ts_float_s[1],params,x1_last,x2_last) \
                        * get_fwd_g2pp(ts_float_s[0],ts_float_s[1],params,x1_last,x2_last)).T 
    # float CF after 1st CF
    if len(tau_float_s) > 1: # if still floating
        pv_float += np.array([tau_float_s[i] * get_zcb_g2pp(s,s+ts_float_s[i+1],params,x1,x2) \
                             * get_fwd_g2pp(ts_float_s[i],ts_float_s[i+1],params,x1,x2) \
                             for i in range(1,len(tau_float_s))]).T.sum(axis=1)
    return omega * (pv_float - pv_fixed)

#%% --------------------------------------------------
# Monte Carlo simulation
# ----------------------------------------------------
# two factor simulator
def monte_carlo_g2pp(num_mc,horizon,params_mc,contract,num_grid=1000,seed=1234):
    ts_pre = np.linspace(0.0,horizon,num=num_grid+1)
    # simulation grids (need to have irs contract first)
    ts_agg = list(set(ts_pre)|set(contract['grids_fixed'])|set(contract['grids_float']))
    ts_agg.sort()
    ts_new = np.array(ts_agg)
    dts = ts_new[1:] - ts_new[:-1]
    # G2++ parameters
    alpha1, sigma1 = params_mc[0], params_mc[2]
    alpha2, sigma2 = params_mc[1], params_mc[3]
    rho = params_mc[4]
    # initialization
    np.random.seed(seed) # reset random seet for reproductivity
    x1 = np.array([0.0 for x in range(num_mc)])
    x2 = np.array([0.0 for x in range(num_mc)])
    lst_x1s, lst_x2s = [], []
    lst_x1s.append(x1)
    lst_x2s.append(x2)
    # generate G2++ paths
    for dt in dts:
        # generate brownian motions
        dw1 = np.random.normal(0.0,np.sqrt(dt),size=num_mc)
        dw2 = rho*dw1 + np.sqrt(1.0-rho**2)*np.random.normal(0.0,np.sqrt(dt),size=num_mc)
        x1 = x1 + (-alpha1*x1*dt + sigma1*dw1)
        x2 = x2 + (-alpha2*x2*dt + sigma2*dw2)
        lst_x1s.append(x1)
        lst_x2s.append(x2)
    # srote into np.array
    return np.array(lst_x1s).T, np.array(lst_x2s).T, ts_new

#%% --------------------------------------------------
# build irs portfolio
# ----------------------------------------------------
t_tnr = 7.0
X = 0.0125 #0.0125 for 7y, 0.0155 for 10y around ATM
omega = -1 # 1 for payer, -1 for receiver
if omega == 1:
    label_side = 'Payer'
else:
    label_side = 'Receiver'
# irs contract generator
def build_contract_irs(t_tnr,strike,omega):
    ts_fixed = time_grids(0.0,t_tnr,freq=2) # semi-annually
    ts_float = time_grids(0.0,t_tnr,freq=4) # quarterly
    contract = {'grids_fixed':ts_fixed,'grids_float':ts_float,'strike':X,'side': omega}
    return contract
# test trade
contract_irs = build_contract_irs(t_tnr,X,omega)

#%% --------------------------------------------------
# run g2++ simulation
# ----------------------------------------------------
# settings
num_mc = 100
horizon = 10.0
# run simulation
# contract is only used to add simulation grids
x1s, x2s, ts_new = monte_carlo_g2pp(num_mc,horizon,params_new,contract_irs)

#%% --------------------------------------------------
# exposure calculation
# ----------------------------------------------------
num_grid_new = len(ts_new)
lst_res_mc = []
# exposure simulation
xt_last, yt_last = x1s[:,0], x2s[:,0]
for i in range(num_grid_new-1): # up to just before last grid
    if np.min(np.abs(ts_new[i]-contract_irs['grids_float'])) < 1.0e-6: # fixing grid for float
        x1_last, x2_last = x1s[:,i], x2s[:,i]
    lst_res_mc.append(pv_irs_g2pp(ts_new[i],contract_irs,params_new,x1s[:,i],x2s[:,i],x1_last,x2_last))
# result aggregation
res_mc = np.stack(lst_res_mc,axis=0).T
epe_mc = res_mc * (res_mc>0.0)
ene_mc = res_mc * (res_mc<0.0)

#%% --------------------------------------------------
# exposure sample
# ----------------------------------------------------
plt.figure()
for i in range(7):
    plt.plot(ts_new[:-1],res_mc[i,:],label='Exposure '+str(i+1))
plt.title('Exposure profile samples for '+label_side+' IRS '+str(int(t_tnr))+'yrs')
plt.xlabel('Time')
plt.ylabel('Exposure')
plt.legend()
plt.show()

#%% --------------------------------------------------
# exposure plot
# ----------------------------------------------------
plt.figure()
plt.plot(ts_new[:-1],epe_mc.mean(axis=0),label='EPE')
plt.plot(ts_new[:-1],ene_mc.mean(axis=0),label='ENE')
plt.plot(ts_new[:-1],res_mc.mean(axis=0),label='EE(Mean)')
plt.title('Exposure profile for '+label_side+' IRS '+str(int(t_tnr))+'yrs')
plt.xlabel('Time')
plt.ylabel('Exposure')
plt.legend()
plt.show()

#%% --------------------------------------------------
# scatter plot
# ----------------------------------------------------
t_train = []
e_train = epe_mc.flatten('F')
for i in range(num_grid_new-1):
    t_train = t_train + [ts_new[i]]*num_mc

#%% figure
plt.figure(figsize=(8,8))
plt.scatter(t_train,e_train,alpha=0.5,s=1.0)
plt.show()

# %%


