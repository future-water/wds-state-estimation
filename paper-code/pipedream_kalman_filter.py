import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wntr
import scipy as sc
import networkx as nx
import networkx.drawing.nx_pylab as nxp
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation
from pipedream_solver.nutils import interpolate_sample, _kalman_semi_implicit
import random
import time
import pickle
import pipedream_utility as pdu
from pipedream_utility import *
import pipedream_simulation as pd_sim
from pipedream_simulation import *
import pipedream_simulation_sensor_results as pd_sim_sensor
from pipedream_simulation_sensor_results import *

#Don't show future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def kalman_filter(model, Z, H, C, Qcov, Rcov, P_x_k_k,
                  dt, **kwargs):
    """
    Apply Kalman Filter to fuse observed data into model.

    Inputs:
    -------
    Z : np.ndarray (b x 1)
        Observed data
    H : np.ndarray (M x b)
        Observation matrix
    C : np.ndarray (a x M)
        Signal-input matrix
    Qcov : np.ndarray (M x M)
        Process noise covariance
    Rcov : np.ndarray (M x M)
        Measurement noise covariance
    P_x_k_k : np.ndarray (M x M)
        Posterior error covariance estimate at previous timestep
    dt : float
        Timestep (seconds)
    """
    A_1, A_2, b = model._semi_implicit_system(_dt=dt)
    b_hat, P_x_k_k = _kalman_semi_implicit(Z, P_x_k_k, A_1, A_2, b, H, C,
                                           Qcov, Rcov)
    model.b = b_hat
    model.iter_count -= 1
    model.t -= dt
    model._solve_step(dt=dt, **kwargs)
    return P_x_k_k

def apply_EKF(inp, sensors, t_run = 24, dt = 3600, Rcov = None, Rcov_case = 2, Qcov = None, banded = False, num_iter=10, use_tank_init_cond=False, sensor_std_dev=0.1):
    
    wn = wntr.network.WaterNetworkModel(inp)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    
    if t_run == None:
        t_run=int(wn.options.time.duration/wn.options.time.hydraulic_timestep)+1

    H_df_model, Q_df_model, Q_pump_model, model, Q_in_all_df_model, _ = run_pipedream_simulation(inp, t_run, dt, banded = banded)
        
    H_df_real, Q_df_real, Q_pump_real, model_real, H_df_sensor, Q_in_all_df_real = run_pipedream_simulation_sensor(inp, t_run=t_run, dt=dt, 
                                                                                                                   banded=banded, Rcov=Rcov, Qcov=Qcov, 
                                                                                                                   sensor_std_dev=sensor_std_dev)
    
    superjunctions, superlinks, orifices, pumps, H_bc, Q_in, pats, mult_df, tank_min, tank_max, tank_dict, time_controls_compiled, events_controls_pairs = pdu.wntr_2_pd(wn, t_run, dt)
    
    sensor_inds=[superjunctions['name'].to_list().index(s) for s in sensors]
    sensor_rows=list(np.arange(0,len(sensors)))
    
    # Set up Kalman filtering parameters
    n = wn.num_nodes + wn.num_tanks # we are adding the "fake" nodes connected to tank orifices
    p = n
    m = len(sensors)
    
    process_std_dev = .1
    measurement_std_dev = .1
    neg_dem_nodes  = [x for x in range(len(Q_in)) if Q_in[x] > 0]
    neg_list = []
    for id in neg_dem_nodes:
        neg_list.append(list(model.superjunctions.loc[model.superjunctions['id']==id,'name'])[0])

    if Rcov is None:
        if Rcov_case == 1:
            Rcov = [0.00001]*np.eye(m)
        
        elif Rcov_case == 2:
            Rcov = [sensor_std_dev**2]*np.eye(m)
            
        elif Rcov_case == 3:
            Rcov = np.square(np.array(np.std(H_df_model[sensors])))*np.eye(m)
            
        elif Rcov_case == 4:
            Rcov = [0.5**2]*np.eye(m)
            for node in sensors:
                # print(node)
                if node in wn.tank_name_list:
                    # print('tank!!!!')
                    Rcov[sensors.index(node)][sensors.index(node)] = 0.00001
                if node in neg_list:
                    # print('neg dem node!!!!')
                    Rcov[sensors.index(node)][sensors.index(node)] = 0.00001
                    
        else:
            Rcov = np.square(np.array(np.std(H_df_model[sensors])))*np.eye(m)
            for node in sensors:
                if node in wn.tank_name_list:
                    Rcov[sensors.index(node)][sensors.index(node)] = 0.00001
                if node in neg_list:
                    Rcov[sensors.index(node)][sensors.index(node)] = 0.00001
    
    if Qcov is None:
        Qcov = np.square(np.array(np.std(Q_in_all_df_real)))*np.eye(p)        

    #%% Implement Kalman Filter
                
    #Enter the sensor locations
    H = np.zeros((m, n))
    H[sensor_rows, sensor_inds] = 1.
    
    C = np.zeros((n, p))
    C[np.arange(n), np.arange(p)] = 1.
    
    P_x_k_k = C @ Qcov @ C.T
    
    measurements = H_df_sensor.iloc[:,sensor_inds]
    measurements.index += dt
    # Set initial, end, minimum and maximum timesteps
    
    t_end = np.max(H_df_model.index)
    
    mult_df['-'] = 0.
    multipliers = mult_df
    
    #%% Run Model- Baseline
    
    # Specify number of internal links in each superlink and timestep size in seconds
    internal_links = 1
    num_tanks=wn.num_tanks
    u_o = np.zeros(num_tanks)
    u_p = np.ones(wn.num_pumps)
    
    model = SuperLink(superlinks, superjunctions,  
                      internal_links=internal_links, orifices=orifices, pumps=pumps, auto_permute=banded)
    
    is_tank = model.superjunctions['tank'].values
    tank_min = model.superjunctions['tank_min'].values
    tank_max = model.superjunctions['tank_max'].values 
    
    Q_in_t = -(model.superjunctions['demand_pattern'].map(multipliers.loc[0]).fillna(0.) * model.superjunctions['base_demand']).values
    model.spinup(n_steps=10, dt=60, Q_in=Q_in_t, H_bc=H_bc, u_o=u_o, u_p=u_p, banded=banded, num_iter=num_iter, head_tol=0.0001)
    
    Hj = []
    Q = []
    Q_pump = []
    Q_in_all = []
    t=[]
        
    #Run model for 24 hours
    # While time is less than 24 hours
    while model.t < (t_run * 3600):    
        
        hour=model.t/3600
        j=int(np.floor(model.t/3600))
        if 'Net1' in inp:
            j = int(np.floor(model.t/3600)) //2
        Q_in_t = -(model.superjunctions['demand_pattern'].map(multipliers.loc[j]).fillna(0.) * model.superjunctions['base_demand']).values
        Q_in_all.append(Q_in_t)
        H_bc_t = H_bc.copy()
        
        # Set tank initial conditions
        if use_tank_init_cond:
            if model.t == 0:
                H_bc_t[is_tank] = model.superjunctions['h_0'].values[is_tank] + model.superjunctions['z_inv'].values[is_tank]
                model.bc[is_tank] = True
                H_bc_t_0=H_bc_t
            else:
                model.bc[is_tank] = False
        
        u_o = ((model.H_j[is_tank] > tank_min[is_tank]) & (model.H_j[is_tank] < tank_max[is_tank])).astype(np.float64)
#        # Check control rule status
#        # open link --> 1, close link --> 0
                
        # Event based controls -- only assuming pump - tank control rules for here. Modify for NWC
        # this is also for > upper limit
        
        for key in events_controls_pairs.keys():
            node = events_controls_pairs[key]['Node']
            link = events_controls_pairs[key]['Link']
            node_id = list(model.superjunctions.loc[model.superjunctions['name']==node,'id'])[0]
            if link in wn.pump_name_list:
                pump_id = model.pumps.loc[model.pumps['name']==link,'id'].values
                
            if model.H_j[node_id] > events_controls_pairs[key]['Upper lim']:
                u_p[pump_id] = events_controls_pairs[key]['Upper lim stat']
            if model.H_j[node_id] < events_controls_pairs[key]['Lower lim']:
                u_p[pump_id] = events_controls_pairs[key]['Lower lim stat']
                    
        #Run model
        model.step(dt=dt, H_bc = H_bc_t, Q_in=Q_in_t, u_o=u_o, u_p=u_p, 
                   banded=banded, num_iter=num_iter, head_tol=0.00001) # initial conditions
        next_measurement = measurements.loc[model.t].values
        #next_measurement = interpolate_sample(model.t,
        #                                      measurements.index.values,
        #                                      measurements.values)
        
        P_x_k_k = kalman_filter(model, next_measurement, H, C, Qcov, Rcov, P_x_k_k,
                                dt, u_o=u_o, banded=banded)
        #Extract results at each timestep
        Hj.append(model.H_j.copy())
        Q.append(model.Q_ik.copy())
        Q_pump.append(model.Q_p)
        t.append(model.t)
        
    Hj = np.vstack(Hj)
    Q = np.vstack(Q)
    Q_pump = np.vstack(Q_pump)
    t=np.vstack(t)
        
    #Sample down the Q matrix to only every column from a new link, not each sub-link
    #i.e. if there are 12 internal links in each superlink, then each of the first 
    #12 columns in q will basically be the same
    
    n_superlinks,x=superlinks.shape
    Q_superlinks=Q[:,0:n_superlinks*internal_links:internal_links]
    
    #put H and Q into a dataframe
    #Unscramble the head matrix
    perm_inv = np.argsort(model.permutations)
    Hj = Hj[:, perm_inv]
    
    #Do not use model.superjunctions because I want to use the head matrix in the same order
    #as the original (unpermuted) superjunctions DF because then the columns correspond
    #to the wntr results
    H_df_filtered = pd.DataFrame(columns=superjunctions['name'],index=np.arange(0,t_run*3600,dt),data=Hj)
    
    return H_df_filtered, H_df_real, H_df_model, H_df_sensor, model
       