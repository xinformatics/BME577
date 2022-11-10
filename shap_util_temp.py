import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as K
import shap
from copy import deepcopy
#from timeshap.plot import plot_temp_coalition_pruning, plot_event_heatmap, plot_feat_barplot, plot_cell_level
from timeshap.explainer import local_pruning, local_event, local_feat, local_cell_level
from timeshap.utils import calc_avg_event
import warnings
from util import *
from preprocess import *
from model import *
warnings.filterwarnings('ignore')

class StationaryTimeWindow():
    '''
    A class for computing the shapely values for time sereis data. Only the shap values for the first output
    is reported.
    '''
    def __init__(self, model, window_len, B_ts, test_ts, B_mask=None, B_dem=None,
                 test_mask=None, test_dem=None, model_type='grud_dem'):
        self.model = model
        self.window_len = window_len
        self.num_window = np.ceil(B_ts.shape[1]/self.window_len).astype('int')
        self.num_background = len(B_ts)
        self.num_test = len(test_ts)
        self.background_ts = B_ts
        self.background_mask = B_mask
        self.background_dem = B_dem
        self.test_ts = test_ts
        self.test_mask = test_mask
        self.test_dem = test_dem
        self.model_type = model_type
        self.ts_phi = None
        self.dem_phi = None
        self.explainer = None
        
        # Problem sizes
        self.num_ts_ftr = B_ts.shape[2]
        self.num_ts_step = B_ts.shape[1]
        self.num_dem_ftr = 0 if B_dem is None else B_dem.shape[1]
        
        # Creating all data (background and test together)
        self.all_ts = np.concatenate((self.background_ts, self.test_ts), axis=0)
        self.all_mask = None if test_mask is None else np.concatenate((self.background_mask, self.test_mask), axis=0)
        self.all_dem = None if test_dem is None else np.concatenate((self.background_dem, self.test_dem), axis=0)
        
        # Creating converted data for SHAP
        self.background_data = self.data_prepare(ts_x=self.background_ts, dem_x=self.background_dem, start_idx=0)
        self.test_data = self.data_prepare(ts_x=self.test_ts, dem_x=self.test_dem, start_idx=self.num_background)
        
        
    def data_prepare(self, ts_x, dem_x=None, start_idx=0):
        assert len(ts_x.shape) == 3
        assert dem_x is None or len(dem_x.shape) == 2
        dem_len = 0 if dem_x is None else dem_x.shape[1]

        total_num_features = self.num_dem_ftr + self.num_ts_ftr * self.num_window

        x_ = [[i]*total_num_features for i in range(start_idx, start_idx + ts_x.shape[0])]

        return np.array(x_)
        
    
    def wraper_predict(self, x):
        assert len(x.shape) == 2

        dem_x, ts_x = x[:, :self.num_dem_ftr].copy(), x[:, self.num_dem_ftr:].copy()

        # initializing the value of all arrays
        ts_x_ = np.zeros((x.shape[0], self.all_ts.shape[1], self.all_ts.shape[2]))
        mask_x_ = np.zeros_like(ts_x_)
        dem_x_ = np.zeros_like(dem_x, dtype=float)
        tstep = np.ones((x.shape[0], self.all_ts.shape[1], 1)) * \
                    np.reshape(np.arange(0, self.all_ts.shape[1]), (1, self.all_ts.shape[1], 1))

        # Reshaping the ts indices based on the num time windows and features
        ts_x = ts_x.reshape((ts_x.shape[0], self.num_window, self.num_ts_ftr))

        for i in range(x.shape[0]):
            # creating time series data
            for t in range(self.num_ts_step):
                for j in range(self.num_ts_ftr):
                    # Finding the corresponding time interval
                    wind_t = np.ceil((t+1)/self.window_len).astype('int') - 1
                    ind = ts_x[i, wind_t, j]
                    ts_x_[i, t, j] = self.all_ts[ind, t, j]
                    mask_x_[i, t, j] = None if self.all_mask is None else self.all_mask[ind, t, j]
            # creating static data
            for j in range(dem_x.shape[1]):
                ind = dem_x[i,j]
                dem_x_[i, j] = None if self.all_dem is None else self.all_dem[ind, j]

        # Creating the input of the model based on the different models. 
        # This part should be updated as new models get involved in the project
        if self.model_type == 'grud_dem':
            model_input = [ts_x_, mask_x_, tstep, dem_x_]
        elif self.model_type == 'grud':
            model_input = [ts_x_, mask_x_, tstep]
        elif self.model_type == 'lstm':
            model_input = ts_x_
        
        return self.model.predict(model_input)
    
    def shap_values(self, num_output=1):
        self.explainer = shap.KernelExplainer(self.wraper_predict, self.background_data)
        shap_values = self.explainer.shap_values(self.test_data)
        shap_values = np.array(shap_values)
        
        self.dem_phi = shap_values[:, :, :self.num_dem_ftr]
        ts_shap_values = shap_values[:, :, self.num_dem_ftr:]
        self.ts_phi = ts_shap_values.reshape((num_output, self.num_test, self.num_window, self.num_ts_ftr))
        
        # assign values to each single time step by deviding the values by window length
        self.ts_phi = np.repeat(self.ts_phi/self.window_len, self.window_len, axis=2)[:,:,:self.num_ts_step,:]
        
        # Reporting only the first output
        self.ts_phi = self.ts_phi[0]
        self.dem_phi = self.dem_phi[0]

        return self.ts_phi if self.num_dem_ftr==0 else (self.dem_phi, self.ts_phi)
    

class SlidingTimeWindow():
    def __init__(self, model, stride, window_len, B_ts, test_ts, B_mask=None,
                 B_dem=None, test_mask=None, test_dem=None, model_type='grud_dem'):
        self.model = model
        self.model_type = model_type
        self.stride = stride
        self.window_len = window_len
        self.num_window = 2 #Specific to the sliding time window
        self.num_background = len(B_ts)
        self.num_test = len(test_ts)
        self.background_ts = B_ts
        self.background_mask = B_mask
        self.background_dem = B_dem
        self.test_ts = test_ts
        self.test_mask = test_mask
        self.test_dem = test_dem
        self.ts_phi = None
        self.dem_phi = None
        self.explainer = None
        
        # Problem sizes
        self.num_ts_ftr = B_ts.shape[2]
        self.num_ts_step = B_ts.shape[1]
        self.num_dem_ftr = 0 if B_dem is None else B_dem.shape[1]
        
        
        # Creating all data (background and test together)
        self.all_ts = np.concatenate((self.background_ts, self.test_ts), axis=0)
        self.all_mask = None if test_mask is None else np.concatenate((self.background_mask, self.test_mask), axis=0)
        self.all_dem = None if test_dem is None else np.concatenate((self.background_dem, self.test_dem), axis=0)
        
        # Creating converted data for SHAP
        self.background_data = self.data_prepare(ts_x=self.background_ts, dem_x=self.background_dem, start_idx=0)
        self.test_data = self.data_prepare(ts_x=self.test_ts, dem_x=self.test_dem, start_idx=self.num_background)
    
    def data_prepare(self, ts_x, dem_x=None, start_idx=0):
        # Modified for sliding time window
        assert len(ts_x.shape) == 3
        assert dem_x is None or len(dem_x.shape) == 2

        total_num_features = self.num_dem_ftr + self.num_ts_ftr * self.num_window

        x_ = [[i] * total_num_features for i in range(start_idx, start_idx + ts_x.shape[0])]

        return np.array(x_)
    
    def wraper_predict(self, x, start_ind=0):
        assert len(x.shape) == 2
        
        # Calculating the indices inside the time window
        inside_ind = list(range(start_ind, start_ind + self.window_len))
        
        dem_x, ts_x = x[:, :self.num_dem_ftr].copy(), x[:, self.num_dem_ftr:].copy()

        # initializing the value of all arrays
        ts_x_ = np.zeros((x.shape[0], self.num_ts_step, self.num_ts_ftr))
        mask_x_ = np.zeros_like(ts_x_)
        dem_x_ = np.zeros_like(dem_x, dtype=float)
        tstep = np.ones((x.shape[0], self.num_ts_step, 1)) * \
                    np.reshape(np.arange(0, self.num_ts_step), (1, self.num_ts_step, 1))

        # Reshaping the ts indices based on the num time windows and features
        ts_x = ts_x.reshape((ts_x.shape[0], self.num_window, self.num_ts_ftr))

        for i in range(x.shape[0]):
            # creating time series data
            for t in range(self.num_ts_step):
                for j in range(self.num_ts_ftr):
                    # Finding the corresponding time interval
                    wind_t = 0 if (t in inside_ind) else 1
                    ind = ts_x[i, wind_t, j]
                    ts_x_[i, t, j] = self.all_ts[ind, t, j]
                    mask_x_[i, t, j] = None if self.all_mask is None else self.all_mask[ind, t, j]
            # creating static data
            for j in range(dem_x.shape[1]):
                ind = dem_x[i,j]
                dem_x_[i, j] = None if self.all_dem is None else self.all_dem[ind, j]
        
        # Creating the input of the model based on the different models. 
        # This part should be updated as new models get involved in the project
        if self.model_type == 'grud_dem':
            model_input = [ts_x_, mask_x_, tstep, dem_x_]
        elif self.model_type == 'grud':
            model_input = [ts_x_, mask_x_, tstep]
        elif self.model_type == 'lstm':
            model_input = ts_x_
        
        return self.model.predict(model_input)
    
    def shap_values(self, num_output=1, nsamples='auto'):
        # Initializing number of time windows and contribution score matrices
        seq_len = self.background_ts.shape[1]
        num_sw = np.ceil((seq_len - self.window_len)/self.stride).astype('int') + 1
        ts_phi = np.zeros((self.num_test, num_sw, 2, self.background_ts.shape[2]))
        dem_phi = np.zeros((self.num_test, num_sw, self.num_dem_ftr))
        
        # Determining the number of samples
        if nsamples=='auto':
            nsamples = 10 * self.num_ts_ftr + 5 * self.num_dem_ftr
        
        # Main loop on different possible windows
        for stride_cnt in range(num_sw):
    
            predict = lambda x: self.wraper_predict(x, start_ind=stride_cnt * self.stride)

            # Running SHAP
            self.explainer = shap.KernelExplainer(predict, self.background_data)
            shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples)
            shap_values = np.array(shap_values)

            # Extracting the SHAP values and storing them
            dem_shap_values_ = shap_values[:, :, :self.num_dem_ftr]
            ts_shap_values = shap_values[:, :, self.num_dem_ftr:]
            ts_shap_values = ts_shap_values.reshape((num_output, self.num_test, 2, self.num_ts_ftr))

            ts_phi[:, stride_cnt, :, :] = ts_shap_values[0]
            dem_phi[:, stride_cnt, :] = dem_shap_values_[0]
            
        # Averaging shap values from different windows
        ts_phi_agg = np.empty((self.num_test, num_sw, self.num_ts_step, self.num_ts_ftr))
        ts_phi_agg[:] = np.nan
        for k in range(num_sw):
            ts_phi_agg[:,k, k * self.stride:k * self.stride + self.window_len, :] = ts_phi[:, k, 0, :][:, np.newaxis, :]
        ts_phi_agg = np.nanmean(ts_phi_agg, axis=1)
        dem_phi = np.nanmean(dem_phi, axis=1)
        
        self.dem_phi = dem_phi
        self.ts_phi = ts_phi_agg
        
        return ts_phi_agg if self.num_dem_ftr==0 else (dem_phi, ts_phi_agg)


class BinaryTimeWindow():
    def __init__(self, model, delta, n_w, B_ts, test_ts, B_mask=None, B_dem=None, test_mask=None, test_dem=None, model_type='grud_dem'):
        self.model = model
        self.model_type = model_type
        self.num_background = len(B_ts)
        self.num_test = len(test_ts)
        self.background_ts = B_ts
        self.background_mask = B_mask
        self.background_dem = B_dem
        self.test_ts = test_ts
        self.test_mask = test_mask
        self.test_dem = test_dem
        self.ts_phi = None
        self.dem_phi = None
        self.explainer = None
        
        # Problem sizes
        self.num_ts_ftr = B_ts.shape[2]
        self.num_ts_step = B_ts.shape[1]
        self.num_dem_ftr = 0 if B_dem is None else B_dem.shape[1]
        
        ## Specific to Binary Time Window
        assert self.num_test == 1 # For binary time window algorithm, samples should be fed to the algorithm one-by-one
        self.delta = delta
        self.n_w = n_w
        self.split_points = [[self.num_ts_step - 1]] * self.num_ts_ftr # Splitting points
        self.num_window = [1] * self.num_ts_ftr
        
        
        # Creating all data (background and test together)
        self.all_ts = np.concatenate((self.background_ts, self.test_ts), axis=0)
        self.all_mask = None if test_mask is None else np.concatenate((self.background_mask, self.test_mask), axis=0)
        self.all_dem = None if test_dem is None else np.concatenate((self.background_dem, self.test_dem), axis=0)
        
        # Creating converted data for SHAP
        self.background_data = self.data_prepare(ts_x=self.background_ts, dem_x=self.background_dem, start_idx=0)
        self.test_data = self.data_prepare(ts_x=self.test_ts, dem_x=self.test_dem, start_idx=self.num_background)
    
    def data_prepare(self, ts_x, dem_x=None, start_idx=0):
        assert len(ts_x.shape) == 3
        assert dem_x is None or len(dem_x.shape) == 2
        total_num_features = self.num_dem_ftr + sum(self.num_window) ## Specific to Binary Time Window
        
        x_ = [[i] * total_num_features for i in range(start_idx, start_idx + ts_x.shape[0])]

        return np.array(x_)
    
    def wraper_predict(self, x):
        assert len(x.shape) == 2
        
        dem_x, ts_x = x[:, :self.num_dem_ftr].copy(), x[:, self.num_dem_ftr:].copy()

        # initializing the value of all arrays
        ts_x_ = np.zeros((x.shape[0], self.num_ts_step, self.num_ts_ftr))
        mask_x_ = np.zeros_like(ts_x_)
        dem_x_ = np.zeros_like(dem_x, dtype=float)
        tstep = np.ones((x.shape[0], self.num_ts_step, 1)) * \
                    np.reshape(np.arange(0, self.num_ts_step), (1, self.num_ts_step, 1))

        # Reshaping the ts indices based on the time windows for each feature
        ## Specific to Binary Time Window
        temp_ts_x = np.zeros((ts_x.shape[0], max(self.num_window), self.num_ts_ftr), dtype=int)
        for i in range(self.num_ts_ftr):
            temp_ts_x[:, :self.num_window[i], i] = ts_x[:, sum(self.num_window[:i]):sum(self.num_window[:i+1])]
        ts_x = temp_ts_x

        for i in range(x.shape[0]):
            # creating time series data
            for j in range(self.num_ts_ftr):
                # Finding the corresponding time interval
                wind_t = np.searchsorted(self.split_points[j], np.arange(self.num_ts_step)) ## Specific to Binary Time Window
                for t in range(self.num_ts_step):
                    ind = ts_x[i, wind_t[t], j]
                    ts_x_[i, t, j] = self.all_ts[ind, t, j]
                    mask_x_[i, t, j] = None if self.all_mask is None else self.all_mask[ind, t, j]
            # creating static data
            for j in range(dem_x.shape[1]):
                ind = dem_x[i,j]
                dem_x_[i, j] = None if self.all_dem is None else self.all_dem[ind, j]
        
        # Creating the input of the model based on the different models. 
        # This part should be updated as new models get involved in the project
        if self.model_type == 'grud_dem':
            model_input = [ts_x_, mask_x_, tstep, dem_x_]
        elif self.model_type == 'grud':
            model_input = [ts_x_, mask_x_, tstep]
        elif self.model_type == 'lstm':
            model_input = ts_x_
        
        return self.model.predict(model_input)
    
    def shap_values(self, num_output=1, nsamples_in_loop='auto', nsamples_final='auto'):
        flag = 1
        while flag:
            flag = 0
            # Updating the number of time windows for each time series feature
            self.num_window = [len(self.split_points[i]) for i in range(self.num_ts_ftr)]
            
            # Updating converted data for SHAP
            self.background_data = self.data_prepare(ts_x=self.background_ts, dem_x=self.background_dem, start_idx=0)
            self.test_data = self.data_prepare(ts_x=self.test_ts, dem_x=self.test_dem, start_idx=self.num_background)

            # Running SHAP
            if nsamples_in_loop == 'auto':
                nsamples = 2 * sum(self.num_window)
            else:
                nsamples = nsamples_in_loop
            
            self.explainer = shap.KernelExplainer(self.wraper_predict, self.background_data)
            shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples)
            shap_values = np.array(shap_values)
            dem_phi = shap_values[0, :, :self.num_dem_ftr] # Extracting dem SHAP values
            ts_shap_values = shap_values[:, :, self.num_dem_ftr:] # Extracting ts SHAP values
            
            # Checking the maximum number of windows condition
            if max(self.num_window) >= self.n_w: break
            
            for i in range(self.num_ts_ftr):
                S = set(self.split_points[i])
                for j in range(self.num_window[i]):
                    if abs(ts_shap_values[0, 0, sum(self.num_window[:i]) + j]) > self.delta:
                        S.add(int(self.split_points[i][j]/2) if j == 0 else int((self.split_points[i][j-1] + self.split_points[i][j])/2))
                if set(S) != set(self.split_points[i]):
                    flag += 1
                    self.split_points[i] = list(S)
                    self.split_points[i].sort()
        
        # Running SHAP with large number of samples for the final evaluation of Shapely values
        self.explainer = shap.KernelExplainer(self.wraper_predict, self.background_data)
        shap_values = self.explainer.shap_values(self.test_data, nsamples=nsamples_final)
        shap_values = np.array(shap_values)
        dem_phi = shap_values[0, :, :self.num_dem_ftr] # Extracting dem SHAP values
        ts_shap_values = shap_values[:, :, self.num_dem_ftr:] # Extracting ts SHAP values
        
        # Assigning Shap values to each single time step
        ts_phi = np.zeros((self.num_test, self.num_ts_step, self.num_ts_ftr))
        for i in range(self.num_ts_ftr):
            for j in range(self.num_window[i]):
                # This part of the code is written in a way that each splitting point belongs to the time window that starts from that point
                # For the last time window, both splitting points at the end and start of the time window belong to it
                start_ind = 0 if j==0 else self.split_points[i][j-1]
                end_ind = self.split_points[i][j] + int((j + 1) / self.num_window[i])
                ts_phi[0, start_ind:end_ind, i] = ts_shap_values[0, :, sum(self.num_window[:i]) + j] / (end_ind - start_ind)
        self.dem_phi = dem_phi
        self.ts_phi = ts_phi
        
        return ts_phi if self.num_dem_ftr==0 else (dem_phi, ts_phi)
    

def xai_eval_fnc(model, relevence, input_x, model_type='grud', percentile=90,
                 eval_type='prtb', seq_len=10, by='all'):
    
    input_new = deepcopy(input_x)
    relevence = np.absolute(relevence)
    
    # TO DO: Add other type of models
    if model_type == 'grud':
        input_ts = input_x[0]
        input_new_ts = input_new[0]
    elif model_type == 'lstm':
        input_ts = input_x
        input_new_ts = input_new
    
    assert len(input_ts.shape)==3 # the time sereis data needs to be 3-dimensional
    num_feature = input_ts.shape[2]
    num_time_step = input_ts.shape[1]
    num_instance = input_ts.shape[0]
        
    if by=='time':
        top_steps = math.ceil((1 - percentile/100) * num_time_step) # finding the number of top steps for each feature
        top_indices = np.argsort(relevence, axis=1)[:, -top_steps:, :] # a 3d array of top time steps for each feature
        for j in range(num_feature): # converting the indices to a flatten version
            top_indices[:, :, j] = top_indices[:, :, j] * num_feature + j
        top_indices = top_indices.flatten()
    elif by=='all':
        top_steps = math.ceil((1 - percentile/100) * num_time_step * num_feature) # finding the number of all top steps
        top_indices = np.argsort(relevence, axis=None)[-top_steps:]
    
    # Create a masking matrix for top time steps
    top_indices_mask = np.zeros(input_ts.size)
    top_indices_mask[top_indices] = 1
    top_indices_mask = top_indices_mask.reshape(input_ts.shape)
    
    
    # Evaluating different metrics
    for p in range(num_instance):
        for v in range(num_feature):
            for t in range(num_time_step):
                if top_indices_mask[p, t, v]:
                    if eval_type == 'prtb':
                        input_new_ts[p,t,v] = np.max(input_ts[p,:,v]) - input_ts[p,t,v]
                    elif eval_type == 'sqnc_eval':
                        input_new_ts[p, t:t + seq_len, v] = 0
    
    return model.predict(input_new)

def timeshap_to_array(f, train_x, test_x, test_ts, var_to_ind=None, nsample='auto',
                      tol=0.001, top_x_events=5, top_x_feats=5):
    if var_to_ind is None:
        var_to_ind = {}
        for i in range(test_ts.shape[2]):
            var_to_ind['Column' + str(i)] = i
    input_vars = list(var_to_ind)
    
    # Calculate timeSHAP
    average_event = np.median(np.concatenate((train_x, test_x), axis=0), axis=(0,1))
    average_event = pd.DataFrame([average_event], columns=None)
    pruning_dict = {'tol': tol,}
    coal_plot_data, coal_prun_idx = local_pruning(f, test_ts, pruning_dict, average_event, verbose=False)
    pruning_idx = test_ts.shape[1] + coal_prun_idx

    event_dict = {'rs': 42, 'nsamples': nsample}
    event_data = local_event(f, test_ts, event_dict, baseline=average_event, 
                             pruned_idx=pruning_idx, entity_col=None, entity_uuid=None)

    feature_dict = {'rs': 42, 'nsamples': nsample, 'feature_names': input_vars, 'plot_features':None}
    feature_data = local_feat(f, test_ts, feature_dict, baseline=average_event,
                              pruned_idx=pruning_idx, entity_col=None, entity_uuid=None)

    cell_dict = {'rs': 42, 'nsamples': nsample, 'top_x_events': top_x_events, 'top_x_feats': top_x_feats}
    cell_data = local_cell_level(f, test_ts, cell_dict, event_data, feature_data, baseline=average_event,
                                 pruned_idx=pruning_idx, entity_col=None, entity_uuid=None)
    
    # Calculate the final array like
    timeshap_ts_phi = np.zeros((test_ts.shape[1], test_ts.shape[2]))
    listed_features = list(cell_data.Feature.unique())
    other_features = list(set(input_vars) - set(listed_features))
    listed_events = list(cell_data.Event.unique())
    listed_events = [test_ts.shape[1] + int(i[6:]) for i in listed_events if i != 'Pruned Events' and i != 'Other Events']
    pruned_events = list(range(0, test_ts.shape[1] + coal_prun_idx))
    other_events = list(set(range(0, test_ts.shape[1])) - (set(listed_events + pruned_events)))
    for row in cell_data.itertuples():
        shap_col = row._3
        if row.Event != 'Pruned Events' and row.Event != 'Other Events':
            i = int(row.Event[6:])
            i = test_ts.shape[1] + i
            if row.Feature != 'Other Features':
                j = var_to_ind[row.Feature]
                timeshap_ts_phi[i,j] = shap_col
            else:
                for feature in other_features:
                    j = var_to_ind[feature]
                    timeshap_ts_phi[i,j] = shap_col/len(other_features)
        elif row.Event == 'Pruned Events':
            for i in pruned_events:
                for j in input_vars:
                    j = var_to_ind[j]
                    timeshap_ts_phi[i,j] = shap_col/(len(pruned_events)*len(input_vars))
        elif row.Event == 'Other Events':
            if row.Feature != 'Other Features':
                for i in other_events:
                    j = var_to_ind[row.Feature]
                    timeshap_ts_phi[i,j] = shap_col/len(other_events)
            else:
                for i in other_events:
                    for j in other_features:
                        j = var_to_ind[j]
                        timeshap_ts_phi[i,j] = shap_col/(len(other_events)*len(other_features))
    timeshap_ts_phi = timeshap_ts_phi[np.newaxis, :, :]
    return timeshap_ts_phi

if __name__=="__main__":
    c = Connection(verbose=False)
    c.clean_clinic_data(miss_rate=0.2, max_gcs=15)
    c.clean_gcs_data()
    c.clean_vital_data()
    c.clean_lab_data()
    # Variable names----------
    _, _, var_to_ind = ts_builder_3d(c, masking=False)
    var = list(var_to_ind.keys())
    # ------------------------
    K.clear_session()
    rnn = RNNSimple(c, output_type='Binary', imput_method='multivar', max_len=120, num_hidden=35,
                    num_neurons=62, epochs=13, total_fold=5, current_fold=2,
                    drop_rates=[0.421, 0.584, 0.297, 0.297], reg_rate=0.208, es_patience=49, learning_rate=0.00024)