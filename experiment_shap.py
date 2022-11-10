# SET RANDOM SEEDS-------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------

# Apparently you may use different seed values at each stage
seed_value= 0

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------
# ----------------------------------------------------------------


import timeit
import sys
import argparse


from memory_profiler import memory_usage
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, regularizers
from tensorflow.keras import backend as K


from util import *
sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", "track_project"))
from shap_util import *


def main(exp_id, test_id, cv):
    model_type = 'lstm'
    exp_df = pd.read_csv(DATA_PATH + 'Shap Experiments/experiment features.csv')
    exp_df = exp_df[exp_df.id==exp_id]
    
    print('Experiment ID = {}, Test ID = {}, and CV= {}'.format(exp_id, test_id, cv))
    print(exp_df)

    

    ### DATA PREPARATION and MODEL--------------------
    with (open(MIMIC_DATA_PATH + 'patient_vital_preprocessed.pkl', "rb")) as obj:
        mimic_data = pickle.load(obj)
    value = np.zeros((len(mimic_data), mimic_data[0][0].shape[0], mimic_data[0][0].shape[1]))*np.nan
    output = np.zeros(len(mimic_data))*np.nan
    for i, data in enumerate(mimic_data):
        value[i] = data[0]
        output[i] = data[1]
    
    # Removing demographic features
    value = value[:,:-4,:]
    value = np.transpose(value, axes=(0,2,1))
    # Removing 8th feature since it is constant
    value = np.delete(value, 8, axis=2)
    
    # CV setting
    if cv==0:
        train_x, test_x, train_y, test_y = train_test_split(value, output,
                                                            test_size=0.2, random_state=0,
                                                            stratify=output, shuffle=True)
    else:
        kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        train_idx, test_idx = list(kf.split(X=value, y=output))[cv - 1]
        train_x, test_x = value[train_idx], value[test_idx]
        train_y, test_y = output[train_idx], output[test_idx]
    
    # Validation set
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
                                                      test_size=0.2, random_state=0,
                                                      stratify=train_y, shuffle=True)
    
    # normalizing train and test set
    train_scaler, test_scaler, val_scaler = TimeSeriesScaler(), TimeSeriesScaler(), TimeSeriesScaler()
    train_x = train_scaler.fit_transform(train_x)
    test_x = test_scaler.fit_transform(test_x)
    val_x = val_scaler.fit_transform(val_x)

    # Model
    es_patience = 100
    learning_rate = 0.0002
    reg_rate = 0.01
    epochs = 3000
    bach_size = 64
    droup_rate = 0.4

    K.clear_session()
    model = Sequential()
    model.add(Input(shape=(value.shape[1], value.shape[2])))
    model.add(GRU(70, activation='relu', dropout=0.3, recurrent_dropout=0.3))
    model.add(Dropout(droup_rate))
    model.add(Dense(40, activation='relu', kernel_regularizer=regularizers.l2(reg_rate)))
    model.add(Dropout(droup_rate))
    model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(reg_rate)))
    model.add(Dense(1, activation='sigmoid'))

    # train the model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=es_patience, restore_best_weights=True)

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy',
                  metrics=['accuracy', f1_m])
    model.summary()

    # Sample weights
    train_sample_weights, val_sample_weights = np.ones(len(train_y)), np.ones(len(val_y))
    train_sample_weights[train_y==0] = len(train_y)/(train_y==0).sum()
    train_sample_weights[train_y==1] = len(train_y)/(train_y==1).sum()

    val_sample_weights[val_y==0] = len(val_y)/(val_y==0).sum()
    val_sample_weights[val_y==1] = len(val_y)/(val_y==1).sum()
    
    train_history = model.fit(train_x, train_y, batch_size=bach_size,
                              epochs=epochs, verbose=2,
                              validation_data=(val_x, val_y, val_sample_weights),
                              sample_weight=train_sample_weights,
                              callbacks=[es])


    predicted_y = model.predict(test_x[test_id:test_id + 1])
    true_y = test_y[test_id:test_id + 1]
    
    ### SHAP Methods-------------------------------
    method = exp_df.method.iloc[0]
    num_background = int(exp_df.num_background)
    background_ts_1 = train_x[:num_background]
    test_ts_1 = test_x[test_id:test_id + 1]
    
    tic = timeit.default_timer()
    if method == 'STW':
        window_len = int(exp_df.window_len)
        stw = StationaryTimeWindow(model, window_len, B_ts=background_ts_1,
                                   test_ts=test_ts_1, model_type=model_type)
        ts_phi = stw.shap_values()
    elif method == 'SLTW':
        window_len = int(exp_df.window_len)
        stride = int(exp_df.stride)
        sltw = SlidingTimeWindow(model, stride, window_len, B_ts=background_ts_1, 
                                 test_ts=test_ts_1, model_type=model_type)
        ts_phi = sltw.shap_values()
    elif method == 'BTW':
        delta = float(exp_df.delta)
        n_w = int(exp_df.n_w)
        btw = BinaryTimeWindow(model, delta, n_w, B_ts=background_ts_1,
                               test_ts=test_ts_1, model_type=model_type)
        ts_phi = btw.shap_values()
    elif method == 'TIMESHAP':
        top_feat = int(exp_df.top_feat)
        top_event = int(exp_df.top_event)
        tol = float(exp_df.tol)
        ts_phi = timeshap_to_array(model.predict, train_x, test_x, test_ts_1,
                                   tol=tol, top_x_events=top_event, top_x_feats=top_feat)
    toc = timeit.default_timer()
    print('Total time: {}'.format(toc-tic))
    
    # Evaluation of the results
    max_p=95
    min_p=10
    max_seq_len = round(ts_phi.shape[1]/3)
    min_seq_len=1
    prtb_eval_time = np.zeros((max_p-min_p))
    seq_eval_time = np.zeros((max_p-min_p, max_seq_len-min_seq_len))
    prtb_eval_all = np.zeros((max_p-min_p))
    seq_eval_all = np.zeros((max_p-min_p, max_seq_len-min_seq_len))
    test_data = test_x[test_id:test_id + 1]
    for p in range(min_p, max_p):
        prtb_eval_time[p - min_p] = xai_eval_fnc(model, ts_phi, test_data, model_type=model_type,
                                                 eval_type='prtb', percentile=p, by='time')
        prtb_eval_all[p - min_p] = xai_eval_fnc(model, ts_phi, test_data, model_type=model_type,
                                                eval_type='prtb', percentile=p, by='all')
        for seq_len in range(min_seq_len, max_seq_len):
            seq_eval_time[p-min_p, seq_len-min_seq_len] = xai_eval_fnc(model, ts_phi, test_data,
                                                                       model_type=model_type, eval_type='sqnc_eval', 
                                                                       seq_len=seq_len, percentile=p, by='time')
            seq_eval_all[p-min_p, seq_len-min_seq_len] = xai_eval_fnc(model, ts_phi, test_data,
                                                                      model_type=model_type, eval_type='sqnc_eval', 
                                                                      seq_len=seq_len, percentile=p, by='all')
    

    ### SAVE FILE--------------------------------
    save_name = 'mimic_exp{}_test{}_cv{}'.format(exp_id, test_id, cv)
    with open(OUTPUT_PATH + save_name, 'wb') as f:
        saveObject = {'ts_phi':ts_phi,
                      'prtb_eval_time':prtb_eval_time,
                      'seq_eval_time':seq_eval_time,
                      'prtb_eval_all':prtb_eval_all,
                      'seq_eval_all':seq_eval_all,
                      'total_time':toc-tic,
                      'predicted_y': predicted_y,
                      'true_y': true_y}
        pickle.dump(saveObject, f)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shap Experiments')
    parser.add_argument('--index', type=int)
    parser.add_argument('--num_tests', type=int)
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--memory', type=bool, default=False)
    args = parser.parse_args()
    
    index = int(args.index) # Must start from 1 to total_experiments * num_tests
    num_tests = int(args.num_tests)
    cv = int(args.cv)
    
    # Calculating the exp_id and test sample id
    exp_id = int((index - 1)/num_tests) + 1
    test_id = index % num_tests
    
    if args.memory:
        print('******** Memory mode is enabled **********')
        mem = max(memory_usage((main, (exp_id, test_id, cv))))

        # Read already saved result
        save_name = 'mimic_exp{}_test{}_cv{}'.format(exp_id, test_id, cv)
        with open(OUTPUT_PATH + save_name, 'rb') as f:
            saveObject = pickle.load(f)
        # Change result file and save it again
        saveObject['memory']=mem
        with open(OUTPUT_PATH + save_name, 'wb') as f:
            pickle.dump(saveObject, f)
    else:
        main(exp_id, test_id, cv)