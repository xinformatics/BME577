import matplotlib
matplotlib.use('Agg')

FIG_PATH = 'figures/'
DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

from util import *
from preprocess import *

#from nn_utils.grud_layers import Bidirectional_for_GRUD, GRUD
from nn_utils.grud_layers import GRUD
from nn_utils.layers import ExternalMasking
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.python.keras import layers, regularizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from tqdm import tqdm
from tensorflow.python.keras.layers import Input, Dense, Concatenate, RNN, Masking, Dropout, GRU, Lambda, LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
import mord
import tensorflow.python.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import Model, Sequential


np.random.seed(7)


class Models:
    def __init__(self, c, total_fold=10, current_fold=1):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.c = c
        self.clinic_data = c.clinic_data
        self.gcs_data = c.gcs_data
        self.lab_data = c.lab_data
        self.vital_data = c.vital_data
        self.cleaned_version = c.cleaned_version
        self.guid = []
        self.model = None
        self.ts_data = []
        self.dem_data = []
        self.output = []
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
        self.val_x = []
        self.val_y = []
        self.num_features = 0
        self.mask = False
        self.train_history = None
        self.output_type = None
        self.total_fold = total_fold
        self.current_fold = current_fold
        self.sample_weight = None
        self.train_guid = None
        self.test_guid = None
        self.val_guid = None

    def evaluate(self, metric='MSE'):
        train_predict = self.model.predict(self.train_x)
        test_predict = self.model.predict(self.test_x)
        
        train_predict = np.around(train_predict)
        test_predict = np.around(test_predict)

        train_measure = evaluation(self.train_y, train_predict, metric=metric, output_type=self.output_type)
        test_measure = evaluation(self.test_y, test_predict, metric=metric, output_type=self.output_type)
        
        # calculate mean squared error
        print('##########################')
        print('Train ' + metric + ': ' + str(train_measure))
        print('Test ' + metric + ': ' + str(test_measure))

        return train_measure, test_measure
    
    def plot_loss(self, save_name='loss', drop_items=1):
        plt.plot(self.train_history.history['loss'][drop_items:])
        plt.plot(self.train_history.history['val_loss'][drop_items:])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'validation loss'], loc='upper left')
        plt.savefig(FIG_PATH + save_name)
        plt.show()
    
    def save(self, save_name='output'):
        with open(OUTPUT_PATH + save_name, 'wb') as f:
            saveObject = {'train_y': self.train_y,
                          'train_predict': self.model.predict(self.train_x),
                          'test_y': self.test_y,
                          'test_predict': self.model.predict(self.test_x)}
            pickle.dump(saveObject, f)
class TSRegression(Models):
    def __init__(self, c, output_type='Regression', filename=DATA_PATH + 'time_series_data.csv',
                 train_split=0.8, max_len=120, imput_method='mean', drop_rates=[0.3]*4, reg_rate=0.01,
                 es_patience=20, epochs=10, num_neurons=20, num_hidden=20, learning_rate=0.001,
                 save_bool=True, running=True, plot_bool=False, total_fold=10, current_fold=1):
        Models.__init__(self, c, total_fold=total_fold, current_fold=current_fold)
        self.data_mean = None
        self.data_std = None
        self.num_time_steps = 0
        self.output_type = output_type
        self.num_output = 1

class RNNGRUD(Models):
    def __init__(self, c, output_type='Regression', filename=DATA_PATH + 'time_series_data.csv',
                 train_split=0.8, max_len=120, imput_method='mean', drop_rates=[0.3]*4, reg_rate=0.01,
                 es_patience=20, epochs=10, num_neurons=20, num_hidden=20, learning_rate=0.001,
                 save_bool=True, running=True, plot_bool=False, total_fold=10, current_fold=1):
        Models.__init__(self, c, total_fold=total_fold, current_fold=current_fold)
        self.data_mean = None
        self.data_std = None
        self.num_time_steps = 0
        self.output_type = output_type
        self.num_output = 1

        if running:
            # Running the model
            self.preprocess(filename=filename, train_split=train_split,
                            max_len=max_len, imput_method=imput_method)
            self.fit(epochs=epochs, num_neurons=num_neurons, num_hidden=num_hidden,
                     drop_rates=drop_rates, reg_rate=reg_rate, es_patience=es_patience,
                     learning_rate=learning_rate)

            # Saving the output
            if save_bool:
                save_name = 'output_grud_type_' + output_type + '_maxLen_' + str(max_len) + '_imput_' + \
                            imput_method + '_epochs_' + str(epochs) + '_units_' + str(num_neurons) + '_' + \
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.save(save_name=save_name)
            if plot_bool:
                save_name = 'plot_grud_type_' + output_type + '_maxLen_' + str(max_len) + '_imput_' + \
                            imput_method + '_epochs_' + str(epochs) + '_units_' + str(num_neurons) + '_' + \
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.plot_loss(save_name=save_name)

    def preprocess(self, filename=DATA_PATH + 'time_series_data.csv',
                   train_split=0.8, max_len=120, imput_method='mean'):
        df = pd.read_csv(filename)

        # Adjusting the length of time series
        self.num_time_steps = min(max(df.DLTimeSinceInj), max_len)
        df = df[df.DLTimeSinceInj < self.num_time_steps]

        # Finding patients with both lab and output data
        guid_ts = df.Guid.unique()
        guid_dem = self.clinic_data.Guid.unique()
        self.guid = np.intersect1d(guid_ts, guid_dem)
        df = df[df.Guid.isin(self.guid)]

        # Initializing some variables
        varis = sorted(list(set(df['variable'])))
        self.num_features = len(varis)
        pat_to_ind = inv_list(self.guid)
        var_to_ind = inv_list(varis)
        values = np.zeros((len(self.guid), self.num_time_steps, self.num_features))
        masking = np.zeros((len(self.guid), self.num_time_steps, self.num_features))

        # Populating the data
        for row in tqdm(df.itertuples()):
            pind = pat_to_ind[getattr(row, 'Guid')]
            vind = var_to_ind[getattr(row, 'variable')]
            tstep = getattr(row, 'DLTimeSinceInj')
            values[pind, tstep, vind] = getattr(row, 'value')
            masking[pind, tstep, vind] = 1

        # Populating the output
        dem_df = self.clinic_data[self.clinic_data.Guid.isin(self.guid)]
        outcomes = np.zeros(len(self.guid))
        for row in dem_df.itertuples():
            pind = pat_to_ind[getattr(row, 'Guid')]
            outcomes[pind] = getattr(row, 'GOSEScore')
        outcomes = outcomes.astype(int)

        # Populating demographic data
        self.dem_data = [np.squeeze(
            dem_df.loc[dem_df.Guid == i, (dem_df.columns != 'Guid') & (dem_df.columns != 'GOSEScore')].values.astype(
                'float32'))
                         for i in self.guid]

        # Imputation
        if imput_method == 'multivar':
            imputer = IterativeImputer(missing_values=np.nan)
        else:
            imputer = SimpleImputer(missing_values=np.nan, strategy=imput_method)
        self.dem_data = imputer.fit_transform(self.dem_data)

        # Normalize time series data
        if self.data_mean is None:
            mvalues = np.ma.array(values, mask=1 - masking)
            self.data_mean = np.array(mvalues.mean(axis=(0, 1)))
            self.data_std = np.array(mvalues.std(axis=(0, 1)))
        values = (values - self.data_mean.reshape((1, 1, self.num_features))) / self.data_std.reshape(
            (1, 1, self.num_features))
        values = masking * values

        # Normalizing demographic data
        self.dem_data = self.scaler.fit_transform(self.dem_data)

        # Split data to train, test and validation set
        if self.total_fold is None or self.total_fold < self.current_fold:
            raise Exception('The values of total fold and/or current fold does not make sense')
        else:
            kf = StratifiedKFold(n_splits=self.total_fold, shuffle=True, random_state=0)
            f = 1
            for train_ind, test_ind in kf.split(X=values, y=outcomes):
                if f == self.current_fold:
                    train_ts = np.take(values, train_ind, axis=0)
                    test_ts = np.take(values, test_ind, axis=0)

                    train_m = np.take(masking, train_ind, axis=0)
                    test_m = np.take(masking, test_ind, axis=0)

                    train_dem = np.take(self.dem_data, train_ind, axis=0)
                    test_dem = np.take(self.dem_data, test_ind, axis=0)

                    self.train_y = np.take(outcomes, train_ind, axis=0)
                    self.test_y = np.take(outcomes, test_ind, axis=0)
                    
                    self.train_guid = np.take(self.guid, train_ind, axis=0)
                    self.test_guid = np.take(self.guid, test_ind, axis=0)
                f += 1
            train_ts, val_ts, train_m, val_m, train_dem, val_dem, \
            self.train_y, self.val_y, self.train_guid, self.val_guid = train_test_split(train_ts, train_m, train_dem, self.train_y, self.train_guid,
                                                                                        test_size=1/(self.total_fold-1), random_state=0,
                                                                                        stratify=self.train_y)
            self.train_x = [train_ts, train_m, train_dem]
            self.val_x = [val_ts, val_m, val_dem]
            self.test_x = [test_ts, test_m, test_dem]

        # Defining sample weights
        # self.sample_weight = sample_weight_builder(self.train_y)

        # Discretizing the outcome
        self.train_y, self.num_output = output_builder(self.train_y, self.output_type)
        self.test_y, _ = output_builder(self.test_y, self.output_type)
        self.val_y, _ = output_builder(self.val_y, self.output_type)

    def fit(self, epochs=10, num_neurons=20, num_hidden=20,
            drop_rates=[0.3] * 4, reg_rate=0.01, es_patience=20, learning_rate=0.001):
        activation_dict = {'Regression': None,
                           'Binary': 'sigmoid',
                           'Multiclass': 'softmax',
                           'OrdinalMulticlass': 'sigmoid'}
        loss_dict = {'Regression': 'mean_squared_error',
                     'Binary': 'binary_crossentropy',
                     'Multiclass': 'categorical_crossentropy',
                     'OrdinalMulticlass': 'mean_squared_error'}

        # Inputs
        input_x = Input(shape=(self.num_time_steps, self.num_features))
        input_m = Input(shape=(self.num_time_steps, self.num_features))
        input_s = Input(shape=(self.num_time_steps, 1))
        # Adopted peice of code for input_s-----------------------
        tstep = np.ones((len(self.train_x[0]) + len(self.val_x[0]) + len(self.test_x[0]), self.num_time_steps, 1)) * \
                np.reshape(np.arange(0, self.num_time_steps), (1, self.num_time_steps, 1))
        self.train_x.insert(2, tstep[:len(self.train_x[0])])
        self.val_x.insert(2, tstep[len(self.train_x[0]):len(self.train_x[0]) + len(self.val_x[0])])
        self.test_x.insert(2, tstep[len(self.train_x[0]) + len(self.val_x[0]):])
        # ---------------------------------------------------------
        input_list = [input_x, input_m, input_s]

        input_x = ExternalMasking()([input_x, input_m])
        input_s = ExternalMasking()([input_s, input_m])
        input_m = Masking()(input_m)

        # GRU layer
        grud_layer = GRUD(units=num_neurons,
                          return_sequences=True,
                          activation='sigmoid',
                          dropout=drop_rates[0],
                          recurrent_dropout=drop_rates[1]
                          )
        rec = grud_layer([input_x, input_m, input_s])
        rec = Dropout(drop_rates[2])(rec)
        att_layer = Attention(64)
        alpha = att_layer(rec)

        fused = Lambda(lambda x: K.sum(x[0] * x[1], axis=-2))([rec, alpha])

        # Non-temporal layer
        dem_input = Input(shape=(len(self.dem_data[0]),))
        input_list.insert(3, dem_input)
        concat = Concatenate()([fused, dem_input])

        # Hidden layer
        hdn = Dense(units=num_hidden, activation='relu', kernel_regularizer=regularizers.l2(reg_rate))(concat)
        hdn = Dropout(drop_rates[3])(hdn)

        # Output layer
        op = Dense(self.num_output,
                   activation=activation_dict[self.output_type])(hdn)

        # Creating and compiling model
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=es_patience, restore_best_weights=True)

        self.model = Model(inputs=input_list, outputs=op)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_dict[self.output_type],
                           metrics=['accuracy'])
        self.model.summary()
        self.train_history = self.model.fit(self.train_x, self.train_y,
                                            epochs=epochs, verbose=2,
                                            validation_data=(self.val_x, self.val_y),
                                            callbacks=[es],
                                            sample_weight=self.sample_weight)


class ANN(Models):
    def __init__(self, c, output_type='Regression', filename=DATA_PATH + 'time_series_data.csv',
                 train_split=0.8, imput_method='mean', epochs=10, num_hidden=20,
                 save_bool=True, running=True, plot_bool=False, total_fold=None, current_fold=1):
        Models.__init__(self, c, total_fold=total_fold, current_fold=current_fold)
        self.data_mean = None
        self.data_std = None
        self.num_time_steps = 0
        self.output_type = output_type
        self.num_output = 1

        if running:
            # Running the model
            self.preprocess(filename=filename, train_split=train_split, imput_method=imput_method)
            self.fit(epochs=epochs, num_hidden=num_hidden)

            # Saving the output
            if save_bool:
                save_name = 'output_ann_type_' + output_type + '_imput_' + \
                            imput_method + '_epochs_' + str(epochs) + '_' + \
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.save(save_name=save_name)
            if plot_bool:
                save_name = 'plot_ann_type_' + output_type + '_imput_' + \
                            imput_method + '_epochs_' + str(epochs) + '_' + \
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.plot_loss(save_name=save_name)

    def preprocess(self, filename=DATA_PATH + 'time_series_data.csv',
                   train_split=0.8, imput_method='mean'):

        # Finding patients with both lab and output data
        ts_df = pd.read_csv(filename)
        guid_ts = ts_df.Guid.unique()
        guid_dem = self.clinic_data.Guid.unique()
        self.guid = np.intersect1d(guid_ts, guid_dem)

        dem_df = self.clinic_data.loc[self.clinic_data.Guid.isin(self.guid), :]
        self.dem_data = dem_df.drop(['GOSEScore', 'Guid'], axis=1)
        self.output = np.array(dem_df['GOSEScore'].tolist()).astype(int)
        outcomes = self.output

        # Populating demographic data
        self.dem_data = [np.squeeze(
            dem_df.loc[dem_df.Guid == i, (dem_df.columns != 'Guid') & (dem_df.columns != 'GOSEScore')].values.astype(
                'float32'))
                         for i in self.guid]

        # Imputation
        if imput_method == 'multivar':
            imputer = IterativeImputer(missing_values=np.nan)
        else:
            imputer = SimpleImputer(missing_values=np.nan, strategy=imput_method)
        self.dem_data = imputer.fit_transform(self.dem_data)

        # Normalizing demographic data
        self.dem_data = self.scaler.fit_transform(self.dem_data)

        # Split data to train, test and validation set
        if self.total_fold is None or self.total_fold < self.current_fold:
            raise Exception('The values of total fold and/or current fold does not make sense')
        else:
            kf = StratifiedKFold(self.total_fold, shuffle=True, random_state=0)
            f = 1
            for train_ind, test_ind in kf.split(X=self.dem_data, y=outcomes):
                if f == self.current_fold:

                    train_dem = np.take(self.dem_data, train_ind, axis=0)
                    test_dem = np.take(self.dem_data, test_ind, axis=0)

                    self.train_y = np.take(outcomes, train_ind, axis=0)
                    self.test_y = np.take(outcomes, test_ind, axis=0)
                f += 1
        train_dem, val_dem,\
        self.train_y, self.val_y = train_test_split(train_dem, self.train_y,
                                                    test_size=1 / (self.total_fold - 1), random_state=0,
                                                    stratify=self.train_y)
        self.train_x = train_dem
        self.test_x = test_dem
        self.val_x = val_dem

        # Discretizing the outcome
        self.train_y, self.num_output = output_builder(self.train_y, self.output_type)
        self.test_y, _ = output_builder(self.test_y, self.output_type)
        self.val_y, _ = output_builder(self.val_y, self.output_type)

    def fit(self, epochs=10, num_neurons=20, num_hidden=20):
        activation_dict = {'Regression': None,
                           'Binary': 'sigmoid',
                           'Multiclass': 'softmax',
                           'OrdinalMulticlass': 'sigmoid'}
        loss_dict = {'Regression': 'mean_squared_error',
                     'Binary': 'binary_crossentropy',
                     'Multiclass': 'categorical_crossentropy',
                     'OrdinalMulticlass': 'mean_squared_error'}

        # Non-temporal layer
        dem_input = Input(shape=(len(self.dem_data[0]),))

        # Hidden layer
        hdn = Dense(units=num_hidden, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dem_input)
        hdn = Dropout(.3)(hdn)

        # Output layer
        op = Dense(self.num_output,
                   activation=activation_dict[self.output_type])(hdn)

        # Creating and compiling model
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)

        self.model = Model(inputs=dem_input, outputs=op)
        self.model.compile(optimizer='adam', loss=loss_dict[self.output_type],
                           metrics=['accuracy'])
        self.model.summary()
        self.train_history = self.model.fit(self.train_x, self.train_y,
                                            epochs=epochs, verbose=2,
                                            validation_data=(self.val_x, self.val_y),
                                            callbacks=[es])


class RNNModel(Models):
    def __init__(self, c, output_type='Regression', filename=DATA_PATH + 'time_series_data.csv',
                 train_split=0.8, max_len=120, imput_method='mean', drop_rates=[0.3]*4, reg_rate=0.01,
                 es_patience=20, epochs=10, num_neurons=20, num_hidden=20, learning_rate=0.001,
                 save_bool=True, running=True, plot_bool=False, total_fold=10, current_fold=1):
        Models.__init__(self, c, total_fold=total_fold, current_fold=current_fold)
        self.data_mean = None
        self.data_std = None
        self.num_time_steps = 0
        self.output_type = output_type
        self.num_output = 1

        if running:
            # Running the model
            self.preprocess(filename=filename, train_split=train_split,
                            max_len=max_len, imput_method=imput_method)
            self.fit(epochs=epochs, num_neurons=num_neurons, num_hidden=num_hidden,
                     drop_rates=drop_rates, reg_rate=reg_rate, es_patience=es_patience,
                     learning_rate=learning_rate)

            # Saving the output
            if save_bool:
                save_name = 'output_rnn_type_' + output_type + '_maxLen_' + str(max_len) + '_imput_' + \
                            imput_method + '_epochs_' + str(epochs) + '_units_' + str(num_neurons) + '_' + \
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.save(save_name=save_name)
            if plot_bool:
                save_name = 'plot_rnn_type_' + output_type + '_maxLen_' + str(max_len) + '_imput_' + \
                            imput_method + '_epochs_' + str(epochs) + '_units_' + str(num_neurons) + '_' + \
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.plot_loss(save_name=save_name)

    def preprocess(self, filename=DATA_PATH + 'time_series_data.csv',
                   train_split=0.8, max_len=120, imput_method='mean'):
        df = pd.read_csv(filename)

        # Adjusting the length of time series
        self.num_time_steps = min(max(df.DLTimeSinceInj), max_len)
        df = df[df.DLTimeSinceInj < self.num_time_steps]

        # Finding patients with both lab and output data
        guid_ts = df.Guid.unique()
        guid_dem = self.clinic_data.Guid.unique()
        self.guid = np.intersect1d(guid_ts, guid_dem)
        df = df[df.Guid.isin(self.guid)]

        # Initializing some variables
        varis = sorted(list(set(df['variable'])))
        self.num_features = len(varis)
        pat_to_ind = inv_list(self.guid)
        var_to_ind = inv_list(varis)
        values = np.zeros((len(self.guid), self.num_time_steps, self.num_features))
        masking = np.zeros((len(self.guid), self.num_time_steps, self.num_features))

        # Populating the data
        for row in tqdm(df.itertuples()):
            pind = pat_to_ind[getattr(row, 'Guid')]
            vind = var_to_ind[getattr(row, 'variable')]
            tstep = getattr(row, 'DLTimeSinceInj')
            values[pind, tstep, vind] = getattr(row, 'value')
            masking[pind, tstep, vind] = 1

        # Populating the output
        dem_df = self.clinic_data[self.clinic_data.Guid.isin(self.guid)]
        outcomes = np.zeros(len(self.guid))
        for row in dem_df.itertuples():
            pind = pat_to_ind[getattr(row, 'Guid')]
            outcomes[pind] = getattr(row, 'GOSEScore')
        outcomes = outcomes.astype(int)

        # Normalize time series data
        if self.data_mean is None:
            mvalues = np.ma.array(values, mask=1 - masking)
            self.data_mean = np.array(mvalues.mean(axis=(0, 1)))
            self.data_std = np.array(mvalues.std(axis=(0, 1)))
        values = (values - self.data_mean.reshape((1, 1, self.num_features))) / self.data_std.reshape(
            (1, 1, self.num_features))
        values = masking * values

        # Split data to train, test and validation set
        if self.total_fold is None or self.total_fold < self.current_fold:
            raise Exception('The values of total fold and/or current fold does not make sense')
        else:
            kf = StratifiedKFold(n_splits=self.total_fold, shuffle=True, random_state=0)
            f = 1
            for train_ind, test_ind in kf.split(X=values, y=outcomes):
                if f == self.current_fold:
                    train_ts = np.take(values, train_ind, axis=0)
                    test_ts = np.take(values, test_ind, axis=0)

                    train_m = np.take(masking, train_ind, axis=0)
                    test_m = np.take(masking, test_ind, axis=0)

                    self.train_y = np.take(outcomes, train_ind, axis=0)
                    self.test_y = np.take(outcomes, test_ind, axis=0)
                f += 1
            train_ts, val_ts, train_m, val_m, \
            self.train_y, self.val_y = train_test_split(train_ts, train_m, self.train_y,
                                                        test_size=1 / (self.total_fold - 1), random_state=0,
                                                        stratify=self.train_y)
            self.train_x = [train_ts, train_m]
            self.val_x = [val_ts, val_m]
            self.test_x = [test_ts, test_m]

        # Discretizing the outcome
        self.train_y, self.num_output = output_builder(self.train_y, self.output_type)
        self.test_y, _ = output_builder(self.test_y, self.output_type)
        self.val_y, _ = output_builder(self.val_y, self.output_type)

    def fit(self, epochs=10, num_neurons=20, num_hidden=20,
            drop_rates=[0.3] * 4, reg_rate=0.01, es_patience=20,
            learning_rate=0.001):
        activation_dict = {'Regression': None,
                           'Binary': 'sigmoid',
                           'Multiclass': 'softmax',
                           'OrdinalMulticlass': 'sigmoid'}
        loss_dict = {'Regression': 'mean_squared_error',
                     'Binary': 'binary_crossentropy',
                     'Multiclass': 'categorical_crossentropy',
                     'OrdinalMulticlass': 'mean_squared_error'}

        # Inputs
        input_x = Input(shape=(None, self.num_features))
        input_m = Input(shape=(None, self.num_features))
        input_s = Input(shape=(None, 1))
        # Adopted peice of code for input_s-----------------------
        tstep = np.ones((len(self.train_x[0]) + len(self.val_x[0]) + len(self.test_x[0]), self.num_time_steps, 1)) * \
                np.reshape(np.arange(0, self.num_time_steps), (1, self.num_time_steps, 1))

        self.train_x.insert(2, tstep[:len(self.train_x[0])])
        self.val_x.insert(2, tstep[len(self.train_x[0]):len(self.train_x[0]) + len(self.val_x[0])])
        self.test_x.insert(2, tstep[len(self.train_x[0]) + len(self.val_x[0]):])
        # ---------------------------------------------------------
        input_list = [input_x, input_m, input_s]

        input_x = ExternalMasking()([input_x, input_m])
        input_s = ExternalMasking()([input_s, input_m])
        input_m = Masking()(input_m)

        # GRU layer
        grud_layer = GRUD(units=num_neurons,
                          return_sequences=False,
                          activation='sigmoid',
                          dropout=drop_rates[0],
                          recurrent_dropout=drop_rates[1]
                          )
        rec = grud_layer([input_x, input_m, input_s])
        rec = Dropout(drop_rates[2])(rec)

        # Hidden layer
        hdn = Dense(units=num_hidden, activation='relu', kernel_regularizer=regularizers.l2(reg_rate))(rec)
        hdn = Dropout(drop_rates[3])(hdn)

        # Output layer
        op = Dense(self.num_output,
                   activation=activation_dict[self.output_type])(hdn)

        # Creating and compiling model
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=es_patience)

        self.model = Model(inputs=input_list, outputs=op)
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_dict[self.output_type],
                           metrics=['accuracy'])
        self.model.summary()
        self.train_history = self.model.fit(self.train_x, self.train_y,
                                            epochs=epochs, verbose=2,
                                            validation_data=(self.val_x, self.val_y),
                                            callbacks=[es])


class RNNSimple(RNNModel):
    def __init__(self, c, output_type='Regression', filename=DATA_PATH + 'time_series_data.csv',
                 train_split=0.8, max_len=120, imput_method='mean', drop_rates=[0.3]*4, reg_rate=0.01,
                 es_patience=20, epochs=10, num_neurons=20, num_hidden=20, learning_rate=0.001,
                 save_bool=True, running=True, plot_bool=False, total_fold=10, current_fold=1):
        RNNModel.__init__(self, c, output_type=output_type, filename=filename,
                          train_split=train_split, max_len=max_len, imput_method=imput_method,
                          drop_rates=drop_rates, reg_rate=reg_rate, es_patience=es_patience,
                          epochs=epochs, num_neurons=num_neurons, num_hidden=num_hidden,
                          learning_rate=learning_rate, save_bool=save_bool, running=running,
                          plot_bool=plot_bool, total_fold=total_fold, current_fold=current_fold)

    def preprocess(self, filename=DATA_PATH + 'time_series_data.csv',
                   train_split=0.8, max_len=120, imput_method='mean'):
        values, masking, pat_to_ind, var_to_ind = ts_builder_3d(self.c, file_name=filename, masking=True)
        self.num_time_steps = max_len
        self.num_features = len(var_to_ind)

        # Populating the output
        self.guid = list(pat_to_ind.keys())
        dem_df = self.clinic_data[self.clinic_data.Guid.isin(self.guid)]
        outcomes = np.zeros(len(self.guid))
        for row in dem_df.itertuples():
            pind = pat_to_ind[getattr(row, 'Guid')]
            outcomes[pind] = getattr(row, 'GOSEScore')
        outcomes = outcomes.astype(int)

        # Normalize time series data
        if self.data_mean is None:
            mvalues = np.ma.array(values, mask=1 - masking)
            self.data_mean = np.array(mvalues.mean(axis=(0, 1)))
            self.data_std = np.array(mvalues.std(axis=(0, 1)))
        values = (values - self.data_mean.reshape((1, 1, self.num_features))) / self.data_std.reshape(
            (1, 1, self.num_features))
        values = np.where(masking == 0, np.nan, values)

        # Data Imputation
        for j in range(values.shape[2]):
            for i in range(values.shape[0]):
                val = pd.Series(values[i, :, j])
                val = val.interpolate(limit_direction='both')
                values[i, :, j] = val.values
        values = np.nan_to_num(values, nan=0)  # imputing time series with all nan

        # Split data to train, test and validation set
        if self.total_fold is None or self.total_fold < self.current_fold:
            raise Exception('The values of total fold and/or current fold does not make sense')
        else:
            kf = StratifiedKFold(n_splits=self.total_fold, shuffle=True, random_state=0)
            f = 1
            for train_ind, test_ind in kf.split(X=values, y=outcomes):
                if f == self.current_fold:
                    train_ts = np.take(values, train_ind, axis=0)
                    test_ts = np.take(values, test_ind, axis=0)

                    train_m = np.take(masking, train_ind, axis=0)
                    test_m = np.take(masking, test_ind, axis=0)

                    self.train_y = np.take(outcomes, train_ind, axis=0)
                    self.test_y = np.take(outcomes, test_ind, axis=0)
                f += 1
            train_ts, val_ts, train_m, val_m, \
            self.train_y, self.val_y = train_test_split(train_ts, train_m, self.train_y,
                                                        test_size=1 / (self.total_fold - 1), random_state=0,
                                                        stratify=self.train_y)
            self.train_x = train_ts
            self.val_x = val_ts
            self.test_x = test_ts

        # Discretizing the outcome
        self.train_y, self.num_output = output_builder(self.train_y, self.output_type)
        self.test_y, _ = output_builder(self.test_y, self.output_type)
        self.val_y, _ = output_builder(self.val_y, self.output_type)

    def fit(self, epochs=10, num_neurons=20, num_hidden=20,
            drop_rates=[0.3] * 4, reg_rate=0.01, es_patience=20,
            learning_rate=0.001):
        activation_dict = {'Regression': None,
                           'Binary': 'sigmoid',
                           'Multiclass': 'softmax',
                           'OrdinalMulticlass': 'sigmoid'}
        loss_dict = {'Regression': 'mean_squared_error',
                     'Binary': 'binary_crossentropy',
                     'Multiclass': 'categorical_crossentropy',
                     'OrdinalMulticlass': 'mean_squared_error'}

        self.model = Sequential()
        input_shape = (self.num_time_steps, self.num_features)
        self.model.add(LSTM(units=num_neurons, activation='relu', input_shape=input_shape))
        self.model.add(Dense(units=num_hidden, activation='relu', kernel_regularizer=regularizers.l2(reg_rate)))
        self.model.add(Dropout(drop_rates[3]))
        self.model.add(Dense(units=self.num_output, activation=activation_dict[self.output_type]))

        # Creating and compiling model
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=es_patience)

        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_dict[self.output_type],
                           metrics=['accuracy'])
        self.model.summary()
        self.train_history = self.model.fit(self.train_x, self.train_y,
                                            epochs=epochs, verbose=2, batch_size=16,
                                            validation_data=(self.val_x, self.val_y),
                                            callbacks=[es])


class Regression(Models):
    def __init__(self, c, output_type='Regression', train_split=0.8, imput_method='mean',
                 save_bool=True, running=True, total_fold=None, current_fold=1):
        Models.__init__(self, c, total_fold=total_fold, current_fold=current_fold)
        self.data_mean = None
        self.data_std = None
        self.output_type = output_type
        self.num_output = 1
        
        if running:
            # Running the model
            self.preprocess(train_split=train_split, imput_method=imput_method)
            self.fit()
        
            # Saving the output
            if save_bool:
                save_name = 'output_allReg_type_' + output_type + '_imput_' + imput_method + '_' +\
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.save(save_name=save_name)

    def preprocess(self, train_split=0.8, imput_method='mean', filename=DATA_PATH + 'time_series_data.csv'):

        # Finding patients with both lab and output data
        ts_df = pd.read_csv(filename)
        guid_ts = ts_df.Guid.unique()
        guid_dem = self.clinic_data.Guid.unique()
        self.guid = np.intersect1d(guid_ts, guid_dem)
        
        
        df = self.clinic_data.loc[self.clinic_data.Guid.isin(self.guid), :]
        self.dem_data = df.drop(['GOSEScore', 'Guid'], axis=1)
        self.output = df['GOSEScore']
        
        # Imputation
        if imput_method == 'multivar':
            imputer = IterativeImputer(missing_values=np.nan)
        else:
            imputer = SimpleImputer(missing_values=np.nan, strategy=imput_method)
        self.dem_data = imputer.fit_transform(self.dem_data)

        # Split data to train and test
        if self.total_fold is None:
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.dem_data, self.output,
                                                                                    test_size=1 - train_split,
                                                                                    random_state=0)
        else:
            kf = KFold(self.total_fold, shuffle=True, random_state=0)
            f = 1
            for train_ind, test_ind in kf.split(self.output):
                if f == self.current_fold:
                    self.train_x = np.take(self.dem_data, train_ind, axis=0)
                    self.test_x = np.take(self.dem_data, test_ind, axis=0)

                    self.train_y = np.take(self.output, train_ind, axis=0)
                    self.test_y = np.take(self.output, test_ind, axis=0)
                f += 1

    def fit(self):
        self.model = LinearRegression()
        self.model.fit(self.train_x, self.train_y)


class IMPACT_Reg(Models):
    def __init__(self, c, output_type='Regression', train_split=0.8, imput_method='mean',
                 save_bool=True, running=True, plot_bool=False, total_fold=None, current_fold=1,
                 ts_stats=False, max_len=120):
        Models.__init__(self, c, total_fold=total_fold, current_fold=current_fold)
        self.data_mean = None
        self.data_std = None
        self.output_type = output_type
        self.num_output = 1
        
        if running:
            # Running the model
            self.preprocess(train_split=train_split, imput_method=imput_method, ts_stats=ts_stats, max_len=max_len)
            self.fit()
        
            # Saving the output
            if save_bool:
                save_name = 'output_impact_type_' + output_type + '_imput_' + imput_method + '_' +\
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.save(save_name=save_name)

            if plot_bool:
                save_name = 'plot_impact_type_' + output_type + '_imput_' + imput_method + '_' +\
                            str(self.total_fold) + 'Fold_' + str(self.current_fold)
                self.plot_loss(save_name=save_name)
    
    def preprocess(self, train_split=0.8, imput_method='mean', filename=DATA_PATH + 'time_series_data.csv',
                  ts_stats=False, max_len=120):

        ts_df = pd.read_csv(filename)
        
        # Adjusting the length of time series
        self.num_time_steps = min(max(ts_df.DLTimeSinceInj), max_len)
        ts_df = ts_df[ts_df.DLTimeSinceInj < self.num_time_steps]
        
        # Finding patients with both lab and output data
        guid_ts = ts_df.Guid.unique()
        guid_dem = self.clinic_data.Guid.unique()
        self.guid = np.intersect1d(guid_ts, guid_dem)
        self.guid = self.clinic_data[self.clinic_data.Guid.isin(self.guid)].Guid.unique()

        if self.cleaned_version:
            # Collecting the necessary columns
            add_cols = ['GOSEScore', 'Hypoxia_cleaned', 'Hypotension_cleaned', 'EDGlucose',
                        'EDHemoglobin', 'AgeRecodedPHI', 'mGCS']

            add_cols.extend(self.clinic_data.columns[self.clinic_data.columns.to_series().str.startswith('Pupils')])
            df = self.clinic_data.loc[self.clinic_data.Guid.isin(self.guid), add_cols]

            # Applying IMPACT scores----------
            df['AgeRecodedPHI'] = pd.cut(df.AgeRecodedPHI, bins=[0, 30, 39, 49, 59, 69, 100], right=True, labels=False)
            mapping = {0: 3,
                       1: 1,
                       2: 6,
                       3: 4,
                       4: 2,
                       5: 0,
                       6: 0}
            df['mGCS'] = df['mGCS'].replace(mapping)
            df['Pupils'] = df['Pupils_One'] * 2 + df['Pupils_Neither'] * 4
            df['Hypotension_cleaned'] = df.Hypotension_cleaned * 2
            df['EDHemoglobin'] = 3 - pd.cut(df.EDHemoglobin, bins=[0, 9, 11.9, 14.9, 100], right=True, labels=False)
            df['EDGlucose'] = pd.cut(df.EDGlucose * 0.0555, bins=[0, 6, 8.9, 11.9, 14.9, 100], right=True, labels=False)
            # ----------------------------------

            drop_cols = ['GOSEScore']
            drop_cols.extend(self.clinic_data.columns[self.clinic_data.columns.to_series().str.startswith('Pupils_')])
        else:
            # Collecting the necessary columns
            add_cols = ['GOSEScore', 'EDComplEventHypoxia', 'EDComplEventHypotension', 'EDGlucose',
                        'EDHemoglobin', 'AgeRecodedPHI']
            add_cols.extend(self.clinic_data.columns[self.clinic_data.columns.to_series().str.startswith('GcsEDArrMotor')])
            add_cols.extend(self.clinic_data.columns[self.clinic_data.columns.to_series().str.startswith('Pupils')])
            df = self.clinic_data.loc[self.clinic_data.Guid.isin(self.guid), add_cols]

            # Applying IMPACT scores----------
            df['AgeRecodedPHI'] = pd.cut(df.AgeRecodedPHI, bins=[0, 30, 39, 49, 59, 69, 100], right=True, labels=False)
            df['GcsEDArrMotor'] = (df['GcsEDArrMotor_1-No Response'] + df['GcsEDArrMotor_2-Extension']) * 6 +\
                                  df['GcsEDArrMotor_3-Flexion Abnormal'] * 4 + \
                                  df['GcsEDArrMotor_4-Flexion Withdrawal'] * 2 +\
                                  df['GcsEDArrMotor_P-Untestable (Paralyzed)'] * 3
            df['Pupils'] = df['Pupils_One'] * 2 + df['Pupils_Neither'] * 4
            df['EDComplEventHypotension'] = df.EDComplEventHypotension * 2
            df['EDHemoglobin'] = 3 - pd.cut(df.EDHemoglobin, bins=[0, 9, 11.9, 14.9, 100], right=True, labels=False)
            df['EDGlucose'] = pd.cut(df.EDGlucose * 0.0555, bins=[0, 6, 8.9, 11.9, 14.9, 100], right=True, labels=False)
            # ----------------------------------

            drop_cols = ['GOSEScore']
            drop_cols.extend(self.clinic_data.columns[self.clinic_data.columns.to_series().str.startswith('GcsEDArrMotor_')])
            drop_cols.extend(self.clinic_data.columns[self.clinic_data.columns.to_series().str.startswith('Pupils_')])
        
        # Adding Time series stats to data
        if ts_stats:
            
            # Finding patients with both lab and output data
            ts_df = ts_df[ts_df.Guid.isin(self.guid)]

            # Initializing some variables
            varis = sorted(list(set(ts_df['variable'])))
            self.num_features = len(varis)
            pat_to_ind = inv_list(self.guid)
            var_to_ind = inv_list(varis)
            values = np.zeros((len(self.guid), self.num_time_steps, self.num_features))
            values.fill(np.nan)

            # Populating the data
            for row in tqdm(ts_df.itertuples()):
                pind = pat_to_ind[getattr(row, 'Guid')]
                vind = var_to_ind[getattr(row, 'variable')]
                tstep = getattr(row, 'DLTimeSinceInj')
                values[pind, tstep, vind] = getattr(row, 'value')
            values_summary = np.nanmean(values, axis=1)
            
            # Concatenate static and temporal data
            df.reset_index(drop=True, inplace=True)
            df = pd.concat([df, pd.DataFrame(values_summary, columns=varis)], axis=1)
        
        # Creating data as numpy arrays
        self.dem_data = df.drop(drop_cols, axis=1).values
        self.output = df['GOSEScore'].values
        
        # Imputation
        if imput_method == 'multivar':
            imputer = IterativeImputer(missing_values=np.nan)
        else:
            imputer = SimpleImputer(missing_values=np.nan, strategy=imput_method)
        self.dem_data = imputer.fit_transform(self.dem_data)

        # Split data to train and test
        if self.total_fold is None:
            self.train_x, self.test_x, self.train_y, self.test_y \
                = train_test_split(self.dem_data, self.output,
                                   test_size=1 - train_split,
                                   random_state=0,
                                   stratify=self.output)
        else:
            kf = StratifiedKFold(self.total_fold, shuffle=True, random_state=0)
            f = 1
            for train_ind, test_ind in kf.split(X=self.dem_data, y=self.output):
                if f == self.current_fold:
                    self.train_x = np.take(self.dem_data, train_ind, axis=0)
                    self.test_x = np.take(self.dem_data, test_ind, axis=0)

                    self.train_y = np.take(self.output, train_ind, axis=0)
                    self.test_y = np.take(self.output, test_ind, axis=0)
                f += 1

        # Discretizing the outcome
        self.train_y, self.num_output = output_builder(self.train_y, self.output_type)
        self.test_y, _ = output_builder(self.test_y, self.output_type)

    def fit(self):
        # self.model = LinearRegression()
        # self.model.fit(self.train_x, self.train_y)
        self.model = mord.OrdinalRidge()
        self.model.fit(self.train_x, self.train_y)


if __name__ == '__main__':
    from preprocess import *
    import shap

    c = Connection(verbose=False)
    c.clean_clinic_data(miss_rate=0.2, max_gcs=15)
    c.clean_gcs_data()
    c.clean_vital_data()
    c.clean_lab_data()
    
    K.clear_session()
    rnn = RNNGRUD(c, output_type='Binary', imput_method='multivar', max_len=120, num_hidden=35,
                     num_neurons=62, epochs=13, total_fold=10, current_fold=1,
                     drop_rates=[0.421, 0.584, 0.297, 0.297], reg_rate=0.208, es_patience=49, learning_rate=0.00024)

    explainer = shap.DeepExplainer(rnn.model, rnn.test_x)