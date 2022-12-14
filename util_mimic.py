import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score, cohen_kappa_score
from scipy.stats import kendalltau
import pandas as pd
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Layer
from sklearn.preprocessing import OneHotEncoder

OUTPUT_PATH = 'output/'
FIG_PATH = 'figures/'
DATA_PATH = 'data/'

ALL_VAR_MAP = {'EDPlatelet': 'ED measured Platelet',
 'EDWbc': 'ED measured WBC',
 'EDComplEventHypoxia': 'Complication Hypoxia',
 'EDComplEventHypotension': 'Complication Hypotension',
 'EDComplEventSeizures': 'Complication Seizures',
 'EDComplEventCardArr': 'Complication CardArr',
 'EDInr': 'ED measured INR',
 'PresIntubation':'ED intubated',
 'EDBun': 'ED measured BUN',
 'EDCreatinine': 'ED measured Creatinine',
 'EDGlucose': 'ED measured Glucose',
 'EDCO2': 'ED measured CO2',
 'EDChloride': 'ED measured Chloride',
 'EDPotassium': 'ED measured Potassium',
 'EDSodium': 'ED measured Sodium',
 'EDDischSpO2': 'ED Discharge measured SPO2',
 'EDArrSBP': 'ED Arrival measured SBP',
 'EDArrHR': 'ED Arrival measured HR',
 'EDArrSpO2': 'ED Arrival measured SPO2',
 'EDDischSBP': 'ED Arrival measured SBP',
 'EDDischDBP': 'ED Arrival measured DBP',
 'EDDischHR': 'ED Discharge measured HR',
 'EDArrDBP': 'ED Arrival DBP',
 'EDDrugScreenCocaine': 'ED Drug Screen: Cocaine',
 'EDDrugScreenAmph': 'ED Drug Screen: AMPH',
 'EDDrugScreenPCP': 'ED Drug Screen: PCP',
 'EDDrugScreenCannabis': 'ED Drug Screen: Cannabis',
 'EDDrugScreenMethadone': 'ED Drug Screen: Methadone',
 'EDHemoglobin': 'ED measured Hemoglobin',
 'EDDrugScreenBenzo': 'ED Drug Screen: Benzo',
 'EDDrugScreenBarb': 'ED Drug Screen: Barb',
 'EDDrugScreenOpioids': 'ED Drug Screen: Opioids',
 'AgeRecodedPHI': 'Age',
 'PatientType': 'ICU addmission status',
 'Sex': 'Sex=Female',
 'GCS': 'ED measured total GCS',
 'mGCS': 'ED numeric measured GCS Motor',
 'GcsEDArrMotor_1-No Response': 'ED measured GCS Motor=No Response',
 'GcsEDArrMotor_2-Extension': 'ED measured GCS Motor=Extension',
 'GcsEDArrMotor_3-Flexion Abnormal': 'ED measured GCS Motor=Flexion Abnormal',
 'GcsEDArrMotor_4-Flexion Withdrawal': 'ED measured GCS Motor=Flexion Withdrawal',
 'GcsEDArrMotor_5-Localizes to Pain': 'ED measured GCS Motor=Localize to Pain',
 'GcsEDArrMotor_6-Obeys Commands': 'ED measured GCS Motor=Obeys Command',
 'GcsEDArrMotor_P-Untestable (Paralyzed)': 'ED measured GCS Motor=Untestable (Paralyzed)',
 'EDComplEventAsp_No': 'Complication ASP=No',
 'EDComplEventAsp_Suspected': 'Complication ASP=Suspected',
 'EDComplEventAsp_Yes': 'Complication ASP=Yes',
 'Pupils_Both': 'ED Pupils Reactivity=Both',
 'Pupils_Missing at least one eye': 'ED Pupils Reactivity=Missing at least one eye',
 'Pupils_Neither': 'ED Pupils Reactivity=Neither',
 'Pupils_One': 'ED Pupils Reactivity=One',
 'DLAlatSgpt': 'Daily Lab AlatSgpt',
 'DLAlkalinePhosphatase': 'Daily Lab AlkalinePhosphatase',
 'DLAmylase': 'Daily Lab Amylase',
 'DLAsatSgot': 'Daily Lab AsatSgot',
 'DLBd': 'Daily Lab Bd',
 'DLBe': 'Daily Lab Be',
 'DLBicarbonate': 'Daily Lab Bicarbonate',
 'DLCalcium': 'Daily Lab Calcium',
 'DLCreatinine': 'Daily Lab Creatinine',
 'DLEosinophils': 'Daily Lab Eosinophils',
 'DLFibrinogen': 'Daily Lab Fibrinogen',
 'DLGlucose': 'Daily Lab Glucose',
 'DLHematocrit': 'Daily Lab Hematocrit',
 'DLHemoglobin': 'Daily Lab Hemoglobin',
 'DLInr': 'Daily Lab Inr',
 'DLLactate': 'Daily Lab Lactate',
 'DLLdh': 'Daily Lab Ldh',
 'DLLymphocytes': 'Daily Lab Lymphocytes',
 'DLMagnesium': 'Daily Lab Magnesium',
 'DLNeutrophils': 'Daily Lab Neutrophils',
 'DLOtherWbc': 'Daily Lab OtherWbc',
 'DLPaCO2': 'Daily Lab PaCO2',
 'DLPaO2': 'Daily Lab PaO2',
 'DLPh': 'Daily Lab Ph',
 'DLPlatelet': 'Daily Lab Platelet',
 'DLPotassium': 'Daily Lab Potassium',
 'DLProthrombineTime': 'Daily Lab Prothrombine Time',
 'DLSodium': 'Daily Lab Sodium',
 'DLTotalBilirubin': 'Daily Lab Total Bilirubin',
 'DLUrea': 'Daily Lab Urea',
 'DLWbc': 'Daily Lab Wbc',
 'DLaPtt': 'Daily Lab aPtt',
 'DVDBP': 'Daily Vitals DBP',
 'DVHR': 'Daily Vitals HR',
 'DVResTyp_Intubation': 'Daily Respiratory=Intubated',
 'DVResTyp_NIPPV': 'Daily Respiratory=NIPPV',
 'DVResTyp_Non-Intubated': 'Daily Respiratory=NonItubated',
 'DVSBP': 'Daily Vitals SBP',
 'DVSpO2': 'Daily Vitals SPO2',
 'DvTemp': 'Daily Vitals Temperature',
 'GCSEye_1-No Response': 'Daily GCS Eye=No Response',
 'GCSEye_2-To Pain': 'Daily GCS Eye=To Pain',
 'GCSEye_3-To Verbal Command': 'Daily GCS Eye=To Verbal Command',
 'GCSEye_4-Spontaneously': 'Daily GCS Eye=Spontaneously',
 'GCSEye_S-Untestable (Swollen)': 'Daily GCS Eye=Untestable',
 'GCSMtr_1-No Response': 'Daily GCS Motor=No Response',
 'GCSMtr_2-Extension': 'Daily GCS Motor=Extension',
 'GCSMtr_3-Flexion Abnormal': 'Daily GCS Motor=Flexion Abnormal',
 'GCSMtr_4-Flexion Withdrawal': 'Daily GCS Motor=Flexion Withdrawal',
 'GCSMtr_5-Localizes to Pain': 'Daily GCS Motor=Localize to Pain',
 'GCSMtr_6-Obeys Commands': 'Daily GCS Motor=Obeys Commands',
 'GCSMtr_P-Untestable (Paralyzed)': 'Daily GCS Motor=Untestable',
 'GCSVrb_1-No Response': 'Daily GCS Verbal=No Response',
 'GCSVrb_2-Incomprehensible Sounds': 'Daily GCS Verbal=Incomprehensible Sounds',
 'GCSVrb_3-Inappropriate Words': 'Daily GCS Verbal=Inappropriate Words',
 'GCSVrb_4-Disoriented & Converses': 'Daily GCS Verbal=Disoriented & Converses',
 'GCSVrb_5-Oriented & Converses': 'Daily GCS Verbal=Oriented & Converses',
 'GCSVrb_T-Untestable (Artificial Airway)': 'Daily GCS Verbal=Untestable',
 'PupilReactivity_Both': 'Daily Pupil Reactivity=Both',
 'PupilReactivity_Neither': 'Daily Pupil Reactivity=Neither',
 'PupilReactivity_One': 'Daily Pupil Reactivity=One',
 'PupilReactivity_Untestable': 'Daily Pupil Reactivity=Untestable'}

def inv_list(l):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i
    return d


def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def read_output(model_type='allReg', output_type='Regression', max_len=40,
                imput_method='mean', epochs=10, num_neurons=20, total_fold=None, current_fold=1):
    if model_type == 'allReg' or model_type == 'impact':
        save_name = 'output_' + model_type + '_type_' + output_type + '_imput_' + imput_method + '_' + \
                    str(total_fold) + 'Fold_' + str(current_fold)
    elif model_type == 'grud' or model_type == 'rnn':
        save_name = 'output_' + model_type + '_type_' + output_type + '_maxLen_' + str(max_len) + '_imput_' + \
                    imput_method + '_epochs_' + str(epochs) + '_units_' + str(num_neurons) + '_' +\
                                str(total_fold) + 'Fold_' + str(current_fold)
    elif model_type == 'ann':
        save_name = 'output_' + model_type + '_type_' + output_type + '_imput_' + \
                    imput_method + '_epochs_' + str(epochs) + '_' + \
                    str(total_fold) + 'Fold_' + str(current_fold)

    with open(OUTPUT_PATH + save_name, 'rb') as f:
        saveObject = pickle.load(f)
    return saveObject


def evaluation(y_true, y_predict, metric, output_type=None):

    y_true = reverse_output_builder(y_true, output_type=output_type)
    y_predict = reverse_output_builder(y_predict, output_type=output_type)

    # Calculating the performance measurements
    if metric == 'MSE':
        metric_value = mean_squared_error(y_true, y_predict, squared=False)
    elif metric == 'AMSE':
        train_count = [np.count_nonzero(y_true == i) for i in range(9)]
        train_weights = [1 / train_count[i] for i in y_true]
        metric_value = mean_squared_error(y_true, y_predict, squared=False, sample_weight=train_weights)
    elif metric == 'ACC':
        metric_value = accuracy_score(y_true, y_predict)
    elif metric == 'AACC':
        train_count = [np.count_nonzero(y_true == i) for i in range(9)]
        train_weights = [1 / train_count[i] for i in y_true]
        metric_value = accuracy_score(y_true, y_predict, sample_weight=train_weights)
    elif metric == 'Kendall':
        metric_value, _ = kendalltau(y_true, y_predict)
    elif metric == 'AUC':
        if output_type == 'Binary':
            metric_value = roc_auc_score(y_true, y_predict, average='macro')
        else:
            enc = OneHotEncoder(categories=[[1, 2, 3, 4, 5, 6, 7, 8]])
            enc.fit(y_true.reshape(-1, 1))
            y_true = enc.transform(y_true.reshape(-1, 1)).toarray()
            enc.fit(y_predict.reshape(-1, 1))
            y_predict = enc.transform(y_predict.reshape(-1, 1)).toarray()
            metric_value = roc_auc_score(y_true, y_predict, average='micro')
    elif metric == 'F1':
        metric_value = f1_score(y_true, y_predict, average='weighted')
    elif metric == 'Kappa':
        train_count = [np.count_nonzero(y_true == i) for i in range(9)]
        train_weights = [1 / train_count[i] for i in y_true]
        metric_value = cohen_kappa_score(y_true, y_predict, sample_weight=train_weights)

    else:
        raise

    return metric_value

def read_and_evaluation(output_type='Multiclass', metric='MSE', imput_method='mean', verbose=False, **kwargs):
    output = read_output(output_type=output_type, imput_method=imput_method, **kwargs)
    train_y, train_predict, test_y, test_predict = output['train_y'], output['train_predict'], output['test_y'], output['test_predict']

    train_measure = evaluation(train_y, train_predict, output_type=output_type, metric=metric)
    test_measure = evaluation(test_y, test_predict, output_type=output_type, metric=metric)
    
    if verbose:
        print('##########################')
        print('Output Type: ' + output_type + ', Imputation Type: ' + imput_method)
        print('Train ' + metric + ': ' + str(train_measure))
        print('Test ' + metric + ': ' + str(test_measure))

    return train_measure, test_measure


def metric_fun(output_type, metric):
    def temp_fun(y_true, y_predict):
        return evaluation(y_true=y_true, y_predict=y_predict, output_type=output_type, metric=metric)

    return temp_fun


class Attention(Layer):
    def __init__(self, attn_dim):
        self.attn_dim = attn_dim
        super(Attention, self).__init__()



    def build(self, input_shape):
        self.emb_size = input_shape[-1]

        self.W = self.add_weight(shape=(self.emb_size, self.attn_dim), name='Att_W', initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(1, 1, self.attn_dim), name='Att_b', initializer='zeros', trainable=True)
        self.u = self.add_weight(shape=(self.attn_dim, 1), name='Att_u', initializer='glorot_uniform', trainable=True)
        super(Attention, self).build(input_shape)


    def call(self, x, mask=None):
        emb = x
        # mask = K.cast(mask, K.floatx())
        attn_weights = K.dot(K.tanh(K.dot(emb, self.W) + self.b), self.u)
        # mask = K.expand_dims(mask, axis=-1)
        # attn_weights = mask * attn_weights + (1 - mask) * mask_value
        attn_weights = K.softmax(attn_weights, axis=-2)
        return attn_weights


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        return mask


def output_builder(outcomes, output_type, num_output=8):
    """
    Converting integer levels to the output that is readable for the prediction model
    :param outcomes:
    :param output_type:
    :param num_output:
    :return:
    """
    if output_type == 'Binary':
        num_output = 1
        outcomes = np.floor((outcomes - 1) / 4).astype(int)
    elif output_type == 'Multiclass':
        temp = np.zeros((len(outcomes), num_output))
        temp[np.arange(len(outcomes)), outcomes - 1] = 1
        outcomes = temp
    elif output_type == 'OrdinalMulticlass':
        num_output = num_output - 1
        temp = np.zeros((len(outcomes), num_output))
        for i in range(len(temp)):
            temp[i, 0:(outcomes[i] - 1)] = [1] * (outcomes[i] - 1)
        outcomes = temp
    elif output_type == 'Regression':
        num_output = 1
    return outcomes, num_output


def reverse_output_builder(y, output_type):
    """
    Converting the output from the prediction model to integer levels
    :param y:
    :param output_type:
    :return:
    """

    # if isinstance(y_true, pd.Series):
    #     y_true = np.array(y_true.tolist()).astype(int)
    #     y_predict = np.array(y_predict.tolist()).astype(int)
    #
    # # Converting the predicted values to the class labels
    # if output_type == 'Regression':
    #     y_predict = np.around(y_predict)
    # elif output_type == 'Multiclass':
    #     y_true = np.array([np.where(r == 1)[0][0] for r in y_true]) + 1
    #     y_predict = np.argmax(y_predict, axis=-1) + 1
    # elif output_type == 'OrdinalMulticlass':
    #     y_true = np.array([max(np.append([-1], np.where(i > 0.5)[0]) + 2) for i in y_true])
    #     y_predict = np.array([max(np.append([-1], np.where(i > 0.5)[0]) + 2) for i in y_predict])

    if isinstance(y, pd.Series):
        y = np.array(y.tolist()).astype(int)

    # Converting the predicted values to the class labels
    if output_type == 'Regression':
        y = np.around(y).astype(int)
    elif output_type == 'Multiclass':
        y = np.argmax(y, axis=-1) + 1
    elif output_type == 'OrdinalMulticlass':
        # y = np.array([max(np.append([-1], np.where(i > 0.5)[0]) + 2) for i in y])
        y = np.array([len(np.where(i > 0.5)[0]) + 1 for i in y])
    elif output_type == 'Binary':
        y = np.around(y).astype(int)
    return y


def sample_weight_builder(y):
    elements, counts = np.unique(y.squeeze(), return_counts=True)
    class_weights = {elements[i]: 1 / counts[i] for i in range(len(elements))}
    sample_weights = np.array([class_weights[y[i]] for i in range(len(y))])
    return sample_weights


if __name__ == '__main__':
    model_type = 'impact'
    output_type = 'Regression'
    metrics = ['AUC']
    for metric in metrics:
        test_evals = []
        train_evals = []
        for j in range(8):
            train_val, test_val = read_and_evaluation(model_type=model_type, output_type=output_type, metric=metric,
                                                      imput_method='multivar', max_len=120, epochs=1381, num_neurons=62,
                                                      total_fold=10, current_fold=j + 1)
            train_evals.append(train_val)
            test_evals.append(test_val)
        print('{} Metric ###########'.format(metric))
        #     print(np.mean(train_evals))
        print(np.mean(test_evals))
        # print(2 * sem(test_evals))