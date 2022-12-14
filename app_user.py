from shiny import App,  reactive, render, ui
from tensorflow.keras import backend as K
from sklearn.metrics import *
from tensorflow import keras
import pandas as pd
from asyncio import sleep
import dill
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from util import TimeSeriesScaler
import matplotlib.pyplot as plt
from sklearn import metrics

from heatmap import heat_map, heat_map_all_features 


########### addtional path addition for SHAP util
#import sys
#sys.path.insert(1, '/home/shashank/Desktop/arizona_sem01/BME577/temporal-shap-bme-577-master/track_project')

from shap_util import *
import matplotlib as mpl
import shap


shap.initjs()
#################### path added



#### workaround for feedback display
pkl_file = open("feed.pkl", "rb")
df_feed = dill.load(pkl_file)
pkl_file.close()




### data added
value = np.load('mimic_processed_2k_48t_26f.npy')
output = np.load('output.npy')


# Splitting data
train_x, test_x, train_y, test_y= train_test_split(value, output, test_size=0.2, random_state=0,stratify=output)


# normalizing train and test set
train_scaler, test_scaler = TimeSeriesScaler(), TimeSeriesScaler()
train_x = train_scaler.fit_transform(train_x)
test_x = test_scaler.fit_transform(test_x)

## Part 0: define variables

model_choices = {"gru": "GRU  (Gated Recurrent Unit) Model", "lstm": "LSTM (Long-Short Term Memory) Model"}

shap_choices = {"shmeth1":'Stationary Time Window Method', "shmeth2":'Sliding Time Window Method', "shmeth3":'Binary Time Window Method'}

interpret_mode = {"glb" : "All features together", "loc" : "One feature at a time"}

feature_map_mimic_list = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
                     'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT',
                     'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate', 'SysBP', 'DiasBP', 'MeanBP', 'RespRate', 'SpO2',
                     'Glucose', 'Temp', 'gender','age','ethnicity','first_icu_stay'] ## 32 

feature_map_mimic ={"anion_gap":   "ANION GAP",  "albumin" :    "ALBUMIN", "bicarbonate": "BICARBONATE", "bilirubin":   "BILIRUBIN", 
                    "creatinine":  "CREATININE", "chloride" :   "CHLORIDE","glucose" :    "GLUCOSE", "hematocrit":  "HEMATOCRIT", 
                    "hemoglobin" : "HEMOGLOBIN", "lactate":     "LACTATE", "magnesium":   "MAGNESIUM", "phosphate":   "PHOSPHATE", 
                    "platelet" :   "PLATELET", "potassium":   "POTASSIUM","ptt": "PTT", "inr":  "INR", 
                    "pt":          "PT", "sodium" :     "SODIUM", "bun" :        "BUN", "wbc" :        "WBC", 
                    "heartrate" :  "HeartRate", "sysbp"     :  "SysBP", "diasbp" :     "DiasBP", "meanbp":      "MeanBP", 
                    "resprate" :   "RespRate", "spo2":        "SpO2", "glucose1":"Glucose"}


num_feature_map_mimic ={"anion_gap": 1,"albumin" :2,"bicarbonate": 3, "bilirubin":   4, "creatinine":  5, "chloride" :   6, "glucose" :    7, "hematocrit":  8, 
"hemoglobin" : 9, "lactate":     10, "magnesium":   11, "phosphate":   12, "platelet" :   13, "potassium":  14,"ptt": 15,"inr": 16, "pt": 17, 
"sodium" :     18, "bun" :  19, "wbc" : 20, "heartrate" :  21, "sysbp" :  22, "diasbp" :     23, "meanbp":      24, "resprate" :   25, "spo2": 26}#"glucose1": 27}

#feature_choices = {"feat1":'Blood Pressure', "feat2":'Oxygen', "feat3":'Breathing rate', "feat4":'feature name 4',}



##feedback_choices = {"feed1":  "Shit app", "feed2": "super shit app"} not used

## Part 1: ui ----
app_ui = ui.page_fluid(
    #ui.h1("MLExplain EHR"),
    #ui.tags.title("Page title is this"),
    ui.img(src="top_banner4.png", class_="sticky-md-top img-fluid"),
    #ui.p(ui.panel_main(ui.p(ui.HTML("<marquee> Developed for the BME577 course project by Shashank Yadav and Khuong Duy Mac. </marquee>"))), class_="sticky-md-top"),

    ui.layout_sidebar(

        ui.panel_sidebar(
            ui.p(ui.p(ui.panel_title("Please provide the inputs")),
            ui.p(ui.input_radio_buttons("x", "Select model to analyze", model_choices)),
            ui.input_action_button("run", "Load model and display model evaluation metrics", class_="btn-success btn-lg")),
            
            ui.p(ui.input_radio_buttons("shaps", "Select SHAP Method", shap_choices)),
            ui.input_action_button("run_tshap", "Run Temporal SHAP", class_="btn-success btn-lg"),

            ui.p(ui.input_radio_buttons("feats", "Select interpretation mode", interpret_mode)),

            ui.panel_conditional("input.feats === 'loc'",
                #ui.input_radio_buttons("feat_select", "Select the feature", feature_map_mimic),
                ui.input_select("feat_select", "Select the feature", feature_map_mimic),
            ),

            ui.input_action_button("show_tshap", "Show Temporal Shap Results", class_="btn-success btn-lg"),


            ui.p(ui.input_text("feedback", "Please provide feedback in the box below and rate this web UI")),
            #i.p(ui.input_slider("user_rating", "Rating", value=5, min=1, max=5, step=1)),
            #ui.p(ui.input_radio_buttons("feedback", "Choose your feedback type", feedback_choices)),
            ui.p(ui.input_slider("user_rating", "Rating", value=5, min=1, max=5, step=0.5)),
            ui.p(ui.input_text("user_name", "Your name: ")),
            ui.input_action_button("sub_feed", "Submit the rating and feedback", class_="btn-success btn-lg"),

            #ui.p(ui.panel_main(ui.p("Developed for the BME577 course project by Shashank Yadav and Khuong Duy Mac"))),
            ui.p(ui.panel_main(ui.p(ui.HTML("<marquee> Developed for the BME577 course project by Shashank Yadav and Khuong Duy Mac. </marquee>")))),
            

        width=4),
        ui.panel_sidebar(
        ui.panel_title("Outputs"),
        ui.p(ui.output_text_verbatim("res_text")),
        ui.output_plot("plot_model"),
        ui.p(ui.output_text_verbatim("shaprun_text")),
        ui.output_plot("plot_shap"),
        ui.p(ui.output_text_verbatim("show_feed"),
        #ui.HTML("<marquee> Developed for the BME577 course project by Shashank Yadav and Khuong Duy Mac </marquee>"),
        #ui.p(ui.panel_main(ui.p(ui.HTML("<marquee> Developed for the BME577 course project by Shashank Yadav and Khuong Duy Mac. </marquee>")))),
        ),


      width=8),



    ),
)

## Part 2: server ----
def server(input, output, session):


    # @reactive.Calc
    # def run():
    #     if input.x() == 'a':
    #         model = keras.models.load_model('temporal_GRU_100epochs.hdf5')
    #     else:
    #         model = keras.models.load_model('temporal_LSTM_37epochs_ES.hdf5')

    #     return f'Number of model parameters: "{model.count_params()}"'
    # @output
    # @render.text
    # @reactive.event(input.run)
    # async def res_text_model():
    #     return f'You have selected the "{model_choices[input.x()]}"'

    @output
    @render.text
    #@reactive.event(lambda: input.run, ignore_none=False)
    @reactive.event(input.run)
    async def res_text():
        #input.run()
        global model
        #with reactive.isolate():
        if input.x() == 'gru':
            model = keras.models.load_model('temporal_GRU_100epochs.hdf5')
        else:
            model = keras.models.load_model('temporal_LSTM_37epochs_ES.hdf5')

        with ui.Progress(min=1, max=100) as p: ### nice progress bar
            p.set(message="Calculation in progress", detail="This may take a while...")
            for i in range(1, 100):
                p.set(i, message="Computing")
                await sleep(0.01)
        metrics_1 = roc_auc_score(test_y, model.predict(test_x, verbose=0))
        metrics_2 = average_precision_score(test_y, model.predict(test_x, verbose=0))
        metrics_3 = f1_score(test_y, np.round(model.predict(test_x,verbose=0)))
        metrics_4 = recall_score(test_y, np.round(model.predict(test_x,verbose=0)))
        metrics_5 = precision_score(test_y, np.round(model.predict(test_x,verbose=0)))


        ##### for the auc roc plot
        global fpr, tpr, roc_auc_val, precision_inte, recall_inte, pr_val

        fpr, tpr, threshold_roc = metrics.roc_curve(test_y, model.predict(test_x, verbose=0))
        roc_auc_val = metrics.auc(fpr, tpr)


        precision_inte, recall_inte, threshold_pr = metrics.precision_recall_curve(test_y, model.predict(test_x, verbose=0))
        pr_val = metrics.auc(recall_inte, precision_inte)

        
        #return f'Number of model parameters: "{model.count_params()}"'
        return f'You have selected the "{model_choices[input.x()]}" with \n' + f'test AUC-ROC Score: {round(metrics_1,3)} \n' + \
        f'test PR Score: {round(metrics_2,3)} \n' + f'test F1 Score: {round(metrics_3,3)} \n' + f'Recall: {round(metrics_4,3)} \n' + f'Precision: {round(metrics_5,3)} \n'

    
    @output
    @render.text
    @reactive.event(input.run_tshap)
    async def shaprun_text():

        global num_background, index, background_ts, test_ts

        num_background = 5
        index = 0
        background_ts, test_ts = train_x[:num_background], test_x[index:index + 5]

        with ui.Progress(min=1, max=len(test_ts)) as p: ### nice progress bar
            p.set(message="Calculation in progress", detail="This may take a while...")
            #for i in range(1, 100):
                #p.set(i, message="Computing")
                #await sleep(0.01)

            global ts_phi_
            ts_phi_ = np.zeros((len(test_ts),test_ts.shape[1], test_ts.shape[2]))

            try:

                if input.shaps() == "shmeth1":
                    ### stationary time window shap
                    #ts_phi_1 = np.zeros((len(test_ts),test_ts.shape[1], test_ts.shape[2]))
                    print('Sta_TW')
                    for i in range(len(test_ts)):
                        window_len = 15
                        gtw = StationaryTimeWindow(model, window_len, B_ts=background_ts, test_ts=test_ts[i:i+1], model_type='lstm')
                        p.set(i, message="Computing")
                        #ts_phi_1[i,:,:] = gtw.shap_values()[0]
                        ts_phi_[i,:,:] = gtw.shap_values()[0]

                elif input.shaps() == "shmeth2":
                    ### sliding time window shap
                    #ts_phi_2 = np.zeros((len(test_ts),test_ts.shape[1], test_ts.shape[2]))
                    print('Sli_TW')
                    for i in range(len(test_ts)):
                        window_len = 20
                        stride = 10
                        stw = SlidingTimeWindow(model, stride, window_len, background_ts, test_ts[i:i+1], model_type='lstm')
                        p.set(i, message="Computing")
                        #ts_phi_2[i,:,:] = stw.shap_values()[0]
                        ts_phi_[i,:,:] = stw.shap_values()[0]
                        
                elif input.shaps() == "shmeth3":
                    ### binary time window
                    #ts_phi_3 = np.zeros((len(test_ts),test_ts.shape[1], test_ts.shape[2]))
                    print('B_TW')
                    for i in range(len(test_ts)):
                        delta = 0.01
                        n_w = 20
                        btw = BinaryTimeWindow(model, delta, n_w, background_ts, test_ts[i:i+1], model_type='lstm')
                        p.set(i, message="Computing")
                        #ts_phi_3[i,:,:] = btw.shap_values(nsamples_in_loop='auto')[0]
                        ts_phi_[i,:,:] = btw.shap_values(nsamples_in_loop='auto')[0]

                shap_run_msg = f'You have provided the {shap_choices[input.shaps()]}' + " SHAP Method. Run Succesful!"

            except:
                shap_run_msg = "Please run the first step with model evaluation"

        return shap_run_msg

    @output
    @render.text
    #@reactive.event(lambda: input.run, ignore_none=False)
    @reactive.event(input.sub_feed)
    def show_feed() -> object:
        feed_msg = input.feedback()
        uname = input.user_name()
        star = input.user_rating()

        df_feed.loc[len(df_feed.index)] = [uname, feed_msg, star]
        df_feed.to_pickle("./feed.pkl")

        return f'Hi {uname}, You have provided the following feedback : {feed_msg} + {star}'
        #return f'You have provided the following feedback{feed_msg}'



    @output
    @render.plot()
    @reactive.event(input.run)
    def plot_model() -> object:
        temp_plot = np.random.normal(55, 5, 30)

        fig, ax = plt.subplots(1,2)
        #ax[0].hist(temp_plot, 20, density=True)
        ax[0].set_title('Receiver Operating Characteristic')
        ax[0].plot(fpr, tpr, 'r', label = 'AUC = %0.2f' % roc_auc_val)
        #ax[0].legend(loc = 'lower right')
        ax[0].plot([0, 1], [0, 1],'r--')
        #ax[0].xlim([0, 1])
        #ax[0].ylim([0, 1])
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_xlabel('False Positive Rate')



        #### pr curve
        #ax[1].hist(temp_plot, 20, density=True)
        ax[1].set_title('Precision-Recall Curve')
        ax[1].plot(recall_inte, precision_inte, 'b', label = 'AUC = %0.2f' % pr_val)
        #ax[1].legend(loc = 'lower right')
        ax[1].plot([1, 0], [0, 1],'r--')
        #ax[1].xlim([0, 1])
        #ax[1].ylim([0, 1])
        ax[1].set_ylabel('Precision')
        ax[1].set_xlabel('Recall')


        return fig

    @output
    @render.plot()
    @reactive.event(input.show_tshap)
    def plot_shap() -> object:
        print(input.feats())

        #%matplotlib inline
        phi_index = 0

        # try:
        #     if input.feats() == "glb":
        #         fig = heat_map_all_features(0, 48, ts_phi_[phi_index, :, :], num_feature=40, var_name=feature_map_mimic_list, plot_type='heat_abs')
        #     else:

        #         fname = input.feat_select()
        #         print(fname)

        #         #temp_plot = np.random.normal(25, 2, 30)
        #         #fig, ax = plt.subplots()
        #         #ax.hist(temp_plot, 60, density=True)
        #         #title = 'blah   N = 241, ' + 'standard deviation =' + str(sigma)
        #         #ax.set_title("This plot is for the " + fname)

        #         var = feature_map_mimic[fname]
        #         # # var_ind = var_to_ind[var]
        #         var_ind = num_feature_map_mimic[fname] - 1
        #         label = 'Unfavorable' if model.predict(test_x[index][np.newaxis,:])>0.5 else 'Favorable'
        #         org_ts = test_scaler.inverse_transfrom(test_ts[phi_index:phi_index + 1])
        #         fig = heat_map(0, 48, org_ts[0, :, var_ind], ts_phi_[phi_index, :, var_ind], var_name=var, plot_type='bar')
        # except:
            
        #     shap_run_msg = "Please run the first two steps"

        #     return shap_run_msg


        if input.feats() == "glb":
            fig = heat_map_all_features(0, 48, ts_phi_[phi_index, :, :], num_feature=40, var_name=feature_map_mimic_list, plot_type='heat_abs')
        else:

            fname = input.feat_select()
            print(fname)

            #temp_plot = np.random.normal(25, 2, 30)
            #fig, ax = plt.subplots()
            #ax.hist(temp_plot, 60, density=True)
            #title = 'blah   N = 241, ' + 'standard deviation =' + str(sigma)
            #ax.set_title("This plot is for the " + fname)

            var = feature_map_mimic[fname]
            # # var_ind = var_to_ind[var]
            var_ind = num_feature_map_mimic[fname] - 1
            label = 'Unfavorable' if model.predict(test_x[index][np.newaxis,:])>0.5 else 'Favorable'
            org_ts = test_scaler.inverse_transfrom(test_ts[phi_index:phi_index + 1])
            fig = heat_map(0, 48, org_ts[0, :, var_ind], ts_phi_[phi_index, :, var_ind], var_name=var, plot_type='bar')

        return fig


    # @output
    # @render.plot()
    # @reactive.event(input.sub_feed)
    # def show_feed() -> object:

    #     return input.feedback()






## Combine into a shiny app.
## Note that the variable must be "app".
app = App(app_ui, server,static_assets='/home/shashank/Desktop/arizona_sem01/BME577/temporal-shap-bme-577-master/mimic/pyshiny')


################### useful code commented

    # @output
    # @render.text
    # def res_text():
    #     #file = pd.read_csv(input.file1())
    #     #input.run()
    #     model = keras.models.load_model('temporal_GRU_100epochs.hdf5')
    #     #size = file.shape
    #     # if input.file1() is None:
    #     #     return "Please upload a csv file"
    #     # f: list[FileInfo] = input.file1()
    #     # data = pd.read_csv(f[0]["datapath"])

    #     return model.summary()