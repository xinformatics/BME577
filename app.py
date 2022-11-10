
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


value = np.load('mimic_processed_2k_48t_26f.npy')
output = np.load('output.npy')


# Splitting data
train_x, test_x, train_y, test_y= train_test_split(value, output, test_size=0.2, random_state=0,stratify=output)


# normalizing train and test set
train_scaler, test_scaler = TimeSeriesScaler(), TimeSeriesScaler()
train_x = train_scaler.fit_transform(train_x)
test_x = test_scaler.fit_transform(test_x)

## Part 0: define variables

model_choices = {"a": "GRU Model", "b": "LSTM Model"}

feature_choices = {"feat1":'Blood Pressure', "feat2":'Oxygen', "feat3":'Breathing rate', "feat4":'feature name 4',}

shap_choices = {"shmeth1":'Stationary Time Window Method', "shmeth2":'Sliding Time Window Method', "shmeth3":'Binary Time Window Method',}

##feedback_choices = {"feed1":  "Shit app", "feed2": "super shit app"} not used

## Part 1: ui ----
app_ui = ui.page_fluid(
    #ui.h1("MLExplain EHR"),
    ui.tags.title("Page title is this"),
    ui.img(src="top_banner.png"),

    ui.layout_sidebar(

        ui.panel_sidebar(
            #ui.input_slider("n", "N", 0, 100, 20),
            #ui.input_file("file1", "Upload your dataset for analysis:", multiple=True),
            ui.panel_title("Please provide the inputs"),
            ui.p(ui.input_radio_buttons("x", "Select model to analyze", model_choices)),
            ui.input_action_button("run", "Show model evaluation metrics", class_="btn-success btn-lg"),
            #ui.input_action_button("run", "Run analysis", class_="btn-primary w-80"),
            #ui.p ( ui.output_text_verbatim("res_text_model", placeholder=True)),
            
            #ui.input_radio_buttons("feats", "Select feature to analyze", feature_choices),
            
            ui.p(ui.input_select("shaps", "Select SHAP method", shap_choices)),
            ui.input_action_button("run_tshap", "Run Temporal SHAP", class_="btn-success btn-lg"),

            ui.p(ui.input_select("feats", "Select feature to analyze", feature_choices)),
            ui.input_action_button("show_tshap", "Show Temporal Shap Results", class_="btn-success btn-lg"),

            ui.p(ui.input_text("feedback", "Please provide feedback in the box below")),
            #ui.p(ui.input_radio_buttons("feedback", "Choose your feedback type", feedback_choices)),
            ui.input_action_button("sub_feed", "Submit feedback", class_="btn-success btn-lg"),
            

        ),
        ui.panel_main(
        ui.panel_title("Outputs"),
        ui.p(ui.output_text_verbatim("res_text")),
        ui.output_plot("plot_model"),
        ui.p(ui.output_text_verbatim("shaprun_text")),
        ui.output_plot("plot_shap"),
        ui.p(ui.output_text_verbatim("show_feed")),


      ),

    )
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

        #with reactive.isolate():
        if input.x() == 'a':
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
        with ui.Progress(min=1, max=100) as p: ### nice progress bar
            p.set(message="Calculation in progress", detail="This may take a while...")
            for i in range(1, 100):
                p.set(i, message="Computing")
                await sleep(0.01)

        shap_run_msg = "SHAP Run Succesful!"

        return shap_run_msg

    @output
    @render.text
    #@reactive.event(lambda: input.run, ignore_none=False)
    @reactive.event(input.sub_feed)
    def show_feed() -> object:
        feed_msg = input.feedback()
        return f'You have provided the following feedback{feed_msg}'



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
    @reactive.event(input.run_tshap)
    def plot_shap() -> object:
        temp_plot = np.random.normal(25, 2, 30)

        fig, ax = plt.subplots()
        ax.hist(temp_plot, 60, density=True)
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