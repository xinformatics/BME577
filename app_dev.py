from shiny import ui, render, App
import pandas as pd
import dill

app_ui = ui.page_fluid(
    ui.img(src="top_banner4.png", class_="sticky-md-top img-fluid"),
    ui.p(ui.panel_main(ui.p(ui.HTML("<marquee> Developer Side App </marquee>"))), class_="sticky-md-top"),
    ui.p(
    ui.output_table("disp_df"),
)
    )

def server(input, output, session):
    @output
    @render.table
    def disp_df():

        pkl_file = open("feed.pkl", "rb")

        df = dill.load(pkl_file)

        pkl_file.close()

        
        return df

app = App(app_ui, server, static_assets='/home/shashank/Desktop/arizona_sem01/BME577/temporal-shap-bme-577-master/mimic/pyshiny')
