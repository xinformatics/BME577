from shiny import App, render, ui

app_ui = ui.page_fluid(
    #ui.h1("MLExplain EHR"),
    ui.img(src="top_banner.png"),
    #ui.input_slider("n", "N", 0, 100, 20),
    #ui.output_text_verbatim("txt"),
    ui.input_file("file1", "Upload your dataset for analysis:", multiple=True),
    ui.input_action_button("run", "Run analysis", class_="btn-primary w-100")
)


def server(input, output, session):
    @output
    @render.text
    def txt():
        return f"n**2 is {input.n() ** 2}"


app = App(app_ui, server,static_assets='/home/shashank/Desktop/arizona_sem01/BME577/sempro/shiny/')
