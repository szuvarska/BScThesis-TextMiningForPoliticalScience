from shiny import reactive, render, ui
from App.utils import collapsible_section
from App.about_module.about_html import introduction_html, guide_html, authors_html

about_module_ui = ui.tags.div(
    ui.tags.div(
        collapsible_section(
            "Introduction",
            "toggle_introduction_button",
            "introduction_content"
        ),
        collapsible_section(
            "Guide",
            "toggle_guide_button",
            "guide_content"
        ),
        collapsible_section(
            "Authors",
            "toggle_authors_button",
            "authors_content"
        ),
        class_="main-left-container"
    ),
    class_="main-container about-module"
)


def about_module_server(input, output, session):
    introduction_visible = reactive.Value(True)
    guide_visible = reactive.Value(True)
    authors_visible = reactive.Value(True)

    @reactive.Effect
    @reactive.event(input.toggle_introduction_button)
    def toggle_introduction_visibility():
        introduction_visible.set(not introduction_visible.get())
        session.send_input_message("toggle_introduction_button", {"label": "⯆" if introduction_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_guide_button)
    def toggle_guide_visibility():
        guide_visible.set(not guide_visible.get())
        session.send_input_message("toggle_guide_button", {"label": "⯆" if guide_visible.get() else "⯈"})

    @reactive.Effect
    @reactive.event(input.toggle_authors_button)
    def toggle_authors_visibility():
        authors_visible.set(not authors_visible.get())
        session.send_input_message("toggle_authors_button", {"label": "⯆" if authors_visible.get() else "⯈"})

    @output
    @render.ui
    def introduction_content():
        if introduction_visible.get():
            return ui.HTML(introduction_html)
        return ui.div()

    @output
    @render.ui
    def guide_content():
        if guide_visible.get():
            return ui.HTML(guide_html)
        return ui.div()

    @output
    @render.ui
    def authors_content():
        if authors_visible.get():
            return ui.HTML(authors_html)
        return ui.div()
