from shiny import ui
import os
from pathlib import Path
import glob
from App.single_module.single_analysis import analyse_single_article
from pathlib import Path

here = Path(__file__).parent.parent


def handle_file_upload(file_input):
    file_info = file_input()
    if file_info is None or len(file_info) == 0:
        return None, None
    try:
        with open(file_info[0]["datapath"], "r", encoding="utf-8", errors="ignore") as file:
            lines = file.readlines()
            if len(lines) < 7:
                return "Invalid file format", None
            published_time = lines[0].split(": ", 1)[1].strip()
            title = lines[1].split(": ", 1)[1].strip()
            categories = lines[2].split(";")
            category1 = categories[0].split(": ", 1)[1].strip()
            category2 = categories[1].split(": ", 1)[1].strip()
        return f"{title} -- {published_time} -- {category1} / {category2}", lines
    except UnicodeDecodeError:
        return "File encoding error", None


def list_files_in_folder(folder_path):
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                relative_path = os.path.relpath(os.path.join(root, file), folder_path)
                display_path = relative_path.replace('Articles_for_Agnieszka/', '').replace('Articles_for_Ania/', '')
                file_list.append((relative_path, display_path))
    return file_list


def analyze_file(file_path_or_lines, is_uploaded, progress_callback=None):
    """
    Blocking file analysis function that updates progress.
    Args:
        file_path_or_lines: Lines of file (uploaded) or file path (selected).
        is_uploaded: Whether the file is uploaded or selected.
        progress_callback: Callable for updating progress.
    Returns:
        Analysis results, entity sentiments, and sentiment sentences.
    """
    if is_uploaded:
        lines = file_path_or_lines
    else:
        with open(file_path_or_lines, "r") as file:
            lines = file.readlines()

    if not lines or len(lines) < 7:
        raise ValueError("Invalid file format or insufficient lines in the file.")

    # Start analysis
    if progress_callback:
        progress_callback(1, message="Reading and preparing file...")

    article_text = "\n".join(line.strip() for line in lines[7:])
    if progress_callback:
        progress_callback(3, message="Analyzing text...")

    # Call `analyse_single_article` directly (blocking)
    return analyse_single_article(article_text, progress_callback)


async def analyze_file_reactive(file_input, file_select, article_analysis, entity_sentiments, sentiment_sentences):
    if file_input() is not None:
        # Handle uploaded file
        _, lines = handle_file_upload(file_input)
        if not lines:
            ui.notification_show("Invalid uploaded file.", type="error")
            return
        file_path_or_lines = lines
        is_uploaded = True
    elif file_select() and file_select() != "None":
        # Handle selected file
        folder_path = here / "BRAT_Data"
        file_choices = list_files_in_folder(folder_path)
        selected_file = next(full for full, display in file_choices if display == file_select())
        file_path_or_lines = folder_path / selected_file
        is_uploaded = False
    else:
        # No file provided
        return

    with ui.Progress(min=1, max=100) as p:
        try:
            # Set progress at the beginning
            p.set(message="Starting analysis...")

            # Blocking function (ensure progress updates are in the main thread)
            analysis, entities, sentences = analyze_file(
                file_path_or_lines=file_path_or_lines,
                is_uploaded=is_uploaded,
                progress_callback=p.set  # Directly pass progress updates
            )
            # Store results
            article_analysis.set(analysis)
            entity_sentiments.set(entities)
            sentiment_sentences.set(sentences)
        except ValueError as e:
            ui.notification_show(str(e), type="error")


def render_uploaded_text_content(article_analysis, view_full_text, file_upload, file_select):
    if article_analysis.get():
        # If analysis is complete, display the results
        lines = article_analysis.get().split("<br>")
        if view_full_text.get():
            content_lines = lines
        else:
            content_lines = lines[:50]  # Show only the first 50 lines by default
        return ui.HTML("<br>".join(content_lines))
    elif file_upload() or file_select() != "None":
        # Show the progress bar while analysis is ongoing
        return ui.div(
            ui.div("Analysis in progress...", class_="progress-message"),
            class_="progress-container"
        )
    return "No file uploaded"


def generate_header(file_input, file_select, article_analysis, current_header):
    header, _ = handle_file_upload(file_input)
    if header:
        current_header.set(header)
        return header
    selected_file = file_select()
    if selected_file and selected_file != "None":
        folder_path = here / "BRAT_Data"
        file_choices = list_files_in_folder(folder_path)
        selected_file_full_path = next(full for full, display in file_choices if display == selected_file)
        file_path = folder_path / selected_file_full_path
        with open(file_path, "r") as file:
            lines = file.readlines()
            if len(lines) >= 7:
                display_header = (f"{lines[1].split(': ', 1)[1].strip()} -- "
                                  f"{lines[0].split(': ', 1)[1].strip()} -- "
                                  f"{lines[2].split(';')[0].split(': ', 1)[1].strip()} / "
                                  f"{lines[2].split(';')[1].split(': ', 1)[1].strip()}")
                current_header.set(display_header)
                return display_header
    if not article_analysis():
        return "No file uploaded"
    return current_header()


def remove_png_files():
    png_files = glob.glob('App/www/*.png')
    for file in png_files:
        if 'logomini.png' not in file:
            os.remove(file)


def collapsible_section(header, button_id, plot_id):
    return ui.div(
        ui.div(
            ui.tags.h3(header, style="display: inline;"),
            ui.input_action_button(button_id, "⯆", class_="toggle-button",
                                   style="font-size: 20px; display: inline; margin-left: 10px;"),
            style="display: flex; align-items: center; margin-bottom: 10px;",
            class_="collapsible-section-header"
        ),
        ui.output_ui(plot_id),
        class_="collapsible-section"
    )
