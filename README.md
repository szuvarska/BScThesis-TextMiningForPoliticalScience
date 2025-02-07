# Web application for text mining to analyse press articles for political scientists
In this project, we introduce a user-friendly web application designed to analyse political news articles. The application is designed primarily for political scientists but can also benefit journalists and researchers interested in examining sentiment towards entities in text or tracking changes in the portrayal of topics over time. The analysis pipeline Exploratory Data Analysis, Named Entity Recognition with Entity Disambiguation, Sentiment Analysis, Community Detection, Topic Modelling, N-grams and Text Summarisation. The tool was developed using Python and the Shiny framework, incorporating primarily LLMs models e.g. BERT. Users can analyse predefined datasets related to the Russia-Ukraine War and the Israel-Palestine Conflict, or upload their own articles for examination. The application is available online within our Faculty network, providing an opportunity for broader academic collaboration. In addition, a blame and praise recognition model was developed but was not integrated into the application due to the limited availability of training data; however, preliminary results indicate promising potential for further research in this area.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/graphical_abstract.png" alt="Graphical abstract"/>
</p>

# Description of application
Our application "Global Times: Articles Analysis" is divided into four modes: `Single`, `Double`, `All` and `About`.

## Single Mode
The `Single Mode` allows users to analyse one article in detail. Users can select an article from the predefined datasets or upload their own. The right menu provides a dropdown to choose from available datasets, such as `Gaza before conflict`. Alternatively, users can upload a text file for analysis.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_single_selecting_article.png" alt="Selecting an article in Single Mode"/>
</p>

Once an article is selected, the analysis begins. A progress bar and loading animation indicate the status of the process.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_single_waiting_analysis.png" alt="Progress bar and loading animation during analysis"/>
</p>

Upon completion, the analysis displays the article's header, summary, and legends, along with highlighted entities in the text, using colours and icons to indicate sentiment. 

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_single_analysis_results.png" alt="Analysis results showing header, summary, and highlighted text"/>
</p>

Hovering over a highlighted entity reveals detailed information, including its type and sentiment. 

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_single_tooltip_entity.png" alt="Tooltip displaying entity details"/>
</p>

Interactive plots, such as sentiment distribution and word clouds, further illustrate key insights.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_single_plots.png" alt="Example plot in Single Mode"/>
</p>

## Double Mode
The `Double Mode` allows users to compare two articles side by side, enabling direct analysis of similarities and differences.

Users can upload one or both articles through the right menu.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_double_uploading_article.png" alt="Uploading an article in Double Mode"/>
</p>

Both articles are analysed independently, with results displayed side by side. Headers, text highlights, and legends are provided for each.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_double_side_by_side.png" alt="Double analysis results side by side"/>
</p>

Plots such as entity distribution comparison offer a visual representation of differences between the two articles

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_double_plot.png" alt="Comparative plot in Double Mode"/>
</p>

## All Mode
The `All Mode` provides an overview of entire datasets, offering powerful filters and visualisation options. An example plot showcases insights from a selected dataset.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_all_example_plot.png" alt="Example plot in All Mode"/>
</p>

Filters, such as selecting an entity type, dynamically update the plots.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_all_filter.png" alt="Plot updated with a select filter"/>
</p>

Users can examine specific word contexts using the concordance table.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_all_concordance_table.png" alt="Concordance table in All Mode"/>
</p>

Filters can also refine the concordance table, narrowing results to specific criteria.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_all_filtered_concordance.png" alt="Filtered concordance table"/>
</p>

The application provides loading animations during analysis and dynamically updates with new plots after uploading a dataset.

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_all_loading.png" alt="View during dataset analysis"/>
</p>

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_all_new_plots.png" alt="New plots generated after dataset upload"/>
</p>

## About Mode
The `About Mode` provides access to the user guide, helping users navigate the application effectively. 

<p align="center">
  <img src="https://github.com/szuvarska/BScThesis-TextMiningForPoliticalScience/blob/main/img/screenshot_about_user_guide.png" alt="Fragment of the user guide in About Mode"/>
</p>

# Installation Instructions
## Installing Docker (if not already installed)
Before running the application, install Docker: [Docker Desktop](https://www.docker.com/products/docker-desktop).

## Running the application
Open Terminal or Command Prompt:

1. **Download the Docker image:**

   ```sh
   docker pull szuvarska/globaltimesanalysis:latest
   ```
2. **Run the Docker container:**
   ```sh
   docker run -p 8000:8000 -it --rm szuvarska/globaltimesanalysis:latest
   ```
   This will map port 8000 on your local machine to port 8000 in the container.
3. **Access the application:**
   Open your web browser and go to the following address:
   ```sh
   http://127.0.0.1:8000/
   ```
## Stopping the application
To stop the container:
1. Press `CTRL+C` in the terminal where the container is running.
2. If needed, stop manually:
   ```sh
   docker ps  # List running containers
   docker stop <container-id>  # Replace with actual container ID
   ```   

# Team

## Authors
* Łukasz Grabarski ([@LukaszGrabarski](https://github.com/LukaszGrabarski))
* Marta Szuwarska ([@szuvarska](https://github.com/szuvarska))

## Supervisor
* Dr Anna Wróblewska ([@awroble](https://github.com/awroble))

## Co-supervisor
* Prof. Agnieszka Kaliska

## Co-operations:
* Prof. Anna Rudakowska
* Dr Daniel Dan


