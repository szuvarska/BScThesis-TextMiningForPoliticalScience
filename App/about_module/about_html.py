introduction_html = """
<div class="collapsible-section-content">
    <p>The "Global Times: Articles Analysis" app is a tool designed to analyze articles mainly from the Global Times, a Chinese tabloid newspaper. The app allows users to explore trends and sentiments in articles related to international conflicts, specifically in Gaza and Ukraine. The app offers different analysis views and visualizations, focusing on aspects such as sentiment, named entities, and more. It also includes tools for comparing articles and performing broader analyses across datasets.</p>

    <p><i>Global Times</i><a href="#global_times">[1]</a> is a daily tabloid newspaper published under the auspices of the Chinese Communist Party's flagship newspaper, the <i>People's Daily</i>. It is known for offering commentary on international issues from a Chinese nationalistic perspective and is sometimes referred to as "China's Fox News" due to its focus on propaganda and the monetisation of nationalism. Articles in the <i>Global Times</i> often discuss issues of international politics, economy, and society, with an emphasis on China's view of world events.</p>

"""

guide_html = """
<div class="collapsible-section-content">
    <h3>Modes Overview</h3>
    <p>The application includes four modes:</p>
    <ul>
        <li><i>Single Mode</i>: Analyze a single article in detail.</li>
        <li><i>Double Mode</i>: Compare two articles side by side.</li>
        <li><i>All Mode</i>: Analyze all articles in a selected dataset.</li>
        <li><i>About</i>: Information about the application and its authors.</li>
    </ul>

    <p>Each mode provides different functionalities to suit the user's needs, whether for individual article analysis, comparison, or comprehensive dataset exploration. The user can switch between the modes using the top-right navigation bar with four buttons corresponding to the modes.</p>

    <h3>Interactivity of Plots</h3>
    <p>Most of the plots generated in the application are interactive thanks to the Plotly<a href="#plotly">[2]</a> package, providing users with the following capabilities:</p>
    <ul>
        <li><strong>Zoom In:</strong> Users can zoom in on a specific part of the plot by clicking and dragging the cursor over the desired area. This feature allows for detailed inspection of smaller regions of the data.</li>
        <li><strong>Zoom Out:</strong> Double-clicking anywhere on the plot will reset the zoom level to the default view, allowing users to return to the original scale.</li>
        <li><strong>Data Insights on Hover:</strong> Hovering over elements of the plot (e.g., bars, points, or lines) displays a tooltip with detailed information, such as exact values, labels, or other relevant data.</li>
        <li><strong>Filter Data via Legend:</strong> Clicking on items in the legend toggles their visibility on the plot. This allows users to filter out specific categories or elements for a more focused analysis.</li>
        <li><strong>Top-Right Toolbar:</strong> Each plot features a toolbar in the top-right corner with the following tools:</li>
        <ul>
            <li><strong>Download as PNG:</strong> Saves the current state of the plot as a .png file.</li>
            <li><strong>Zoom In/Out:</strong> Incrementally adjusts the zoom level of the entire plot.</li>
            <li><strong>Pan:</strong> Enables moving across the plot by dragging the cursor, useful for navigating larger datasets.</li>
            <li><strong>Box Select:</strong> Allows selecting a rectangular area of the plot for closer examination.</li>
            <li><strong>Lasso Select:</strong> Enables free-form selection of areas in the plot, useful for isolating irregular regions.</li>
            <li><strong>Autoscale:</strong> Automatically adjusts the plot to fit all data points within the visible area.</li>
            <li><strong>Reset Axes:</strong> Resets the axes to their original scales, restoring the default view of the plot.</li>
        </ul>
    </ul>

    <h3>File Upload Format</h3>
    <p>To upload a file in <strong>Single</strong> or <strong>Double</strong> sections, click on the Browse button in the right menu and select a file from your device.</p>

    <p>Ensure that the file is in the <strong>.txt</strong> format and it is structured as described below:</p>

    <ul>
        <li><strong>Published Time:</strong> [Date: YYYY-MM-DD]</li>
        <li><strong>Title:</strong> [Title of article]</li>
        <li><strong>Category 1:</strong> [First category] ; <strong>Category 2:</strong> [Second Category]</li>
        <li><strong>Author:</strong> [Author]</li>
        <li><strong>Author Title:</strong> [More information about author or: <i>Author details not found</i>]</li>
        <li><strong>Author Description:</strong> [Description of author or <i>Author details not found</i>]</li>
        <li><strong>Text:</strong> Each sentence begins on a new line.</li>
    </ul>

    <p><strong>Example:</strong></p>
    <pre>
Published Time: 2023-10-22
Title: Israel’s deepening attacks in Gaza likely to embroil more military forces into conflicts: experts
Category 1: CHINA; Category 2: DIPLOMACY
Author: Zhao Yusha
Author Title: Reporter, Beijing
Author Description: Global Times reporter covering international affairs, politics and society.
Text:
Israel's deepening attacks in Gaza for what it has called the "next stage of the war" are likely to embroil more regional military forces.
An escalation of Israeli military operations in Gaza could see the US bogged down in another crisis in the Middle East.
    </pre>
    <h3>Single Mode</h3>
    <p>In <i>Single Mode</i>, users can select or upload one article for analysis. The analysis includes Named Entity Recognition (NER), sentiment analysis, and several visualizations to display trends and key insights.</p>

    <h4>Layout</h4>
    <p>The layout of the Single Mode consists of the following elements:</p>
    <ul>
        <li>On the <strong>left side</strong>, there is a container with a header that initially states <code>No file uploaded</code>. This container dynamically updates based on user interactions, such as uploading an article or selecting one from the dropdown.</li>
        <li>On the <strong>right side</strong>, there is a container referred to as the <strong>right menu</strong>, containing the following elements:</li>
        <ul>
            <li>A <strong>Select Article</strong> button with a dropdown list below it. The dropdown initially displays <code>None</code>, indicating no article is selected. Expanding the dropdown allows users to choose one of the predefined articles from <i>Global Times</i>.</li>
            <li>An <strong>Upload Article</strong> button located below the dropdown. Clicking this button opens a file picker, allowing users to select and upload an article from their computer.</li>
            <li>A <strong>Hide Menu</strong> button situated below the <code>Upload Article</code> button. Clicking this button hides the entire right menu, leaving only a tab sticking out from the right edge of the screen with the label <code>Show Menu</code>.</li>
            <li>If the <code>Show Menu</code> tab is clicked, the right menu reappears, restoring its visibility and functionality.</li>
        </ul>
    </ul>
 <h4>Article Selection</h4>
    <p>Users can choose one article from the following predefined datasets:</p>
    <ul>
        <li><strong>Gaza before conflict:</strong> Articles discussing Gaza before the outbreak of the conflict.</li>
        <li><strong>Gaza during conflict:</strong> Articles discussing Gaza during the ongoing conflict.</li>
        <li><strong>Ukraine before conflict:</strong> Articles discussing Ukraine before the conflict.</li>
        <li><strong>Ukraine during conflict:</strong> Articles discussing Ukraine during the conflict.</li>
    </ul>
    <p>Alternatively, users can upload their own article in a format described earlier.</p>

    <h4>Article Analysis</h4>
    <p>Once an article is either selected from the predefined list or uploaded by the user, the system begins to analyse the article. The process unfolds as follows:</p>

    <h5>Analysis in Progress:</h5>
    <ul>
        <li>During the analysis, the header of the left container displays a loading circle in the centre.</li>
        <li>Additionally, a small progress window appears in the bottom-right corner. This window contains:</li>
        <ul>
            <li>A progress bar that visually indicates the percentage of analysis completed.</li>
            <li>Textual information about the current stage of the analysis and its progress, such as <code>Analysing sentiment for entity 26/77</code>.</li>
        </ul>
    </ul>

    <h5>Post-Analysis View:</h5>
    <p>After the analysis is completed, the <strong>right menu</strong> automatically hides to provide more space for displaying the results. The left container updates with the following elements:</p>
    <ul>
        <li><strong>Header:</strong> The article's header is displayed in the format: <code>title -- published date -- category 1 / category 2</code></li>
        <li><strong>Summary:</strong> Summarized content of the article is provided at the top.</li>
        <li><strong>Legends:</strong> Below the summary, two legends are shown:</li>
        <ul>
            <li>The <strong>entity legend</strong> explains the abbreviations used for entity types.</li>
            <li>The <strong>sentiment legend</strong> explains the meaning of sentiment-based colours (e.g., green for positive, grey for neutral, red for negative).</li>
        </ul>
        <li><strong>Article Text:</strong> A cropped version of the article text is displayed with the following features:</li>
        <ul>
            <li><strong>Named Entity Highlights:</strong></li>
            <ul>
                <li>Named entities detected by the machine learning model are highlighted.</li>
                <li>Each entity is coloured based on its sentiment (positive, neutral, or negative) as determined by the model.</li>
                <li>Hovering over a highlighted entity displays a tooltip with detailed information, including:</li>
                <ul>
                    <li><strong>Value:</strong> The text of the entity (e.g., <code>China</code>).</li>
                    <li><strong>Entity Type:</strong> The category of the entity (e.g., <code>Geopolitical entities: countries, cities, states</code>).</li>
                    <li><strong>Sentiment:</strong> The sentiment classification (e.g., <code>Negative</code>).</li>
                </ul>
            </ul>
            <li><strong>Sentence Sentiment Indicators:</strong></li>
            <ul>
                <li>At the end of each sentence, an emoji represents the overall sentiment of the sentence:</li>
                <ul>
                    <li><strong>Negative:</strong> A sad red emoji.</li>
                    <li><strong>Neutral:</strong> A grey neutral-face emoji.</li>
                    <li><strong>Positive:</strong> A green smiling emoji.</li>
                </ul>
                <li>Hovering over the emoji displays a tooltip with the sentiment classification of the sentence (e.g., <code>Negative sentence</code>).</li>
            </ul>
        </ul>
        <li><strong>Expandable Text:</strong></li>
        <ul>
            <li>Below the displayed text, a <strong>View More</strong> button is available. Clicking it expands the text to its full length.</li>
            <li>Once expanded, the button text changes to <strong>View Less</strong>, allowing the user to collapse the text back to its cropped version.</li>
        </ul>
    </ul>
        <h4>Visualizations</h4>
    <p>The following visualizations are generated in Single Mode:</p>
    <ul>
        <li><strong>Named Entity Types Distribution:</strong> A bar chart displaying the named entity types distribution for entities recognized by SpaCy<a href="#spacy">[3]</a> model. Interactive: yes.</li>
        <li><strong>Frequency of Most Common Named Entities:</strong> A bar chart showing the frequency of the most common named entities in the article. Interactive: yes.</li>
        <li><strong>Sentence-Based vs Entity-Based Sentiment Distribution Comparison:</strong> A bar chart comparing the sentence-based with the entity-based sentiment distribution for the article. Interactive: yes.</li>
        <li><strong>Top 100 Words Word Cloud:</strong> A word cloud showing the most common words for the article. Interactive: no.</li>
    </ul>

    <h3>Double Mode</h3>
    <p><i>Double Mode</i> allows users to analyze two articles side by side, enabling direct comparison of their content, sentiment, and named entities.</p>

    <h4>Layout</h4>
    <p>The layout of <i>Double Mode</i> is similar to <i>Single Mode</i>, with the following modifications to accommodate the comparison of two articles:</p>
    <ul>
        <li>On the <strong>left side</strong>, there are two containers stacked vertically, each with a header initially stating <code>No file uploaded</code>. These correspond to the two articles being compared.</li>
        <li>In the <strong>right menu</strong>:</li>
        <ul>
            <li>There are separate buttons and dropdowns for each article:</li>
            <ul>
                <li><strong>Select The First Article</strong> and <strong>Upload The First Article</strong> for the first container.</li>
                <li><strong>Select The Second Article</strong> and <strong>Upload The Second Article</strong> for the second container.</li>
            </ul>
        </ul>
    </ul>
    <h4>Article Selection</h4>
<p>Users can select two articles from the available datasets or upload their own articles for comparison. Each article will undergo the same analysis as in <i>Single Mode</i>, but the results will be displayed side by side.</p>

<h4>Articles Analysis</h4>
<p>In <i>Double Mode</i>, the article analysis proceeds similarly to the process described in <i>Single Mode</i>, with the following differences:</p>

<h5>Analysis in Progress:</h5>
<ul>
    <li>The first article is analysed first, following the same procedure as in <i>Single Mode</i>, with the loading circle and progress information displayed in the corresponding container.</li>
    <li>Once the analysis of the first article is completed, the second article begins its analysis.</li>
</ul>

<h5>Post-Analysis View:</h5>
<ul>
    <li>The results for each article are displayed in their respective containers on the left side.</li>
    <li>Each container updates independently with its article’s header, legends, text highlights, and expandable text options as described in <i>Single Mode</i>.</li>
</ul>

<h4>Visualizations</h4>
<p>The following visualizations are generated to compare both articles:</p>
<ul>
    <li><strong>Named Entity Types Distribution Comparison:</strong> A bar chart comparing the named entity types distribution in the two articles for entities recognized by SpaCy<a href="#spacy">[3]</a> model. Interactive: yes.</li>
    <li><strong>Frequency of Most Common Named Entities:</strong> A bar chart showing the frequency of the most common named entities in both articles. Interactive: yes.</li>
    <li><strong>Sentence-Based Sentiment Distribution Comparison:</strong> A bar chart comparing the sentence-based sentiment distribution for both articles. Interactive: yes.</li>
    <li><strong>Entity-Based Sentiment Distribution Comparison:</strong> A bar chart comparing the entity-based sentiment distribution for both articles. Interactive: yes.</li>
    <li><strong>Top 100 Words Word Cloud Comparison:</strong> Two word clouds showing the most common words for each article. Interactive: no.</li>
</ul>
<h3>All Mode</h3>
<p><i>All Mode</i> provides an analysis of all articles in a selected dataset.</p>

<h4>Filters and Selectors</h4>
<p>The <i>All Mode</i> includes several filters and selectors to help users refine their analysis. After using a filter or selector all plots that are affected by it are reloaded. The filters and selectors are available in the right menu. There is also "Hide Menu"/"Show Menu" button as in the previous modes.</p>
<ul>
    <li><strong>Select dataset:</strong> Choose a dataset for analysis (options: "Gaza before conflict", "Gaza during conflict", "Ukraine before conflict", "Ukraine during conflict"). This selector works for all plots. It states on which dataset the analysis should be conducted. The dataset is also stated in the title of each plot.</li>
    <li><strong>Enter number of most common words in word clouds:</strong> Specify the number of words to display in the word cloud (default: 100). Users can enter an integer number greater than or equal to 1.</li>
    <li><strong>Select part of speech for word cloud:</strong> Choose a part of speech (e.g., Personal Pronouns, Verbs in Non-3rd Person Singular Present Form) for word cloud analysis. Parts of speech can be searched.</li>
    <li><strong>Select entity type:</strong> Choose the type of named entities (options: "Person", "Organisation", "Location", "Miscellaneous").</li>
    <li><strong>Select sentiment model:</strong> Choose the sentiment analysis model (options: "TSC"<a href="#newssentiment">[4]</a>, "VADER"<a href="#vader">[5]</a>).</li>
    <li><strong>Select sentiment type for word cloud and sentiment by target:</strong> Choose a sentiment category (options: "Positive", "Neutral", "Negative") to focus on in the visualizations.</li>
    <li><strong>Select target for sentiment over time:</strong> Choose a target entity for sentiment analysis over time. The target is selected based on the most common named entities in the dataset.</li>
    <li><strong>Enter words for concordance (comma-separated):</strong> Enter a comma-separated list of words to use for concordance analysis.</li>
    <li><strong>Enter N-gram number:</strong> Specify the number of grams for concordance analysis (an integer greater than or equal to 2).</li>
    <li><strong>Select keywords for trends:</strong> Choose keywords to analyze trends over time.</li>
    <li><strong>Select date aggregation for keywords trends:</strong> Choose the date aggregation for the trends analysis (options: "daily", "weekly", "monthly").</li>
</ul>

<h4>Dataset Upload</h4>
<p>In the right menu, there is a button <strong>Upload Dataset</strong>. This button allows users to upload a new dataset for analysis. The following upload options are available:</p>

<ul>
    <li>Multiple <code>.txt</code> files.</li>
    <li>A single compressed <code>.zip</code> file containing multiple <code>.txt</code> files.</li>
</ul>

<p>All files, including those within a <code>.zip</code>, must adhere to the format described in the <em>File Upload Format</em> section to be successfully analyzed.</p>

<p>Below the <strong>Upload Dataset</strong> button, there is a field <strong>Enter Dataset Name</strong>. Users can specify a name for the uploaded dataset here. If no name is provided:</p>

<ul>
    <li>If a compressed folder is uploaded, the folder's name will be used as the dataset name.</li>
    <li>Otherwise, a random name will be generated.</li>
</ul>

<h5>Dataset Analysis</h5>

<p>Once a dataset is uploaded, users can initiate its analysis by clicking the <strong>Analyze Dataset</strong> button. The analysis process includes the following features:</p>

<ul>
    <li>A small progress window appears in the bottom-right corner of the interface, displaying:
        <ul>
            <li>A progress bar indicating the percentage of completion.</li>
            <li>Informative text about the current stage of the analysis.</li>
        </ul>
    </li>
    <li>The duration of the analysis depends on the dataset size. For instance:
        <ul>
            <li>A dataset with 150 articles takes approximately one hour to analyze.</li>
        </ul>
    </li>
</ul>

<h5>Post-Analysis Features</h5>

<p>After the analysis is completed:</p>

<ul>
    <li>Information about the analysis status is displayed in the progress window:
        <ul>If the analysis is successful, a message appears indicating that the process is complete.</ul>
        <ul>If the dataset is small (less than 50 articles), a warning message appears, indicating that some plots may not be available due to the dataset size.</ul>
    </li>
    <li>New plots and visualizations are generated based on the dataset.</li>
    <li>The uploaded dataset appears as a selectable option under <strong>Select dataset</strong> in the right menu. Users can return to this dataset for further exploration at any time.</li>
</ul>

<h4>Visualizations</h4>
<p>In this mode, the visualizations are divided into five sections:</p>
<ul>
    <li><strong>Exploratory Data Analysis (EDA):</strong> Provides an overview of the article's data through statistical summaries, distributions, and other general insights to explore trends and patterns.</li>
    <li><strong>Named Entity Recognition (NER):</strong> Displays a breakdown of identified entities in the text.</li>
    <li><strong>Sentiment:</strong> Visualizes the sentiment analysis of the article, showing the distribution of sentiments (positive, neutral, negative) for different entities and sentences.</li>
    <li><strong>Community Graphs:</strong> Presents graphical representations of how named entities are interconnected.</li>
    <li><strong>N-grams:</strong> Displays frequent word combinations or phrases (n-grams) from the article.</li>
    <li><strong>Keywords trends:</strong> Shows the frequency of selected keywords over time.</li>
</ul>
<p>Users can expand or collapse each section using toggles styled with arrows (⯆/⯈) that change dynamically based on the section's state.</p>
<h5>Exploratory Data Analysis (EDA)</h5>
<p>In this section, the following visualizations are generated:</p>
<ul>
    <li>
        <strong>Number of words in articles distribution:</strong> 
        A histogram showing the distribution of word counts across all articles and the mean word count. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>Number of sentences in articles distribution:</strong> 
        A histogram showing the distribution of sentence counts across all articles and the mean sentence count. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>Top 100 most common words:</strong> 
        A word cloud of the 100 most common words across the dataset, excluding stopwords. <br>
        <strong>Corresponding filters:</strong> Select dataset, Enter number of most common words in word clouds. <br>
        <strong>Interactive:</strong> No.
    </li>
    <li>
        <strong>Most common parts of speech:</strong> 
        A bar chart showing the distribution of parts of speech across all articles. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>Top 100 most common words for a part of speech:</strong> 
        A word cloud of the 100 most common words for a given part of speech. <br>
        <strong>Corresponding filters:</strong> Select dataset, Enter number of most common words in word clouds, Select part of speech for word cloud. <br>
        <strong>Interactive:</strong> No.
    </li>
</ul>


<h5>Named Entity Recognition (NER)</h5>
<p>For the entire dataset, the following visualizations are available:</p>
<ul>
    <li>
        <strong>Most frequently mentioned Named Entity types:</strong> 
        A bar chart showing the distribution of different named entity types (Location, Organization, Person, Miscellaneous) across the dataset. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>Top 15 words of a Named Entity type:</strong> 
        A bar chart displaying the frequency of the 15 most common words of a given named entity type. <br>
        <strong>Corresponding filters:</strong> Select dataset, Select entity type. <br>
        <strong>Interactive:</strong> Yes.
    </li>
</ul>


<h5>Sentiment Analysis</h5>
<p>For the dataset, sentiment analysis results are displayed as:</p>
<ul>
    <li>
        <strong>Comparison of overall sentiment distribution:</strong> 
        A histogram comparing the sentiment (positive, neutral, negative) across all articles for 
        TSC<a href="#newssentiment">[4]</a> and VADER<a href="#vader">[5]</a> models. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>TSC sentiment proportions over time:</strong> 
        A stacked bar chart showing the distribution of sentiment (positive, neutral, negative) across all articles over time (monthly) for a chosen sentiment model. <br>
        <strong>Corresponding filters:</strong> Select dataset, Select sentiment model. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>Word Cloud for sentences of chosen sentiment:</strong> 
        A word cloud showing the most common words used in sentences with a chosen sentiment (positive, neutral, negative) defined by a chosen model. <br>
        <strong>Corresponding filters:</strong> Select dataset, Select sentiment model, Select sentiment type for word cloud and sentiment by target. <br>
        <strong>Interactive:</strong> No.
    </li>
    <li>
        <strong>Overall sentiment distribution per target:</strong> 
        A bar chart displaying the distribution of sentiment (positive, neutral, negative) per target. Targets were chosen based on the most common named entities for each dataset. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>Sentiment over time for selected target:</strong> 
        A bar chart showing the sentiment distribution (positive, neutral, negative) over time (monthly) for a selected target. <br>
        <strong>Corresponding filters:</strong> Select dataset, Select target for sentiment over time. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>Overall Sentiment Distribution Per Target Over Time:</strong> 
        A heatmap displaying the distribution of selected sentiment (positive, neutral, negative) per target over time (monthly). Targets were chosen based on the most common named entities for each dataset. <br>
        <strong>Corresponding filters:</strong> Select dataset, Select sentiment type for word cloud and sentiment by target. <br>
        <strong>Interactive:</strong> Yes.
    </li>
</ul>

<h5>Community Graphs</h5>
<p>For this category, there is only one graph:</p>
<ul>
    <li>
        <strong>Co-occurrence in same sentence relationship graph:</strong> 
        A graph that shows relationships of co-occurrence in the same sentence. <br>
        <strong>Nodes:</strong> Represent entities. <br>
        <strong>Edges:</strong> Represent co-occurrence within the same sentence. <br>
        <strong>Node size:</strong> Indicates the node strength. <br>
        <strong>Edge width:</strong> Indicates the frequency of co-occurrence. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> No.
    </li>
</ul>


<h5>N-grams Analysis</h5>
<p>In this section, the following analyses are provided:</p>
<ul>
    <li>
        <strong>Most common words in bigrams graph:</strong> 
        A directed graph displaying the relationships between the most common bi-grams in the dataset. <br>
        <strong>Edge width:</strong> States the frequency of the bi-gram in articles from the dataset. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> No.
    </li>
    <li>
        <strong>N-gram Concordance Table:</strong> 
        A table listing the occurrences of a selected n-gram within the dataset. <br>
        <strong>Center column:</strong> Displays the chosen keyword. <br>
        <strong>Lefts column:</strong> Shows the left context of sentences containing the keyword. <br>
        <strong>Rights column:</strong> Shows the right context. Contexts are shortened to the Enter N-gram number. <br>
        <strong>Count column:</strong> States how many times the found N-gram occurred in the articles from a given dataset. <br>
        <strong>Corresponding filters:</strong> Select dataset, Enter words for concordance (comma-separated), Enter N-gram number. <br>
        <strong>Interactive:</strong> No.
    </li>
</ul>


<h5>Keywords Trends</h5>
<p>For this section, the following visualizations are generated:</p>
<ul>
    <li>
        <strong>Most popular keywords trends:</strong> 
        A line chart showing the frequency of the most popular keywords over time. <br>
        <strong>Corresponding filters:</strong> Select dataset. <br>
        <strong>Interactive:</strong> Yes.
    </li>
    <li>
        <strong>Stacked keywords trends:</strong> 
        A stacked bar chart showing the frequency of selected keywords over time. <br>
        <strong>Corresponding filters:</strong> Select dataset, Select keywords for trends, Select date aggregation for keywords trends. <br>
        <strong>Interactive:</strong> Yes.
    </li>
</ul>


<h3>About Mode</h3>
<p>The <i>About Mode</i> provides users with the following information:</p>
<ul>
    <li><strong>Introduction:</strong> A brief overview of the app’s purpose and functionality.</li>
    <li><strong>User Guide:</strong> This manual with detailed instructions.</li>
    <li><strong>Authors:</strong> Information about the authors of the app.</li>
</ul>
<p>Each section can be collapsed or expanded in the same way as in <i>All Mode</i>.
<h3>Glossary</h3>
<p><strong>Stopwords:</strong> Commonly used words in a language (e.g., 'the', 'and', 'is') that are excluded from analysis due to their lack of meaningful content.</p>

<p><strong>Named Entity Recognition (NER):</strong> A process of identifying and classifying entities in text, such as names of people, locations, and organizations.</p>

<p><strong>Sentiment Analysis:</strong> The computational task of determining the sentiment expressed in a text, whether it is positive, neutral, or negative.</p>

<p><strong>TSC:</strong> Text Sentiment Classification, a model for analyzing sentiment at the text level.</p>

<p><strong>VADER:</strong> Valence Aware Dictionary and sEntiment Reasoner, a model for analyzing sentiment based on a predefined dictionary and rules.</p>

<h3>References</h3>
<p><strong>Plotly Technologies Inc.</strong> (2015). Collaborative data science. Plotly Technologies Inc. Montreal, QC. Available at: <a href="https://plot.ly" target="_blank">https://plot.ly</a>. <a id="plotly"></a>[2]</p>

<p><strong>Honnibal, M.</strong> and <strong>Montani, I.</strong> (2017). spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks, and incremental parsing. Unpublished manuscript. <a id="spacy"></a>[3]</p>

<p><strong>Hamborg, F.</strong> and <strong>Donnay, K.</strong> (2021). NewsMTSC: (Multi-)Target-dependent Sentiment Classification in News Articles. In <em>Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL 2021)</em>, Virtual Event, Apr. To appear. <a id="newssentiment"></a>[4]</p>

<p><strong>Hutto, C.J.</strong> and <strong>Gilbert, E.E.</strong> (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. In <em>Eighth International Conference on Weblogs and Social Media (ICWSM-14)</em>, Ann Arbor, MI, June. <a id="vader"></a>[5]</p>

<p><strong>Wikipedia contributors</strong> (2024). Global Times. The Free Encyclopedia. Available at: <a href="https://en.wikipedia.org/w/index.php?title=Global_Times&oldid=1254649071" target="_blank">https://en.wikipedia.org/w/index.php?title=Global_Times&oldid=1254649071</a> [Online; accessed 13-November-2024]. <a id="global_times"></a>[1]</p>
</div>
"""

authors_html = """
<div class="collapsible-section-content">
<p><strong style="color: #333;">Authors:</strong> Łukasz Grabarski & Marta Szuwarska
 (Warsaw University of Technology)</p>
<p><strong style="color: #333;">Supervisor:</strong> Dr. Anna Wróblewska
 (Warsaw University of Technology)</p>
<p><strong style="color: #333;">Co-supervisor:</strong> Prof. Agnieszka Kaliska
 (Adam Mickiewicz University in Poznań)</p>
<p><strong style="color: #333;">Co-operations:</strong> Prof. Anna Rudakowska
 (Tamkang University in Taiwan), Dr. Daniel Dan (Modul University in Vienna)</p>
</div>
"""
