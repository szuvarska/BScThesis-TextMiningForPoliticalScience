# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system packages for debugging
RUN apt-get update && apt-get install -y \
    net-tools \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Install models
RUN python3 -c "from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline;  \
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer;  \
    from NewsSentiment import TargetSentimentClassifier;  \
    pipeline('summarization', model='facebook/bart-large-cnn');  \
    model_name = 'dbmdz/bert-large-cased-finetuned-conll03-english';  \
    tokenizer = AutoTokenizer.from_pretrained(model_name);  \
    model = AutoModelForTokenClassification.from_pretrained(model_name);  \
    pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy='simple', device=-1); \
    SentimentIntensityAnalyzer();  \
    TargetSentimentClassifier()"

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME=GlobalTimesAnalysis

# Run app.py when the container launches
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
