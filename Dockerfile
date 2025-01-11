# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system packages for debugging
RUN apt-get update && apt-get install -y \
    net-tools \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from src.Summary_script import download_summary_model; download_summary_model()"

# Make port 80 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME=GlobalTimesAnalysis

# Run app.py when the container launches
CMD ["shiny", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
