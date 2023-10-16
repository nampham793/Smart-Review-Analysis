# Use the official Python image from Docker Hub as the base image.
FROM python:3.10-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the current directory contents into the container at /app.
RUN apt-get update && \
    apt-get install -y default-jre && \
    apt-get clean;

COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

# Specify the command to run when the container starts.
CMD ["uvicorn", "app:app", "--host = 0.0.0.0", "--port = 8000", "--workers = 1"]
