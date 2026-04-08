OculisAI: Multi-Agent Medical Diagnostics Platform

OculisAI is an advanced AI-driven platform for screening Diabetic Retinopathy. It features a sophisticated Multi-Agent System (MAS) orchestrated by LangGraph to automate image analysis, clinical reporting, and patient communication.

🚀 Key Features

AI-Driven Diagnostics: Employs EfficientNet-B3 for high-accuracy fundus image classification.

Multi-Agent Workflow: A coordinated system of agents handling specialized tasks:

  1. Diagnostic Agent: Image analysis and prediction.

  2. Reporting Agent: Structured clinical report generation (PDF).

  3. Orchestrator: Logic flow and state management via LangGraph.

Intelligent RAG Chatbot: Integrated assistant using FAISS and Llama 3.2 to answer questions based on uploaded medical documents.

Automated Communication: Direct email delivery of reports to patients using Flask-Mail.

Administrative Suite: Comprehensive dashboard for patient management and secure admin registration.

🏗️ Project Structure

app.py: The main Flask application and API hub.

agents.py: Orchestrates the Multi-Agent System logic.

predict.py: Handles the EfficientNet model inference for DR screening.

report_gen.py: Generates clinical summaries and PDF documentation.

database_setup.sql: SQL script to initialize the MySQL database schema.

🛠️ Tech Stack
Framework: Flask (Python)

Orchestration: LangGraph, LangChain

LLM: Ollama (Llama 3.2)

Vector DB: FAISS

Deep Learning: EfficientNet-B3 (PyTorch/TensorFlow)

Database: MySQL

📦 Getting Started
1. Prerequisites

    Python 3.10+

    MySQL Server

    Ollama (with llama3.2:3b model pulled)

2. Setup

  Clone the repository:

    git clone https://github.com/your-username/OculisAI.git
    cd OculisAI

  Install dependencies:

    pip install -r requirements.txt

  Database Configuration:

  Run the database_setup.sql script in your MySQL instance to create the necessary tables.

  .env structure

    DB_HOST=localhost

    DB_USER=root

    DB_PASSWORD=your_password

    DB_NAME=oculisai

    FLASK_SECRET_KEY=your_secret_key

    MAIL_USERNAME=your_email@gmail.com

    MAIL_PASSWORD=your_app_password

3. Run the App

       python app.py
