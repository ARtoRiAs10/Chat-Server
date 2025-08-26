# Sentiment-Aware Chat Room

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python multi-client chat server where every message is analyzed for its sentiment in real-time using a free, open-source Machine Learning model from Hugging Face.

This project serves as a practical demonstration of integrating modern networking (`asyncio`) with powerful ML libraries (`transformers`) to create an intelligent application.

## Core Features

*   **Multi-Client Architecture:** A central server can handle numerous simultaneous client connections.
*   **Asynchronous Networking:** Built with Python's `asyncio` library for high-performance, non-blocking I/O.
*   **Real-time Machine Learning:** The server uses a pre-trained sentiment analysis model to instantly classify messages as POSITIVE or NEGATIVE.
*   **Message Broadcasting:** The server broadcasts messages, along with their sentiment analysis, to all connected clients.
*   **Open-Source & Free:** Uses only free, open-source libraries and models, with no API keys required.

## Technology Stack

*   **Language:** Python 3.8+
*   **Networking:** `asyncio` (built-in Python library)
*   **Machine Learning:** `transformers` by Hugging Face
*   **ML Framework:** `PyTorch`
*   **ML Model:** `distilbert-base-uncased-finetuned-sst-2-english` (a small, fast, and effective model for sentiment analysis)

## Setup and Installation

Follow these steps to get the project running on your local machine.

**1. Clone the Repository (or create the files)**
   ```bash
   git clone https://github.com/your-username/sentiment-chat.git
   cd sentiment-chat
   ```