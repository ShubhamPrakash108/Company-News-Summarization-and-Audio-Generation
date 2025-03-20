# News Summarization and Text-to-Speech Application

This repository contains a **News Summarization and Text-to-Speech (TTS) Application**, which extracts news articles related to a given company, summarizes them, performs sentiment analysis, and converts the summary into audio.

🚀 The application is deployed on **Hugging Face Spaces**:  
👉 [Try it here](https://huggingface.co/spaces/shubhamprakash108/news-summarization-and-yext-to-speech-application)

---

## Features

- **News Extraction**: Fetches the latest news articles related to a given company.
- **Summarization**: Uses NLP techniques to summarize the news articles.
- **Sentiment Analysis**: Classifies the sentiment of each article as *Positive, Neutral, or Negative*.
- **Topic Detection**: Identifies key topics from articles using BERTopic.
- **Comparison Analysis**: Uses a large language model (LLM) to compare articles.
- **Text-to-Speech (TTS)**: Converts summaries into Hindi audio using a Transformer-based TTS model.
- **Streamlit UI**: A user-friendly interface to interact with the application.

---

## Models Used

This project integrates multiple AI models to perform various tasks:

- **Hugging Face Transformers (Falconsai/text_summarization)**: Used for summarizing news articles.
- **Cardiff NLP Twitter RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)**: Used for sentiment analysis, classifying articles as *Positive, Neutral, or Negative*.
- **BERTopic**: Used for topic modeling to extract key topics from articles.
- **Gemini LLM**: Used to compare articles and generate concise comparisons.
- **Facebook MMS-TTS-HIN (facebook/mms-tts-hin)**: Used for generating Hindi speech from the text summary.
- **Sentence Transformers (all-MiniLM-L6-v2)**: Used for embedding-based topic modeling.

These models work together to provide a comprehensive news analysis, summarization, and audio output pipeline.

⚠️ **Note:** This project requires a `.env` file where users must add their own **Google Gemini API keys**. I have used **three free Gemini API keys** to ensure uninterrupted comparisons, but you should add your own keys in the `.env` file.

---

## Installation

Clone the repository:
```bash
git clone https://github.com/shubhamprakash108/news-summarization-and-tts.git
cd news-summarization-and-tts
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage on Another Computer

To use this application on another computer, follow these steps:

1. **Ensure Python is Installed**
   - Install Python (>=3.8) from [python.org](https://www.python.org/downloads/).

2. **Clone the Repository**
   ```bash
   git clone https://github.com/shubhamprakash108/news-summarization-and-tts.git
   cd news-summarization-and-tts
   ```

3. **Set Up a Virtual Environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Add Your Gemini API Keys**
   - Create a `.env` file in the root directory of the project.
   - Add your Gemini API keys in the following format:
     ```plaintext
     GEMINI_API_KEY_1=your_first_api_key
     GEMINI_API_KEY_2=your_second_api_key
     GEMINI_API_KEY_3=your_third_api_key
     ```
   - These keys are used to perform article comparisons.

6. **Run the Application**
   - **For CLI Version:**
     ```bash
     python app.py
     ```
     Enter the company name when prompted, and the application will fetch and process news articles.

   - **For Web UI (Streamlit Version):**
     ```bash
     streamlit run app_sl.py
     ```
     This will open the application in a web browser.

7. **Troubleshooting**
   - Ensure all dependencies are correctly installed.
   - If `torch` or `transformers` fail, try upgrading them:
     ```bash
     pip install --upgrade torch transformers
     ```
   - If Streamlit doesn't launch, try:
     ```bash
     pip install --upgrade streamlit
     ```

---

## Requirements

The project relies on the following major dependencies:
- `transformers`
- `torch`
- `streamlit`
- `beautifulsoup4`
- `newspaper3k`
- `pydub`
- `bertopic`
- `sentence_transformers`
- `deep_translator`
- `google-generativeai`

Check the full list in [`requirements.txt`](requirements.txt).

---

## Deployment on Hugging Face

The application is deployed on Hugging Face Spaces, allowing users to run it in the cloud without installation.  
Visit the live app: **[Hugging Face Deployment](https://huggingface.co/spaces/shubhamprakash108/news-summarization-and-yext-to-speech-application)**

---

## Project Structure

```
├── app.py                 # CLI version of the application
├── app_sl.py              # Streamlit web app version
├── utils.py               # Helper functions for news extraction, summarization, and TTS
├── requirements.txt       # List of dependencies
├── README.md              # Project documentation (this file)
```

---

## Acknowledgments

- **Hugging Face Transformers** for NLP models
- **Google Generative AI** for text comparisons
- **Streamlit** for building the UI
- **BERTopic** for topic extraction

---


## License

This project is licensed under the MIT License.  
Feel free to use and improve it!
