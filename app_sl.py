import streamlit as st
import json
import os
from utils import save_company_news
from utils import sentiment_analysis_model
from utils import news_summarization, audio_output, Topic_finder, GEMINI_LLM_COMPARISON
from collections import Counter
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import gc
import torch
import base64

st.set_page_config(page_title="Company News Summarization", layout="wide")
st.title("Company News Summarization")

def get_audio_player(audio_file):
    audio_file = open(audio_file, 'rb')
    audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    audio_player = f'<audio controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    return audio_player

def process_company_news(company_name):
    os.makedirs("audios", exist_ok=True)
    
    file_path = save_company_news(company_name)
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            articles = json.load(file)
            
            for article in articles:
                st.subheader(f"Title: {article['title']}")
                st.write(f"Content: {article['content'][:100]}...")
                st.write(f"Read more: {article['url']}")
            
        del articles
        gc.collect()
    else:
        st.error("Failed to fetch news. Try again.")
        return False
    
    with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, article in enumerate(data):
        status_text.text(f"Processing article {i+1}/{len(data)}")
        
        topics = Topic_finder(article['title'])
        
        sentiment = sentiment_analysis_model(article['content'])
        article["sentiment"] = sentiment['sentiment']
        
        del sentiment
        gc.collect()
        
        summary = news_summarization(article["content"])
        article["summary"] = summary
        
        article["topics"] = topics
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        progress_bar.progress((i+1)/len(data))
    
    with open(f"Company/{company_name}.json", "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    
    with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
        articles = json.load(file)
    
    if not isinstance(articles, list):
        st.error("JSON data must be a list of articles")
        return False
    
    sentiment_counts = Counter(article["sentiment"] for article in articles if "sentiment" in article)
    
    output_data = {
        "Articles": articles, 
        "Comparative Sentiment Score": {  
            "Sentiment_Counts": {
                "Positive": sentiment_counts.get("Positive", 0),
                "Negative": sentiment_counts.get("Negative", 0),
                "Neutral": sentiment_counts.get("Neutral", 0),
            }
        }
    }
    
    with open(f"Company/{company_name}.json", "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4)
    
    gc.collect()
    
    status_text.text("Processing article comparisons...")
    
    with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
        data = json.load(file)  
    
    articles = data["Articles"] 
    comparative_score = data["Comparative Sentiment Score"] 
    
    comparisons = []
    hindi_text = ""
    
    audio_num = 1
    
    comparison_progress = st.progress(0)
    comparison_status = st.empty()
    total_comparisons = len(articles) * (len(articles) - 1) // 2
    comparison_count = 0
    
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            comparison_count += 1
            comparison_status.text(f"Comparing article {i+1} and {j+1} ({comparison_count}/{total_comparisons})")
            
            x = articles[i]['title']
            y = articles[j]['title']
            result = GEMINI_LLM_COMPARISON(f"Compare {x} and {y}")
            result = result.replace("*", "")
            hindi_text = hindi_text + GoogleTranslator(source="en", target="hi").translate(result)
            audio_output(GoogleTranslator(source="en", target="hi").translate(result), audio_num)
            comparisons.append(result)
            audio_num = audio_num + 1
            
            comparison_progress.progress(comparison_count/total_comparisons)
    
    output_data = {
        "Articles": articles, 
        "Comparative Sentiment Score": comparative_score,
        "Comparison_through_articles": comparisons
    }
    
    with open(f"Company/{company_name}.json", "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=4)
    
    with open("translated_text.txt", "w", encoding="utf-8") as file:
        file.write(hindi_text)
    
    with open("translated_text.txt", "r", encoding="utf-8") as file:
        data = file.read()
    
    status_text.text("Merging audio files...")
    
    audio_folder = "audios"
    
    audio_files = sorted(
        [f for f in os.listdir(audio_folder) if f.endswith(".wav")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    
    final_audio = AudioSegment.empty()
    
    for file in audio_files:
        file_path = os.path.join(audio_folder, file)
        audio = AudioSegment.from_wav(file_path)
        final_audio += audio  
    
    output_file = "merged_audio.wav"
    final_audio.export(output_file, format="wav")
    
    status_text.text("Processing topic overlap...")
    
    input_path = f"Company/{company_name}.json"
    output_path = f"Company/{company_name}.json" 
    
    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    topic_counts = Counter(topic for article in data["Articles"] for topic in article["topics"])
    
    common_topics = {topic for topic, count in topic_counts.items() if count > 1}
    unique_topics = [
        {
            "Article": article["title"],
            "Unique Topics": list(set(article["topics"]) - common_topics)
        }
        for article in data["Articles"]
    ]
    
    data["Comparative Sentiment Score"]["Topic Overlap"] = {
        "Common Topics": list(common_topics),
        "Unique Topics per Article": unique_topics
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    
    status_text.text("Analysis complete!")
    return True

def main():
    st.sidebar.header("Company News Analysis")
    
    company_name = st.sidebar.text_input("Enter Company Name:")
    
    if st.sidebar.button("Process Company News"):
        if company_name:
            with st.spinner(f"Processing news for {company_name}..."):
                success = process_company_news(company_name)
                if success:
                    st.success(f"Successfully processed news for {company_name}")
        else:
            st.sidebar.error("Please enter a company name.")
    
    if company_name and os.path.exists(f"Company/{company_name}.json"):
        st.header("Analysis Results")
        
        with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sentiment Analysis")
            sentiment_counts = data["Comparative Sentiment Score"]["Sentiment_Counts"]
            st.bar_chart(sentiment_counts)
        
        with col2:
            if "Topic Overlap" in data["Comparative Sentiment Score"]:
                st.subheader("Common Topics")
                st.write(data["Comparative Sentiment Score"]["Topic Overlap"]["Common Topics"])
        
        st.subheader("Articles")
        for i, article in enumerate(data["Articles"]):
            with st.expander(f"Article {i+1}: {article['title']}"):
                st.write(f"**Summary:** {article['summary']}")
                st.write(f"**Sentiment:** {article['sentiment']}")
                st.write(f"**Topics:** {', '.join(article['topics'])}")
        
        if "Comparison_through_articles" in data:
            st.subheader("Article Comparisons")
            for i, comparison in enumerate(data["Comparison_through_articles"]):
                with st.expander(f"Comparison {i+1}"):
                    st.write(comparison)
        
        st.subheader("Full JSON Data")
        st.json(data)
        
        if os.path.exists("merged_audio.wav"):
            st.subheader("Audio Summary (Hindi)")
            st.markdown(get_audio_player("merged_audio.wav"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()