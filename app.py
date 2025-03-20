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

print("Company News Summarization")
os.makedirs("audios",exist_ok=True)
    
company_name = input("Enter Company Name: ")
    
if company_name:
    file_path = save_company_news(company_name)
        
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            articles = json.load(file)
                
            for article in articles:
                print(f"\nTitle: {article['title']}")
                print(f"Content: {article['content'][:100]}...") 
                print(f"Read more: {article['url']}")
                
        del articles
        gc.collect()
    else:
        print("Failed to fetch news. Try again.")
else:
    print("Please enter a company name.")

with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
    data = json.load(file)

for article in data:
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

with open(f"Company/{company_name}.json", "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)


with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
    articles = json.load(file)


if not isinstance(articles, list):
    raise ValueError("JSON data must be a list of articles")

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


with open(f"Company/{company_name}.json", "r", encoding="utf-8") as file:
    data = json.load(file)  

articles = data["Articles"] 
comparative_score = data["Comparative Sentiment Score"] 

comparisons = []
hindi_text = ""

audio_num = 1

for i in range(len(articles)):
    for j in range(i + 1, len(articles)):  
        x = articles[i]['title']
        y = articles[j]['title']
        result = GEMINI_LLM_COMPARISON(f"Compare {x} and {y}")
        result = result.replace("*", "")
        hindi_text = hindi_text + GoogleTranslator(source="en", target="hi").translate(result)
        audio_output(GoogleTranslator(source="en", target="hi").translate(result),audio_num)
        comparisons.append(result)
        audio_num = audio_num + 1
        # print(f"{x} AND {y}")


output_data = {
    "Articles": articles, 
    "Comparative Sentiment Score": comparative_score,
    "Comparison_through_articles": comparisons
}

with open(f"Company/{company_name}.json", "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=4)



# print(hindi_text)

with open("translated_text.txt", "w", encoding="utf-8") as file:
    file.write(hindi_text)

with open("translated_text.txt", "r", encoding="utf-8") as file:
    data = file.read()

# print(data)

# audio_output(data)

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

print(f"Merged audio saved as {output_file}")

# for file in audio_files:
#     if file != "merged_audio.wav":  # Ensure we don't delete the final merged file
#         os.remove(os.path.join(audio_folder, file))
# print("Deleted segmented audio files after merging.")

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

print(f"Topic Overlap added and saved at: {output_path}")
