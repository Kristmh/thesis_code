import re
from string import punctuation
import emoji
import numpy as np
from langdetect import detect

def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(f"[{re.escape(punctuation)}]", "", text)  # Remove punctuation
    text = " ".join(text.split())  # Remove extra spaces, tabs, and new lines
    text = re.sub(r"https?://\S+", "", text) # Remove URLs.
    text = emoji.demojize(text)
    text = re.sub(r':[a-z_]+:', '', text)  #removes the text representations of emojis
    if not is_english(text):
        # If no english set to NaN.
        text = np.nan
    return text


#Check if english using langdetect
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False
