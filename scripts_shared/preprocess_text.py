import re
from string import punctuation
import emoji
import numpy as np
from langdetect import detect

def preprocess_text(text):

    if not is_english(text):
        # If no english set to NaN.
        text = np.nan
        return text
    text = re.sub(r'https?:\/\/\S+|www\.\S+', '', text)
    text = text.lower()  # Lowercase text
    text = text.strip()  # Remove leading/trailing whitespace
    text = emoji.demojize(text)
    text = re.sub(r':[a-z_]+:', '', text)  #removes the text representations of emojis
    text = re.sub(r'0x[a-fA-F0-9]+', '', text) # Removes hexadecimal strings (ref to commits)
    text = re.sub('\[.*?\]', '', text) # Removes text in square brackets
    text = re.sub('<.*?>+', '', text) # Removes HTML tags
    text = re.sub('[%s]' % re.escape(punctuation), '', text) # Removes punctuation
    text = re.sub('\n', '', text) # Removes new lines
    text = re.sub('\w*\d\w*', '', text) # Removes words containing numbers
    text = " ".join(text.split())  # Remove extra spaces, tabs, and new lines
    return text


#Check if english using langdetect
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False
