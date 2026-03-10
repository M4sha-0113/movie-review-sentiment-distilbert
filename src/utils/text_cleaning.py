import re
from bs4 import BeautifulSoup

def clean_text(text):
    """
    Cleans the input text by removing HTML tags, URLs, and extra whitespace.
    """
    # 1. Remove HTML tags
    # using BeautifulSoup with "html.parser"
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 2. Remove URLs
    # this regex is looking for http, https, and www. addresses
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text 

