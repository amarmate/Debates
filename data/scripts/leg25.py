import logging
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

url = "https://cnnportugal.iol.pt/decisao-25/calendario-dos-debates/calendario-dos-debates-das-legislativas-de-2025-veja-aqui-quando-e-onde/20250417/67f4e39dd34ef72ee44472ff"
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

article = soup.find('div', class_='article-body') or soup.find('article')
data = []
current_date = "Unknown Date"

# Days and months in Portuguese to help identify date lines
pt_days = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]

# Portuguese month names to numbers mapping
pt_months = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12
}

def parse_portuguese_date(date_str, year=2025):
    """Convert Portuguese date format to YYYY-MM-DD"""
    # Remove day of the week if present
    date_clean = re.sub(r'^[^,]+,?\s*', '', date_str).strip()
    
    # Extract day and month
    match = re.match(r'(\d+)\s+de\s+(\w+)', date_clean.lower())
    if match:
        day = int(match.group(1))
        month_name = match.group(2)
        month = pt_months.get(month_name)
        
        if month:
            try:
                date_obj = datetime(year, month, day)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                return date_clean
    return date_clean

if article is None:
    logger.error("Could not find article content")
    exit(1)

for element in article.find_all(['p', 'h2', 'h3', 'strong', 'li']):
    text = element.get_text().strip()
    
    # Check if this element is a date header
    if any(day in text for day in pt_days):
        current_date = text
        continue
    
    # Check if this element contains a debate link
    link = element.find('a', href=True)
    if link and "íntegra" in text.lower():
        # Remove "veja na íntegra" part
        info = re.sub(r'\s*-\s*veja\s+na\s+íntegra.*$', '', text, flags=re.IGNORECASE).strip()
        
        # Convert date to standardized format (YYYY-MM-DD)
        standardized_date = parse_portuguese_date(current_date)
        
        # Extract channel (text in parentheses)
        channel_match = re.search(r'\(([^)]+)\)', info)
        channel = channel_match.group(1) if channel_match else ""
        
        # Remove channel from info to get candidates part
        candidates_text = re.sub(r'\s*\([^)]+\)\s*', '', info).strip()
        
        # Remove time prefix (e.g., "21:00 " or "22:00 ")
        candidates_text = re.sub(r'^\d{1,2}:\d{2}\s+', '', candidates_text).strip()
        
        # Split by " - " (dash with spaces) to get candidate1 and candidate2
        if ' - ' in candidates_text:
            parts = candidates_text.split(' - ', 1)
            candidate1 = parts[0].strip() if len(parts) > 0 else ""
            candidate2 = parts[1].strip() if len(parts) > 1 else ""
        else:
            candidate1 = candidates_text
            candidate2 = ""
        
        data.append({
            "candidate1": candidate1,
            "candidate2": candidate2,
            "date": standardized_date,
            "channel": channel,
            "url": link['href']
        })

df = pd.DataFrame(data)
# Remove duplicates based on URL (same debate might appear in both <li> and <strong> elements)
df = df.drop_duplicates(subset=["url"], keep="first")
df.to_csv("legislativas_debates_2025.csv", index=False)
logger.info("Preview:\n%s", df.head())