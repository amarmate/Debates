import logging
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

url = "https://cnnportugal.iol.pt/presidenciais-2026/calendario-dos-debates/calendario-dos-debates-das-presidenciais-de-2026-todos-os-frente-a-frente/20251119/691df9a6d34e3caad84b764f"
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

for element in article.find_all(['p', 'h2', 'h3', 'strong']):
    text = element.get_text().strip()
    
    # Check if this element is a date header
    if any(day in text for day in pt_days):
        current_date = text
        continue
    
    # Check if this element contains a debate link
    link = element.find('a', href=True)
    if link and "íntegra" in text.lower():
        # Clean up the debate info (removes the "see in full" part)
        info = text.split('-')[0].strip()
        
        # Convert date to standardized format (YYYY-MM-DD)
        standardized_date = parse_portuguese_date(current_date)
        
        # Parse debate info: "Candidate1 vs. Candidate2 (Channel)"
        # Extract channel (text in parentheses)
        channel_match = re.search(r'\(([^)]+)\)', info)
        channel = channel_match.group(1) if channel_match else ""
        
        # Extract candidates (everything before the channel)
        candidates_text = re.sub(r'\s*\([^)]+\)\s*$', '', info).strip()
        
        # Split by "vs." or "vs" to get candidate1 and candidate2
        if ' vs. ' in candidates_text:
            parts = candidates_text.split(' vs. ', 1)
        elif ' vs ' in candidates_text:
            parts = candidates_text.split(' vs ', 1)
        else:
            parts = [candidates_text, ""]
        
        candidate1 = parts[0].strip() if len(parts) > 0 else ""
        candidate2 = parts[1].strip() if len(parts) > 1 else ""
        
        data.append({
            "candidate1": candidate1,
            "candidate2": candidate2,
            "date": standardized_date,
            "channel": channel,
            "url": link['href']
        })

df = pd.DataFrame(data)
df.to_csv("presidencial_debates_2026.csv", index=False)
logger.info("Preview:\n%s", df.head())