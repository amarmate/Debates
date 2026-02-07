import csv
import logging
from pathlib import Path
from datetime import datetime
from debate_downloader import get_debate_audio, sanitize_filename

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CSV_FILE = "data/links/legislativas_debates_2025.csv"
DOWNLOAD_FOLDER = Path("data/debates")
SKIP_EXISTING = True  # Skip debates that already exist

def create_debate_title(row):
    """
    Create a nice title for the debate from CSV row data
    
    Format: YYYY-MM-DD - Candidate1 vs Candidate2 - Channel
    """
    parts = []
    
    # Add date if available
    date_str = row.get('date', '').strip() if row.get('date') else ''
    if date_str:
        try:
            # Parse date and format it
            # Handle different date formats
            if len(date_str) == 10:  # YYYY-MM-DD format
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                parts.append(date_obj.strftime('%Y-%m-%d'))
            else:
                parts.append(date_str)
        except:
            if date_str:
                parts.append(date_str)
    
    # Add candidates
    candidate1 = str(row.get('candidate1', '')).strip()
    candidate2 = str(row.get('candidate2', '')).strip()
    
    if candidate1 and candidate2:
        parts.append(f"{candidate1} vs {candidate2}")
    elif candidate1:
        parts.append(candidate1)
    elif candidate2:
        parts.append(candidate2)
    
    # Add channel if available
    channel = str(row.get('channel', '')).strip()
    if channel:
        parts.append(channel)
    
    # Join all parts
    title = " - ".join(parts) if parts else "debate"
    
    return title

def check_already_downloaded(title, date_str=None):
    """
    Check if a debate with this title already exists in the download folder
    """
    if not SKIP_EXISTING:
        return False
    
    if not DOWNLOAD_FOLDER.exists():
        return False
    
    # Format date for filename matching
    if date_str:
        try:
            date_obj = datetime.strptime(str(date_str), '%Y-%m-%d')
            formatted_date = date_obj.strftime("%Y_%m_%d")
        except:
            formatted_date = None
    else:
        formatted_date = None
    
    # Check for existing files
    sanitized_title = sanitize_filename(title)
    
    # Look for files that match the pattern
    if formatted_date:
        pattern = f"{formatted_date}_{sanitized_title}.*"
    else:
        pattern = f"*{sanitized_title}*"
    
    for file in DOWNLOAD_FOLDER.glob(pattern):
        if file.is_file():
            return True
    
    return False

def download_all_debates(csv_file=None, download_audio_only=True, audio_format="mp3"):
    """
    Download all debates from the CSV file
    
    Args:
        csv_file: Path to CSV file (defaults to CSV_FILE global)
        download_audio_only: Whether to download audio only
        audio_format: Audio format for extraction
    """
    csv_file = csv_file or CSV_FILE
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        logger.error(f"❌ CSV file not found: {csv_path.absolute()}")
        return
    
    logger.info(f"📖 Reading debates from: {csv_path.absolute()}")
    
    # Read CSV file
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    logger.info(f"📊 Found {len(rows)} debates in CSV")
    logger.info("")
    
    # Filter out invalid URLs
    valid_rows = [row for row in rows if row.get('url', '').strip().startswith('http')]
    invalid_count = len(rows) - len(valid_rows)
    
    if invalid_count > 0:
        logger.warning(f"⚠️  Skipping {invalid_count} rows with invalid URLs")
        logger.info("")
    
    successful = 0
    skipped = 0
    failed = 0
    
    for idx, row in enumerate(valid_rows):
        url = str(row.get('url', '')).strip()
        title = create_debate_title(row)
        date_str = row.get('date', '').strip() if row.get('date') else None
        
        logger.info("=" * 80)
        logger.info(f"📥 Debate {idx + 1}/{len(valid_rows)}: {title}")
        logger.info(f"🔗 URL: {url}")
        
        # Check if already downloaded
        if check_already_downloaded(title, date_str):
            logger.info(f"⏭️  Already downloaded, skipping...")
            skipped += 1
            logger.info("")
            continue
        
        try:
            # Download the debate
            get_debate_audio(
                page_url=url,
                download_audio_only=download_audio_only,
                audio_format=audio_format,
                title=title
            )
            successful += 1
            logger.info(f"✅ Successfully downloaded: {title}")
        except Exception as e:
            failed += 1
            logger.error(f"❌ Failed to download {title}: {e}")
        
        logger.info("")
    
    # Summary
    logger.info("=" * 80)
    logger.info("📊 DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    logger.info(f"✅ Successful: {successful}")
    logger.info(f"⏭️  Skipped: {skipped}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📁 Total processed: {len(valid_rows)}")
    logger.info("=" * 80)

if __name__ == "__main__":
    download_all_debates()
