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
CSV_FILE = "data/links/debates_unified.csv"
DOWNLOAD_FOLDER = Path("data/debates")
SKIP_EXISTING = True  # Skip debates that already exist

def create_debate_filename(row):
    """
    Create a clean filename base (no spaces, no extension) for the debate.

    Format: YYYY-MM-DD_Party1-vs-Party2_Channel
    Example: 2025-04-07_AD-vs-CDU_TVI
    """
    parts = []
    date_str = (row.get("date") or "").strip()
    if date_str and len(date_str) >= 10:
        parts.append(date_str[:10])  # YYYY-MM-DD
    name1 = str(row.get("party1") or row.get("candidate1") or "").strip()
    name2 = str(row.get("party2") or row.get("candidate2") or "").strip()
    if name1 and name2:
        parts.append(f"{name1}-vs-{name2}")
    elif name1:
        parts.append(name1)
    elif name2:
        parts.append(name2)
    channel = (row.get("channel") or "").strip()
    if channel:
        parts.append(sanitize_filename(channel).replace(" ", "_"))
    return "_".join(parts) if parts else "debate"


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
    
    # Add candidates (unified schema: use party1/party2 for legislativas, candidate1/candidate2 for presidenciais)
    name1 = str(row.get('candidate1', '')).strip() or str(row.get('party1', '')).strip()
    name2 = str(row.get('candidate2', '')).strip() or str(row.get('party2', '')).strip()

    if name1 and name2:
        parts.append(f"{name1} vs {name2}")
    elif name1:
        parts.append(name1)
    elif name2:
        parts.append(name2)
    
    # Add channel if available
    channel = str(row.get('channel', '')).strip()
    if channel:
        parts.append(channel)
    
    # Join all parts
    title = " - ".join(parts) if parts else "debate"
    
    return title

def check_already_downloaded(title, date_str=None, filename_base=None):
    """
    Check if a debate already exists in the download folder.
    Prefers filename_base (clean format) when provided.
    """
    if not SKIP_EXISTING:
        return False

    if not DOWNLOAD_FOLDER.exists():
        return False

    if filename_base:
        pattern = f"{filename_base}.*"
        for file in DOWNLOAD_FOLDER.glob(pattern):
            if file.is_file():
                return True
        return False

    # Legacy: match by formatted_date + sanitized_title
    if date_str:
        try:
            date_obj = datetime.strptime(str(date_str), "%Y-%m-%d")
            formatted_date = date_obj.strftime("%Y_%m_%d")
        except Exception:
            formatted_date = None
    else:
        formatted_date = None

    sanitized_title = sanitize_filename(title)
    pattern = f"{formatted_date}_{sanitized_title}.*" if formatted_date else f"*{sanitized_title}*"
    for file in DOWNLOAD_FOLDER.glob(pattern):
        if file.is_file():
            return True

    return False

def _model_name_for_path(model_name: str) -> str:
    """Sanitize model name for use in transcript filenames (matches transcribe_audio)."""
    return model_name.replace("/", "_").replace("\\", "_")


def transcript_exists_for_debate(filename_base: str, model: str) -> bool:
    """Return True if a transcript already exists for this debate with the given model."""
    if not model:
        return False
    label = _model_name_for_path(model)
    transcript_path = Path("data/transcripts") / f"{filename_base}_{label}.txt"
    return transcript_path.exists()


def download_all_debates(csv_file=None, download_audio_only=True, audio_format="mp3", model=None):
    """
    Download all debates from the CSV file

    Args:
        csv_file: Path to CSV file (defaults to CSV_FILE global)
        download_audio_only: Whether to download audio only
        audio_format: Audio format for extraction
        model: Optional ASR model name. When provided, skips download if transcript exists.
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
        url = str(row.get("url", "")).strip()
        title = create_debate_title(row)
        filename_base = create_debate_filename(row)
        date_str = row.get("date", "").strip() if row.get("date") else None

        logger.info("=" * 80)
        logger.info(f"📥 Debate {idx + 1}/{len(valid_rows)}: {title}")
        logger.info(f"🔗 URL: {url}")

        # Check if already downloaded
        if check_already_downloaded(title, date_str, filename_base):
            logger.info("⏭️  Already downloaded, skipping...")
            skipped += 1
            logger.info("")
            continue

        # Skip download if transcript already exists for this model
        if transcript_exists_for_debate(filename_base, model):
            logger.info("⏭️  Transcript already exists for this model, skipping download...")
            skipped += 1
            logger.info("")
            continue

        try:
            # Download the debate
            get_debate_audio(
                page_url=url,
                download_audio_only=download_audio_only,
                audio_format=audio_format,
                title=title,
                filename_base=filename_base,
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
