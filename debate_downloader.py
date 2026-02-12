import logging
import re
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright
import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
PAGE_URL = "https://www.rtp.pt/play/p15847/e890933/eleicoes-presidenciais-2026-debates"
DOWNLOAD_AUDIO_ONLY = True
AUDIO_FORMAT = "mp3"
DOWNLOAD_FOLDER = "data/debates"  # Folder where downloads will be saved
TEMP_FOLDER = "data/temp"  # Folder for temporary files during download
# ============================================================================

VIDEO_ID = PAGE_URL.split("/")[-1]

def sanitize_filename(text):
    """Remove invalid characters from filename"""
    # Remove invalid filename characters
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing spaces and dots
    text = text.strip(' .')
    return text

def extract_title(page):
    """Extract title from the webpage"""
    title = None
    
    try:
        # First try structured data (JSON-LD)
        try:
            json_ld = page.evaluate("""
                () => {
                    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                    for (let script of scripts) {
                        try {
                            const data = JSON.parse(script.textContent);
                            if (data.name || data.headline || data.title) {
                                return data.name || data.headline || data.title;
                            }
                            if (data['@type'] === 'VideoObject' && data.name) {
                                return data.name;
                            }
                        } catch(e) {}
                    }
                    return null;
                }
            """)
            if json_ld:
                title = json_ld
                logger.info(f"   📌 Title found via JSON-LD: {title}")
        except:
            pass
        
        # Try meta tags (more reliable)
        if not title:
            meta_selectors = [
                'meta[property="og:title"]',
                'meta[name="title"]',
                'meta[property="twitter:title"]',
                'meta[itemprop="name"]'
            ]
            for selector in meta_selectors:
                try:
                    element = page.query_selector(selector)
                    if element:
                        title = element.get_attribute('content')
                        if title and len(title.strip()) > 0:
                            title = title.strip()
                            logger.info(f"   📌 Title found via meta tag ({selector}): {title}")
                            break
                except:
                    continue
        
        # Try heading elements (more specific)
        if not title:
            heading_selectors = [
                'h1.episode-title',
                'h1.program-title',
                'h1.video-title',
                'h1[class*="title"]',
                'h1'
            ]
            for selector in heading_selectors:
                try:
                    element = page.query_selector(selector)
                    if element:
                        title = element.inner_text()
                        if title and len(title.strip()) > 0:
                            title = title.strip()
                            # Filter out generic titles
                            if title.lower() not in ['rtp', 'play', 'vídeo', 'video', 'debate']:
                                logger.info(f"   📌 Title found via heading ({selector}): {title}")
                                break
                except:
                    continue
        
        # Try JavaScript search for title-like elements
        if not title:
            try:
                js_title = page.evaluate("""
                    () => {
                        // Look for common title patterns
                        const selectors = [
                            '[data-title]',
                            '[data-name]',
                            '.title',
                            '.Title',
                            '.episode-title',
                            '.program-title',
                            '.video-title',
                            'h1, h2'
                        ];
                        
                        for (let sel of selectors) {
                            const els = document.querySelectorAll(sel);
                            for (let el of els) {
                                const text = el.textContent || el.getAttribute('data-title') || el.getAttribute('data-name');
                                if (text && text.trim().length > 5 && 
                                    !text.toLowerCase().includes('rtp') && 
                                    !text.toLowerCase().includes('play')) {
                                    return text.trim();
                                }
                            }
                        }
                        return null;
                    }
                """)
                if js_title:
                    title = js_title
                    logger.info(f"   📌 Title found via JavaScript search: {title}")
            except:
                pass
        
        # Fallback to page title (but clean it)
        if not title:
            try:
                page_title = page.title()
                if page_title:
                    # Remove common suffixes
                    title = page_title.split('|')[0].split('-')[0].strip()
                    if len(title) > 3:
                        logger.info(f"   📌 Title found via page.title(): {title}")
            except:
                pass
        
        # If still no title found, use a default
        if not title or len(title.strip()) < 3:
            title = "debate"
            logger.warning("   ⚠️  Title not found, using 'debate'")
            
    except Exception as e:
        logger.warning(f"⚠️  Error extracting title: {e}")
        title = "debate"
    
    return title

def extract_date(page):
    """Extract date from the webpage"""
    date = None
    
    try:
        # First try structured data (JSON-LD)
        try:
            json_ld_date = page.evaluate("""
                () => {
                    const scripts = document.querySelectorAll('script[type="application/ld+json"]');
                    for (let script of scripts) {
                        try {
                            const data = JSON.parse(script.textContent);
                            if (data.datePublished || data.uploadDate || data.publishedTime) {
                                return data.datePublished || data.uploadDate || data.publishedTime;
                            }
                            if (data['@type'] === 'VideoObject' && data.uploadDate) {
                                return data.uploadDate;
                            }
                        } catch(e) {}
                    }
                    return null;
                }
            """)
            if json_ld_date:
                try:
                    # Try ISO format first
                    date = datetime.fromisoformat(json_ld_date.replace('Z', '+00:00'))
                    logger.info(f"   📌 Date found via JSON-LD: {date.strftime('%Y-%m-%d')}")
                except:
                    pass
        except:
            pass
        
        # Try meta tags (most reliable)
        if not date:
            meta_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publish-date"]',
                'meta[property="video:published_time"]',
                'meta[itemprop="datePublished"]',
                'meta[name="date"]',
                'meta[property="og:published_time"]'
            ]
            for selector in meta_selectors:
                try:
                    element = page.query_selector(selector)
                    if element:
                        date_str = element.get_attribute('content')
                        if date_str:
                            # Try to parse ISO format dates
                            try:
                                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                                logger.info(f"   📌 Date found via meta tag ({selector}): {date.strftime('%Y-%m-%d')}")
                                break
                            except:
                                # Try other common formats
                                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%dT%H:%M:%S']:
                                    try:
                                        date = datetime.strptime(date_str[:10], fmt)
                                        logger.info(f"   📌 Date found via meta tag ({selector}): {date.strftime('%Y-%m-%d')}")
                                        break
                                    except:
                                        continue
                except:
                    continue
        
        # Try time elements
        if not date:
            try:
                time_elements = page.query_selector_all('time')
                for time_el in time_elements:
                    datetime_attr = time_el.get_attribute('datetime')
                    if datetime_attr:
                        try:
                            date = datetime.fromisoformat(datetime_attr.replace('Z', '+00:00'))
                            logger.info(f"   📌 Date found via time element: {date.strftime('%Y-%m-%d')}")
                            break
                        except:
                            pass
            except:
                pass
        
        # Try JavaScript search for date-like elements
        if not date:
            try:
                js_date = page.evaluate("""
                    () => {
                        // Look for common date patterns
                        const selectors = [
                            '[data-date]',
                            '[data-published]',
                            '[class*="date"]',
                            '[class*="Date"]',
                            '[class*="published"]',
                            '[class*="Published"]',
                            'time'
                        ];
                        
                        for (let sel of selectors) {
                            const els = document.querySelectorAll(sel);
                            for (let el of els) {
                                const dateStr = el.getAttribute('datetime') || 
                                               el.getAttribute('data-date') || 
                                               el.getAttribute('data-published') ||
                                               el.textContent;
                                if (dateStr && /\\d{4}[\\-\\/]\\d{2}[\\-\\/]\\d{2}/.test(dateStr)) {
                                    return dateStr.trim();
                                }
                            }
                        }
                        return null;
                    }
                """)
                if js_date:
                    # Try to parse the found date string
                    for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%d-%m-%Y']:
                        try:
                            match = re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', js_date)
                            if match:
                                date_str = match.group(0).replace('/', '-')
                                if len(date_str.split('-')[0]) == 4:  # YYYY-MM-DD
                                    date = datetime.strptime(date_str, '%Y-%m-%d')
                                else:  # DD-MM-YYYY
                                    date = datetime.strptime(date_str, '%d-%m-%Y')
                                logger.info(f"   📌 Date found via JavaScript search: {date.strftime('%Y-%m-%d')}")
                                break
                        except:
                            continue
            except:
                pass
        
        # Fallback: try to extract date from URL or page content
        if not date:
            # Look for date patterns in page content
            try:
                page_content = page.content()
                date_patterns = [
                    (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),  # YYYY-MM-DD
                    (r'(\d{2})/(\d{2})/(\d{4})', '%d/%m/%Y'),  # DD/MM/YYYY
                    (r'(\d{2})-(\d{2})-(\d{4})', '%d-%m-%Y'),  # DD-MM-YYYY
                ]
                
                for pattern, fmt in date_patterns:
                    match = re.search(pattern, page_content)
                    if match:
                        try:
                            date = datetime.strptime(match.group(0), fmt)
                            logger.info(f"   📌 Date found via regex in content: {date.strftime('%Y-%m-%d')}")
                            break
                        except:
                            continue
            except:
                pass
        
        # If still no date found, use current date
        if not date:
            date = datetime.now()
            logger.warning("   ⚠️  Date not found, using current date")
        
    except Exception as e:
        logger.warning(f"⚠️  Error extracting date: {e}")
        if not date:
            date = datetime.now()
    
    return date

def calculate_url_score(url, video_id):
    score = 0
    
    if video_id in url:
        score += 100
    
    video_domains = ["video", "stream", "cdn", "media"]
    for domain in video_domains:
        if domain in url.lower():
            score += 20
    
    if "iol.pt" in url.lower():
        score += 15
    
    if "playlist.m3u8" in url.lower():
        score += 10
    
    if url.count("?") > 3:
        score -= 5
    
    tracking_words = ["ping", "track", "analytics", "beacon"]
    for word in tracking_words:
        if word in url.lower():
            score -= 50
    
    return score

def get_debate_audio(page_url=None, download_audio_only=None, audio_format=None, title=None):
    """
    Download debate audio/video from a given URL
    
    Args:
        page_url: URL of the debate page (defaults to PAGE_URL global)
        download_audio_only: Whether to download audio only (defaults to DOWNLOAD_AUDIO_ONLY global)
        audio_format: Audio format for extraction (defaults to AUDIO_FORMAT global)
        title: Optional title for the debate. If not provided, will be extracted automatically from the page
    """
    # Use parameters or fall back to global defaults
    page_url = page_url or PAGE_URL
    download_audio_only = download_audio_only if download_audio_only is not None else DOWNLOAD_AUDIO_ONLY
    audio_format = audio_format or AUDIO_FORMAT
    
    video_id = page_url.split("/")[-1]
    
    found_urls = []
    formatted_date = None
    sanitized_title = None

    logger.info(f"🕵️  Analyzing page to find stream for ID: {video_id}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        excluded_domains = [
            "chartbeat.net",
            "ping.",
            "analytics",
            "tracking",
            "beacon",
            "doubleclick",
            "google-analytics",
            "googletagmanager",
        ]

        def intercept_request(request):
            url = request.url
            
            if "playlist.m3u8" in url or url.endswith(".m3u8"):
                if any(excluded in url.lower() for excluded in excluded_domains):
                    return
                
                if not url.startswith(("http://", "https://")):
                    return
                
                video_domains = ["video", "stream", "cdn", "media", "iol.pt", "rtp.pt"]
                is_video_url = (
                    video_id in url or 
                    any(domain in url.lower() for domain in video_domains)
                )
                
                if is_video_url and url not in found_urls:
                    found_urls.append(url)
                    score = calculate_url_score(url, video_id)
                    logger.info(f"✅ Link found (score: {score}): {url[:150]}...")
        
        def intercept_response(response):
            url = response.url
            
            if "playlist.m3u8" in url or url.endswith(".m3u8"):
                if any(excluded in url.lower() for excluded in excluded_domains):
                    return
                if not url.startswith(("http://", "https://")):
                    return
                
                video_domains = ["video", "stream", "cdn", "media", "iol.pt", "rtp.pt"]
                is_video_url = (
                    video_id in url or 
                    any(domain in url.lower() for domain in video_domains)
                )
                
                if is_video_url and url not in found_urls:
                    found_urls.append(url)
                    score = calculate_url_score(url, video_id)
                    logger.info(f"✅ Link found (response, score: {score}): {url[:150]}...")

        page.on("request", intercept_request)
        page.on("response", intercept_response)

        try:
            page.goto(page_url, wait_until="load", timeout=60000)
        except Exception as e:
            logger.warning(f"⚠️  Timeout loading page, but continuing... ({e})")
        
        # Extract title if not provided
        if not title:
            logger.info("📝 Extracting title from page...")
            title = extract_title(page)
            logger.info(f"📝 Title found: {title}")
        else:
            logger.info(f"📝 Using provided title: {title}")
        
        try:
            page.wait_for_timeout(12000) 
        except Exception:
            pass
        
        try:
            video_sources = page.evaluate("""
                () => {
                    const sources = [];
                    document.querySelectorAll('video').forEach(video => {
                        if (video.src) sources.push(video.src);
                        video.querySelectorAll('source').forEach(source => {
                            if (source.src) sources.push(source.src);
                        });
                    });
                    document.querySelectorAll('source').forEach(source => {
                        if (source.src) sources.push(source.src);
                    });
                    return sources;
                }
            """)
            
            for src in video_sources:
                    if (".m3u8" in src or "playlist.m3u8" in src) and src.startswith(("http://", "https://")):
                        if src not in found_urls:
                            found_urls.append(src)
                            score = calculate_url_score(src, video_id)
                            logger.info(f"✅ Link found (video element, score: {score}): {src[:150]}...")
        except Exception as e:
            logger.warning(f"⚠️  Error searching for video elements: {e}")
        
        # Extract date from the page (title already extracted)
        logger.info("📅 Extracting date from page...")
        date = extract_date(page)
        formatted_date = date.strftime("%Y_%m_%d")
        sanitized_title = sanitize_filename(title)
        
        logger.info(f"📅 Date found: {date.strftime('%Y-%m-%d')}")
        
        browser.close()
    
    m3u8_url = None
    if found_urls:
        scored_urls = [(url, calculate_url_score(url, video_id)) for url in found_urls]
        scored_urls.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"\n📋 Total m3u8 URLs found: {len(found_urls)}")
        for i, (url, score) in enumerate(scored_urls, 1):
            logger.info(f"   {i}. [Score: {score}] {url}")
        
        m3u8_url = scored_urls[0][0]
        logger.info(f"\n✅ Selected URL with highest score ({scored_urls[0][1]}): {m3u8_url[:150]}...")

    if m3u8_url:
        logger.info("⬇️  Starting download with yt-dlp...")
        
        # Create downloads folder if it doesn't exist
        download_dir = Path(DOWNLOAD_FOLDER)
        download_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Download folder: {download_dir.absolute()}")
        
        # Create temp folder if it doesn't exist
        temp_dir = Path(TEMP_FOLDER)
        temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Temp folder: {temp_dir.absolute()}")
        
        # Fallback if date/title extraction failed
        if not formatted_date:
            formatted_date = datetime.now().strftime("%Y_%m_%d")
        if not sanitized_title:
            sanitized_title = "debate"
        
        # Format temp filename (download to temp first) - use absolute paths
        if download_audio_only:
            temp_file_path = temp_dir / f"{formatted_date}_{sanitized_title}.{audio_format}"
            final_file_path = download_dir / f"{formatted_date}_{sanitized_title}.{audio_format}"
            # Use absolute path for yt-dlp output template
            temp_filename = str(temp_file_path.absolute())
            final_filename = str(final_file_path.absolute())
        else:
            temp_file_path = temp_dir / f"{formatted_date}_{sanitized_title}.%(ext)s"
            temp_filename = str(temp_file_path.absolute())
            final_filename = None  # Will be determined after download based on actual file extension
        
        logger.info(f"💾 Temp filename: {temp_filename}")
        if download_audio_only:
            logger.info(f"💾 Final filename: {final_filename}")
        logger.info("")
        
        # Use yt-dlp Python API with progress_hooks for reliable progress display
        def progress_hook(d):
            status = d.get("status")
            if status == "downloading":
                # These keys are set by yt-dlp's internal report_progress (runs first)
                percent = d.get("_percent_str", "N/A")
                speed = d.get("_speed_str", "")
                eta = d.get("_eta_str", "")
                total = d.get("_total_bytes_str") or d.get("_total_bytes_estimate_str", "?")
                frag = ""
                if "fragment_index" in d and "fragment_count" in d:
                    frag = f" (frag {d['fragment_index']}/{d['fragment_count']})"
                logger.info(f"📥 [download] {percent} of {total} at {speed} ETA {eta}{frag}")
            elif status == "finished":
                logger.info(f"📥 [download] Download completed: {Path(d.get('filename', '')).name}")
        
        ydl_opts = {
            "outtmpl": temp_filename,
            "format": "worst" if download_audio_only else "best",
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [progress_hook],
        }
        if download_audio_only:
            ydl_opts["postprocessors"] = [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "0",
            }]
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([m3u8_url])
            return_code = 0
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"❌ Download failed: {e}")
            return_code = 1
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return_code = 1
        
        if return_code == 0:
            # Download successful, move file from temp to final location
            if download_audio_only:
                # Audio file - simple move
                if final_filename is None:
                    logger.error("❌ Final filename not set for audio download")
                    return
                temp_file = Path(temp_filename)
                if temp_file.exists():
                    final_file = Path(final_filename)
                    # Ensure final directory exists
                    final_file.parent.mkdir(parents=True, exist_ok=True)
                    temp_file.rename(final_file)
                    logger.info(f"✅ File moved to: {final_file}")
                else:
                    logger.warning(f"⚠️  Temp file not found: {temp_filename}")
                    # Try to find any file that was created
                    matching_files = list(temp_dir.glob(f"{formatted_date}_{sanitized_title}.*"))
                    if matching_files:
                        logger.info(f"   Found alternative file: {matching_files[0]}")
                        temp_file = matching_files[0]
                        final_file = download_dir / temp_file.name
                        final_file.parent.mkdir(parents=True, exist_ok=True)
                        temp_file.rename(final_file)
                        logger.info(f"✅ File moved to: {final_file}")
            else:
                # Video file - need to find the actual file (ext might vary)
                temp_file_pattern = temp_dir / f"{formatted_date}_{sanitized_title}.*"
                matching_files = list(temp_dir.glob(f"{formatted_date}_{sanitized_title}.*"))
                if matching_files:
                    temp_file = matching_files[0]
                    final_file = download_dir / temp_file.name
                    final_file.parent.mkdir(parents=True, exist_ok=True)
                    temp_file.rename(final_file)
                    logger.info(f"✅ File moved to: {final_file}")
                else:
                    logger.warning(f"⚠️  No matching temp file found for pattern: {temp_file_pattern}")
            
            logger.info("🎉 Process completed! Check the folder.")
        else:
            logger.error(f"❌ Download failed with return code {return_code}")
    else:
        logger.error("❌ Could not find .m3u8 link. The site may have changed or took too long to load.")

if __name__ == "__main__":
    get_debate_audio()
