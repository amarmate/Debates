# Debates Audio Extractor

Script to extract audio from CNN Portugal debate videos.

## Setup with uv

1. **Create virtual environment:**
   ```powershell
   uv venv
   ```

2. **Activate the virtual environment:**
   ```powershell
   .venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```powershell
   uv pip install playwright
   ```

4. **Install Playwright browsers:**
   ```powershell
   playwright install chromium
   ```

5. **Install yt-dlp (required for downloading):**
   
   You can install `yt-dlp` either:
   
   - **Via pip (recommended):**
     ```powershell
     uv pip install yt-dlp
     ```
   
   - **Or via standalone installer:**
     Download from https://github.com/yt-dlp/yt-dlp/releases or use:
     ```powershell
     pip install yt-dlp
     ```

## Usage

```powershell
python sacar_debates.py
```

Make sure the virtual environment is activated before running the script.

## Dependencies

- `playwright` - For browser automation to intercept network requests
- `yt-dlp` - For downloading and converting video streams to audio


