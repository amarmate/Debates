(function () {
  const toggleBtn = document.getElementById('toggleBtn');
  const statusEl = document.getElementById('status');
  const transcriptEl = document.getElementById('transcript');
  const sentencesList = document.getElementById('sentencesList');
  const debugWrapper = document.getElementById('debugWrapper');
  const debugFramesList = document.getElementById('debugFramesList');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const audioProgressEl = document.getElementById('audioProgress');
  const sourceSelect = document.getElementById('source');
  const fileSelectWrap = document.getElementById('fileSelectWrap');
  const fileSelect = document.getElementById('fileSelect');

  let audioContext = null;
  let stream = null;
  let ws = null;
  let processor = null;
  let analyser = null;
  let source = null;
  let bufferSource = null;
  let animationId = null;
  let isListening = false;
  let fileChunkInterval = null;
  let progressInterval = null;

  const BUFFER_SIZE = 4096;
  const WARMUP_SEC = 2;

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  function setAudioProgress(text) {
    if (audioProgressEl) audioProgressEl.textContent = text;
  }

  function clearAudioProgress() {
    if (progressInterval) {
      clearInterval(progressInterval);
      progressInterval = null;
    }
    setAudioProgress('');
  }

  function appendTranscript(text) {
    transcriptEl.classList.remove('empty');
    const current = transcriptEl.textContent;
    transcriptEl.textContent = current === 'Transcription will appear here as you speak…' ? text : current + ' ' + text;
  }

  function formatTime(seconds) {
    if (seconds === undefined || seconds === null) return '';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return (m > 0 ? m + ':' : '0:') + String(s).padStart(2, '0');
  }

  function appendSentence(text, time) {
    if (!sentencesList || !text) return;
    const entry = document.createElement('div');
    entry.className = 'sentence-item';
    if (time !== undefined && time !== null) {
      const badge = document.createElement('span');
      badge.className = 'sentence-time';
      badge.textContent = '[' + formatTime(time) + ']';
      entry.appendChild(badge);
      entry.appendChild(document.createTextNode(' ' + text));
    } else {
      entry.textContent = text;
    }
    sentencesList.appendChild(entry);
    sentencesList.parentElement.scrollTop = sentencesList.parentElement.scrollHeight;
  }

  function escapeHtml(s) {
    return (s || '').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function appendDebugFrame(timeRange, raw, mergeInfo) {
    if (!debugWrapper || !debugFramesList) return;
    debugWrapper.classList.add('visible');
    const entry = document.createElement('div');
    entry.className = 'debug-frame';
    const [start, end] = timeRange;
    let html = '<div class="debug-frame-label">[' + start + ' - ' + end + ' sec]</div><div class="debug-frame-raw">' + escapeHtml(raw || '(empty)') + '</div>';
    if (mergeInfo && (mergeInfo.new_content !== undefined || mergeInfo.confirmed_units !== undefined)) {
      html += '<div class="debug-frame-merge">';
      if (mergeInfo.confirmed_units !== undefined) html += '<div class="debug-merge-size"><strong>Confirmed:</strong> ' + mergeInfo.confirmed_units + ' words</div>';
      if (mergeInfo.tentative_units !== undefined && mergeInfo.tentative_units > 0) html += '<div class="debug-merge-size"><strong>Tentative (held back):</strong> ' + mergeInfo.tentative_units + ' words</div>';
      if (mergeInfo.dropped_units !== undefined) html += '<div class="debug-merge-size"><strong>Dropped:</strong> ' + mergeInfo.dropped_units + ' words</div>';
      if (mergeInfo.boundary_dropped > 0) html += '<div class="debug-merge-size"><strong>Boundary dedup:</strong> ' + mergeInfo.boundary_dropped + ' word(s)</div>';
      if (mergeInfo.pending_recovered > 0) html += '<div class="debug-merge-size"><strong>Pending recovered:</strong> ' + mergeInfo.pending_recovered + ' word(s)</div>';
      if (mergeInfo.last_committed_before !== undefined && mergeInfo.last_committed_after !== undefined) html += '<div class="debug-merge-size"><strong>Timeline:</strong> ' + mergeInfo.last_committed_before + 's -> ' + mergeInfo.last_committed_after + 's</div>';
      if (mergeInfo.new_content !== undefined) html += '<div class="debug-merge-appended"><strong>Appended:</strong> ' + escapeHtml(mergeInfo.new_content || '(none)') + '</div>';
      html += '</div>';
    }
    entry.innerHTML = html;
    debugFramesList.appendChild(entry);
    debugWrapper.scrollTop = debugWrapper.scrollHeight;
  }

  function drawVisualizer() {
    if (!analyser || !ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const dataArray = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = 'rgba(13, 17, 23, 0.5)';
    ctx.fillRect(0, 0, width, height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = '#58a6ff';
    ctx.beginPath();

    const sliceWidth = width / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
      const v = (dataArray[i] - 128) / 128;
      const y = (v * height * 0.4) + height / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }

    ctx.stroke();
    animationId = requestAnimationFrame(drawVisualizer);
  }

  function stopVisualizer() {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  function getConfig() {
    const langSelect = document.getElementById('language');
    const modelSelect = document.getElementById('modelSize');
    const debugCheck = document.getElementById('debugFrames');
    return {
      language: (langSelect && langSelect.value) ? langSelect.value : null,
      model_size: (modelSelect && modelSelect.value) ? modelSelect.value : 'small',
      debug_frames: debugCheck ? debugCheck.checked : false,
    };
  }

  function startMicrophone() {
    return navigator.mediaDevices.getUserMedia({ audio: true });
  }

  async function startFile(readyPromise) {
    const filename = fileSelect && fileSelect.value;
    if (!filename) {
      setStatus('Please select an audio file from the list.');
      throw new Error('No file selected');
    }
    setStatus('Loading file…');
    const resp = await fetch(`/api/audio/${encodeURIComponent(filename)}`);
    if (!resp.ok) {
      setStatus('Failed to load file.');
      throw new Error('Failed to load file');
    }
    const arrayBuffer = await resp.arrayBuffer();
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const sampleRate = audioBuffer.sampleRate;
    const channels = audioBuffer.numberOfChannels;
    let samples = audioBuffer.getChannelData(0);
    if (channels > 1) {
      for (let c = 1; c < channels; c++) {
        const ch = audioBuffer.getChannelData(c);
        for (let i = 0; i < samples.length; i++) samples[i] += ch[i];
        for (let i = 0; i < samples.length; i++) samples[i] /= channels;
      }
    }
    const int16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    const cfg = getConfig();
    cfg.sample_rate = sampleRate;
    cfg.source = 'file';
    cfg.filename = filename;
    ws.send(JSON.stringify(cfg));

    const warmupSamples = Math.floor(WARMUP_SEC * sampleRate);
    const sentWarmup = warmupSamples >= 2 && int16.length >= warmupSamples;
    if (sentWarmup) {
      const warmupChunk = int16.subarray(0, warmupSamples);
      ws.send(warmupChunk.buffer.slice(warmupChunk.byteOffset, warmupChunk.byteOffset + warmupChunk.byteLength));
    }

    const readyData = await readyPromise;
    const fileChunkDuration = (readyData && readyData.file_chunk_duration) || 4.0;

    const playBuffer = audioContext.createBuffer(1, samples.length, sampleRate);
    playBuffer.copyToChannel(samples, 0);

    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0;
    bufferSource = audioContext.createBufferSource();
    bufferSource.buffer = playBuffer;
    bufferSource.connect(analyser);
    const gainNode = audioContext.createGain();
    gainNode.gain.value = 1;
    analyser.connect(gainNode);
    gainNode.connect(audioContext.destination);
    bufferSource.start(0);

    drawVisualizer();
    const totalDuration = playBuffer.duration;
    const progressStart = performance.now() / 1000;
    progressInterval = setInterval(() => {
      const elapsed = (performance.now() / 1000) - progressStart;
      const pos = Math.min(elapsed, totalDuration);
      setAudioProgress(formatTime(pos) + ' / ' + formatTime(totalDuration));
    }, 200);
    bufferSource.onended = () => {
      setStatus('Playback finished. Transcription continues… Click Stop when done.');
      stopVisualizer();
      clearAudioProgress();
      bufferSource = null;
    };

    const chunkSamples = Math.floor(fileChunkDuration * sampleRate);
    let offset = sentWarmup ? warmupSamples : 0;

    function sendNextChunk() {
      if (offset >= int16.length || ws.readyState !== WebSocket.OPEN) return;
      const end = Math.min(offset + chunkSamples, int16.length);
      const chunk = int16.subarray(offset, end);
      if (chunk.length >= 2) {
        ws.send(chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));
      }
      offset = end;
      if (offset < int16.length) {
        fileChunkInterval = setTimeout(sendNextChunk, fileChunkDuration * 1000);
      }
    }
    setTimeout(sendNextChunk, fileChunkDuration * 1000);
  }

  async function startMic() {
    try {
      setStatus('Requesting microphone access…');
      stream = await startMicrophone();
    } catch (e) {
      setStatus('Microphone access denied. Please allow and try again.');
      throw e;
    }
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const sampleRate = audioContext.sampleRate;

    source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0;

    processor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      const output = e.outputBuffer.getChannelData(0);
      output.set(input);
      if (ws && ws.readyState === WebSocket.OPEN && isListening) {
        const int16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          const s = Math.max(-1, Math.min(1, input[i]));
          int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        ws.send(int16.buffer);
      }
    };

    source.connect(processor);
    processor.connect(analyser);
    const muteGain = audioContext.createGain();
    muteGain.gain.value = 0;
    analyser.connect(muteGain);
    muteGain.connect(audioContext.destination);

    drawVisualizer();

    const recordingStart = performance.now() / 1000;
    progressInterval = setInterval(() => {
      const elapsed = (performance.now() / 1000) - recordingStart;
      setAudioProgress('Recording: ' + formatTime(elapsed));
    }, 200);

    const cfg = getConfig();
    cfg.sample_rate = sampleRate;
    cfg.source = 'mic';
    ws.send(JSON.stringify(cfg));
  }

  function waitForOpen(sock) {
    return new Promise((resolve) => {
      if (sock.readyState === WebSocket.OPEN) resolve();
      else sock.addEventListener('open', () => resolve(), { once: true });
    });
  }

  async function start() {
    if (isListening) return;

    toggleBtn.textContent = 'Stop';
    toggleBtn.classList.add('listening');
    transcriptEl.textContent = 'Transcription will appear here as you speak…';
    transcriptEl.classList.add('empty');
    if (sentencesList) sentencesList.innerHTML = '';
    if (debugFramesList) debugFramesList.innerHTML = '';
    if (debugWrapper) debugWrapper.classList.remove('visible');

    const src = sourceSelect && sourceSelect.value;
    setStatus('Loading model…');
    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${wsProtocol}//${location.host}/ws`);

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === 'ready') {
        setStatus(src === 'file' ? 'Playing file. Transcription in sync.' : 'Listening. Speak into your microphone.');
        isListening = true;
        toggleBtn.textContent = 'Stop';
        toggleBtn.classList.add('listening');
      } else if (data.type === 'transcript' && data.text) {
        appendTranscript(data.text);
      } else if (data.type === 'sentence' && data.text) {
        appendSentence(data.text, data.time);
      } else if (data.type === 'debug_frame' && data.time_range) {
        appendDebugFrame(data.time_range, data.raw, data.merge_info);
      }
    };

    ws.onerror = () => setStatus('WebSocket error. Is the server running?');
    ws.onclose = () => { if (isListening) stop(); };

    await waitForOpen(ws);
    setStatus('Loading model…');

    let resolveReady;
    const readyPromise = new Promise((r) => { resolveReady = r; });
    const origOnMessage = ws.onmessage;
    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === 'ready' && resolveReady) {
        resolveReady(data);
        resolveReady = null;
      }
      origOnMessage(ev);
    };

    try {
      if (src === 'file') {
        await startFile(readyPromise);
      } else {
        await startMic();
        await readyPromise;
      }
    } catch (e) {
      stop();
    }
  }

  function stop() {
    isListening = false;
    toggleBtn.textContent = 'Start';
    toggleBtn.classList.remove('listening');
    setStatus('Stopped. Choose source and click Start again.');

    stopVisualizer();
    clearAudioProgress();
    if (fileChunkInterval) {
      clearTimeout(fileChunkInterval);
      fileChunkInterval = null;
    }

    if (bufferSource) {
      try { bufferSource.stop(); } catch (_) {}
      bufferSource = null;
    }
    if (processor && source) {
      try {
        source.disconnect(processor);
        processor.disconnect();
      } catch (_) {}
    }
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    if (ws) {
      ws.close();
      ws = null;
    }
    processor = null;
    analyser = null;
    source = null;
  }

  async function loadFileList() {
    const sel = fileSelect;
    if (!sel) return;
    try {
      const resp = await fetch('/api/audio/files');
      const files = await resp.json();
      sel.innerHTML = '<option value="">– select file –</option>';
      for (const f of files) {
        sel.appendChild(new Option(f, f));
      }
    } catch (_) {
      sel.innerHTML = '<option value="">– none available –</option>';
    }
  }

  sourceSelect.addEventListener('change', () => {
    const isFile = sourceSelect.value === 'file';
    fileSelectWrap.style.display = isFile ? 'flex' : 'none';
    if (isFile) loadFileList();
    setStatus(isFile ? 'Select a file from the list, then click Start.' : 'Click Start to begin. Allow microphone access when prompted.');
  });

  toggleBtn.addEventListener('click', () => {
    if (isListening) stop();
    else start();
  });

  // Config page
  const navTranscription = document.getElementById('navTranscription');
  const navSettings = document.getElementById('navSettings');
  const pageTranscription = document.getElementById('pageTranscription');
  const pageSettings = document.getElementById('pageSettings');

  const configFields = {
    SILENCE_THRESHOLD: () => document.getElementById('cfgSilenceThreshold'),
    SILENCE_DURATION_MS: () => document.getElementById('cfgSilenceDurationMs'),
    MIN_CHUNK_DURATION: () => document.getElementById('cfgMinChunkDuration'),
    MAX_CHUNK_DURATION: () => document.getElementById('cfgMaxChunkDuration'),
    ROLLING_INTERVAL_SEC: () => document.getElementById('cfgRollingIntervalSec'),
    ROLLING_BUFFER_SEC: () => document.getElementById('cfgRollingBufferSec'),
    DEVICE: () => document.getElementById('cfgDevice'),
    COMPUTE_TYPE: () => document.getElementById('cfgComputeType'),
    CONTEXT_WINDOW_SIZE: () => document.getElementById('cfgContextWindowSize'),
    REPETITION_PENALTY: () => document.getElementById('cfgRepetitionPenalty'),
    COMPRESSION_RATIO_THRESHOLD: () => document.getElementById('cfgCompressionRatioThreshold'),
    INITIAL_PROMPT_ENABLED: () => document.getElementById('cfgInitialPromptEnabled'),
    CONTEXT_INJECTION_ENABLED: () => document.getElementById('cfgContextInjectionEnabled'),
    VAD_FILTER: () => document.getElementById('cfgVadFilter'),
    PUNCTUATION_RESTORE: () => document.getElementById('cfgPunctuationRestore'),
    TRIM_SILENCE_FILE_CHUNKS: () => document.getElementById('cfgTrimSilenceFileChunks'),
    DEBUG_MODE: () => document.getElementById('cfgDebugMode'),
  };

  const configDefaults = {
    SILENCE_THRESHOLD: 0.01,
    SILENCE_DURATION_MS: 500,
    MIN_CHUNK_DURATION: 1,
    MAX_CHUNK_DURATION: 10,
    ROLLING_INTERVAL_SEC: 2,
    ROLLING_BUFFER_SEC: 14,
    DEVICE: 'auto',
    COMPUTE_TYPE: 'auto',
    CONTEXT_WINDOW_SIZE: 450,
    REPETITION_PENALTY: 1.1,
    COMPRESSION_RATIO_THRESHOLD: 3,
    INITIAL_PROMPT_ENABLED: true,
    CONTEXT_INJECTION_ENABLED: true,
    VAD_FILTER: false,
    PUNCTUATION_RESTORE: true,
    TRIM_SILENCE_FILE_CHUNKS: false,
    DEBUG_MODE: false,
  };

  function showPage(page) {
    if (page === 'transcription') {
      pageTranscription.classList.add('active');
      pageSettings.classList.remove('active');
      navTranscription.classList.add('active');
      navSettings.classList.remove('active');
    } else {
      pageTranscription.classList.remove('active');
      pageSettings.classList.add('active');
      navTranscription.classList.remove('active');
      navSettings.classList.add('active');
      loadConfig();
    }
  }

  async function loadConfig() {
    try {
      const resp = await fetch('/api/config');
      const cfg = await resp.json();
      for (const [key, elFn] of Object.entries(configFields)) {
        const el = elFn();
        if (!el || !(key in cfg)) continue;
        const v = cfg[key];
        if (el.type === 'checkbox') el.checked = !!v;
        else if (el.tagName === 'SELECT') el.value = String(v);
        else el.value = v;
      }
    } catch (e) {
      const statusEl = document.getElementById('configStatus');
      if (statusEl) statusEl.textContent = 'Failed to load config';
    }
  }

  async function saveConfig() {
    const statusEl = document.getElementById('configStatus');
    const cfg = {};
    for (const [key, elFn] of Object.entries(configFields)) {
      const el = elFn();
      if (!el) continue;
      if (el.type === 'checkbox') cfg[key] = el.checked;
      else if (el.tagName === 'SELECT') cfg[key] = el.value;
      else cfg[key] = el.type === 'number' ? parseFloat(el.value) || 0 : el.value;
    }
    try {
      const resp = await fetch('/api/config', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(cfg),
      });
      if (resp.ok) {
        statusEl.textContent = 'Saved.';
        setTimeout(() => { statusEl.textContent = ''; }, 2000);
      } else statusEl.textContent = 'Save failed.';
    } catch (e) {
      statusEl.textContent = 'Save failed: ' + e.message;
    }
  }

  function applyConfigToForm(cfg) {
    for (const [key, elFn] of Object.entries(configFields)) {
      const el = elFn();
      if (!el) continue;
      const v = key in cfg ? cfg[key] : configDefaults[key];
      if (el.type === 'checkbox') el.checked = !!v;
      else if (el.tagName === 'SELECT') el.value = String(v);
      else el.value = v;
    }
  }

  navTranscription.addEventListener('click', (e) => {
    e.preventDefault();
    showPage('transcription');
  });
  navSettings.addEventListener('click', (e) => {
    e.preventDefault();
    showPage('settings');
  });

  document.getElementById('configSaveBtn').addEventListener('click', saveConfig);
  document.getElementById('configResetBtn').addEventListener('click', async () => {
    const statusEl = document.getElementById('configStatus');
    try {
      const resp = await fetch('/api/config/reset', { method: 'POST' });
      const cfg = resp.ok ? await resp.json() : null;
      if (cfg) {
        applyConfigToForm(cfg);
        statusEl.textContent = 'Reset to defaults.';
      } else statusEl.textContent = 'Reset failed.';
      setTimeout(() => { statusEl.textContent = ''; }, 2000);
    } catch (e) {
      statusEl.textContent = 'Reset failed: ' + e.message;
    }
  });
})();
