(function () {
  const toggleBtn = document.getElementById('toggleBtn');
  const statusEl = document.getElementById('status');
  const transcriptEl = document.getElementById('transcript');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
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

  const BUFFER_SIZE = 4096;

  function setStatus(msg) {
    statusEl.textContent = msg;
  }

  function appendTranscript(text) {
    transcriptEl.classList.remove('empty');
    const current = transcriptEl.textContent;
    transcriptEl.textContent = current === 'Transcription will appear here as you speak…' ? text : current + ' ' + text;
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
    return {
      language: (langSelect && langSelect.value) ? langSelect.value : null,
      model_size: (modelSelect && modelSelect.value) ? modelSelect.value : 'small',
    };
  }

  function startMicrophone() {
    return navigator.mediaDevices.getUserMedia({ audio: true });
  }

  async function startFile(readyPromise) {
    const filename = fileSelect && fileSelect.value;
    if (!filename) {
      setStatus('Please select an audio file from the list.');
      return;
    }
    setStatus('Loading file…');
    const resp = await fetch(`/api/audio/${encodeURIComponent(filename)}`);
    if (!resp.ok) {
      setStatus('Failed to load file.');
      return;
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
    bufferSource.onended = () => {
      setStatus('Playback finished. Transcription continues… Click Stop when done.');
      stopVisualizer();
      bufferSource = null;
    };

    const chunkSamples = Math.floor(fileChunkDuration * sampleRate);
    let offset = 0;

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
      return;
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

    transcriptEl.textContent = 'Transcription will appear here as you speak…';
    transcriptEl.classList.add('empty');

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

    if (src === 'file') {
      await startFile(readyPromise);
    } else {
      await startMic();
      await readyPromise;
    }
  }

  function stop() {
    isListening = false;
    toggleBtn.textContent = 'Start';
    toggleBtn.classList.remove('listening');
    setStatus('Stopped. Choose source and click Start again.');

    stopVisualizer();
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
})();
