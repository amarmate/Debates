(function () {
  const toggleBtn = document.getElementById('toggleBtn');
  const statusEl = document.getElementById('status');
  const transcriptEl = document.getElementById('transcript');
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');

  let audioContext = null;
  let stream = null;
  let ws = null;
  let processor = null;
  let analyser = null;
  let source = null;
  let animationId = null;
  let isListening = false;

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

  async function start() {
    if (isListening) return;

    try {
      setStatus('Requesting microphone access…');
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      setStatus('Microphone access denied. Please allow and try again.');
      return;
    }

    const wsProtocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = async () => {
      setStatus('Connecting…');
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const sampleRate = audioContext.sampleRate;

      source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0;  // no smoothing for live waveform

      processor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
      processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        const output = e.outputBuffer.getChannelData(0);
        output.set(input);  // pass through so analyser receives audio
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
      muteGain.connect(audioContext.destination);  // graph must reach destination; gain 0 = no playback

      drawVisualizer();  // start visualizer as soon as mic is connected

      const langSelect = document.getElementById('language');
      const modelSelect = document.getElementById('modelSize');
      const language = (langSelect && langSelect.value) ? langSelect.value : null;
      const model_size = (modelSelect && modelSelect.value) ? modelSelect.value : 'small';
      ws.send(JSON.stringify({ sample_rate: sampleRate, language: language, model_size: model_size }));
    };

    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data);
      if (data.type === 'ready') {
        setStatus('Listening. Speak into your microphone.');
        isListening = true;
        toggleBtn.textContent = 'Stop';
        toggleBtn.classList.add('listening');
      } else if (data.type === 'transcript' && data.text) {
        appendTranscript(data.text);
      }
    };

    ws.onerror = () => {
      setStatus('WebSocket error. Is the server running?');
    };

    ws.onclose = () => {
      if (isListening) stop();
    };
  }

  function stop() {
    isListening = false;
    toggleBtn.textContent = 'Start talking';
    toggleBtn.classList.remove('listening');
    setStatus('Stopped. Click Start to begin again.');

    stopVisualizer();

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

  toggleBtn.addEventListener('click', () => {
    if (isListening) stop();
    else start();
  });
})();
