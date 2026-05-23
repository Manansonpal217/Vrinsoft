"""REST API + browser chat UI for the customer-support reply assistant.

Endpoints
---------
GET  /          - browser chat interface
GET  /health    - service / model status
POST /reply     - JSON {"message": "...", "decoding": "greedy"|"sample"}
                  -> JSON {"reply": "...", "decoding": "..."}

Run:  python -m src.api      (then open http://localhost:8000)
"""
from flask import Flask, jsonify, request

from src import config
from src.inference import ModelNotTrainedError, get_generator

app = Flask(__name__)

# Load the model once at startup (None if it has not been trained yet).
try:
    GENERATOR = get_generator()
    LOAD_ERROR = None
except ModelNotTrainedError as exc:
    GENERATOR, LOAD_ERROR = None, str(exc)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Support Reply Assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-0: #07080d;
    --bg-1: #0f1118;
    --bg-2: #161922;
    --surface: rgba(22, 25, 34, 0.72);
    --surface-strong: rgba(28, 32, 44, 0.92);
    --border: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.14);
    --text: #f4f6fb;
    --text-muted: #9aa3b5;
    --text-soft: #c5ccd9;
    --accent: #8b7cff;
    --accent-2: #5eead4;
    --accent-glow: rgba(139, 124, 255, 0.35);
    --customer: linear-gradient(135deg, #6d5efc 0%, #8b7cff 55%, #a78bfa 100%);
    --support: rgba(255, 255, 255, 0.06);
    --danger: #ff7b8a;
    --shadow: 0 24px 80px rgba(0, 0, 0, 0.45);
    --radius-xl: 28px;
    --radius-lg: 18px;
    --radius-md: 14px;
    --radius-sm: 10px;
  }

  * { box-sizing: border-box; }

  html, body {
    height: 100%;
    margin: 0;
  }

  body {
    font-family: "DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--text);
    background:
      radial-gradient(circle at 15% 10%, rgba(139, 124, 255, 0.18), transparent 28%),
      radial-gradient(circle at 85% 0%, rgba(94, 234, 212, 0.12), transparent 24%),
      radial-gradient(circle at 50% 100%, rgba(109, 94, 252, 0.08), transparent 30%),
      linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 45%, #0b0d13 100%);
    overflow: hidden;
  }

  body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.35;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.5'/%3E%3C/svg%3E");
    mix-blend-mode: soft-light;
  }

  .app {
    position: relative;
    z-index: 1;
    height: 100vh;
    display: grid;
    place-items: center;
    padding: 24px;
  }

  .shell {
    width: min(920px, 100%);
    height: min(860px, calc(100vh - 48px));
    display: flex;
    flex-direction: column;
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    background: var(--surface);
    backdrop-filter: blur(24px) saturate(140%);
    box-shadow: var(--shadow);
    overflow: hidden;
  }

  .header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 22px 26px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(255,255,255,0.04), transparent);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 0;
  }

  .brand-mark {
    width: 46px;
    height: 46px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: linear-gradient(145deg, rgba(139,124,255,0.25), rgba(94,234,212,0.12));
    border: 1px solid rgba(139, 124, 255, 0.28);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 10px 30px rgba(109, 94, 252, 0.18);
    flex-shrink: 0;
  }

  .brand-mark svg {
    width: 22px;
    height: 22px;
    color: #ddd6fe;
  }

  .brand-copy h1 {
    margin: 0;
    font-family: "Instrument Serif", Georgia, serif;
    font-size: clamp(1.25rem, 2vw, 1.55rem);
    font-weight: 400;
    letter-spacing: 0.01em;
    line-height: 1.1;
  }

  .brand-copy p {
    margin: 4px 0 0;
    color: var(--text-muted);
    font-size: 0.82rem;
  }

  .status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.03);
    color: var(--text-soft);
    font-size: 0.78rem;
    white-space: nowrap;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-2);
    box-shadow: 0 0 0 4px rgba(94, 234, 212, 0.12);
    animation: pulse 2.4s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(0.92); opacity: 0.75; }
  }

  .chat {
    flex: 1;
    overflow-y: auto;
    padding: 28px 24px 18px;
    scroll-behavior: smooth;
  }

  .chat::-webkit-scrollbar { width: 8px; }
  .chat::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.12);
    border-radius: 999px;
  }

  .welcome {
    max-width: 560px;
    margin: 8vh auto 0;
    text-align: center;
    animation: fadeUp 0.7s ease both;
  }

  .welcome-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    margin-bottom: 18px;
    border-radius: 999px;
    border: 1px solid rgba(139, 124, 255, 0.22);
    background: rgba(139, 124, 255, 0.08);
    color: #c4b5fd;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .welcome h2 {
    margin: 0 0 10px;
    font-family: "Instrument Serif", Georgia, serif;
    font-size: clamp(1.8rem, 4vw, 2.5rem);
    font-weight: 400;
    line-height: 1.05;
  }

  .welcome p {
    margin: 0 auto 24px;
    max-width: 42ch;
    color: var(--text-muted);
    line-height: 1.6;
    font-size: 0.98rem;
  }

  .suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
  }

  .suggestion {
    appearance: none;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.03);
    color: var(--text-soft);
    padding: 11px 14px;
    border-radius: 999px;
    font: inherit;
    font-size: 0.88rem;
    cursor: pointer;
    transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
  }

  .suggestion:hover {
    transform: translateY(-1px);
    border-color: rgba(139, 124, 255, 0.35);
    background: rgba(139, 124, 255, 0.08);
  }

  .messages {
    display: flex;
    flex-direction: column;
    gap: 18px;
  }

  .row {
    display: flex;
    gap: 12px;
    align-items: flex-end;
    animation: fadeUp 0.35s ease both;
  }

  .row.customer { flex-direction: row-reverse; }

  .avatar {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    flex-shrink: 0;
    border: 1px solid var(--border);
  }

  .row.customer .avatar {
    background: rgba(139, 124, 255, 0.18);
    color: #ddd6fe;
  }

  .row.support .avatar {
    background: rgba(94, 234, 212, 0.1);
    color: #99f6e4;
  }

  .bubble-wrap {
    max-width: min(72%, 560px);
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .row.customer .bubble-wrap { align-items: flex-end; }

  .meta {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-muted);
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .bubble {
    padding: 14px 16px;
    border-radius: var(--radius-lg);
    line-height: 1.55;
    font-size: 0.96rem;
    word-break: break-word;
    border: 1px solid transparent;
  }

  .row.customer .bubble {
    background: var(--customer);
    color: #fff;
    border-bottom-right-radius: 6px;
    box-shadow: 0 12px 30px rgba(109, 94, 252, 0.22);
  }

  .row.support .bubble {
    background: var(--support);
    color: var(--text);
    border-color: var(--border);
    border-bottom-left-radius: 6px;
    backdrop-filter: blur(8px);
  }

  .row.error .bubble {
    background: rgba(255, 123, 138, 0.08);
    border-color: rgba(255, 123, 138, 0.22);
    color: #ffd5da;
  }

  .typing .bubble {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 72px;
    padding: 16px 18px;
  }

  .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #a5b4fc;
    animation: bounce 1.2s infinite ease-in-out;
  }

  .dot:nth-child(2) { animation-delay: 0.15s; }
  .dot:nth-child(3) { animation-delay: 0.3s; }

  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.45; }
    40% { transform: translateY(-5px); opacity: 1; }
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .composer {
    padding: 18px 20px 20px;
    border-top: 1px solid var(--border);
    background: linear-gradient(180deg, transparent, rgba(255,255,255,0.02));
  }

  .composer-inner {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 14px;
    border-radius: 22px;
    border: 1px solid var(--border-strong);
    background: var(--surface-strong);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  }

  .composer-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }

  .mode-toggle {
    display: inline-flex;
    padding: 4px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.03);
  }

  .mode-btn {
    appearance: none;
    border: none;
    background: transparent;
    color: var(--text-muted);
    font: inherit;
    font-size: 0.78rem;
    padding: 7px 12px;
    border-radius: 999px;
    cursor: pointer;
    transition: background 0.18s ease, color 0.18s ease, box-shadow 0.18s ease;
  }

  .mode-btn.active {
    color: #fff;
    background: rgba(139, 124, 255, 0.22);
    box-shadow: inset 0 0 0 1px rgba(139, 124, 255, 0.28);
  }

  .hint {
    color: var(--text-muted);
    font-size: 0.76rem;
  }

  .input-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
  }

  #text {
    flex: 1;
    resize: none;
    min-height: 24px;
    max-height: 140px;
    border: none;
    outline: none;
    background: transparent;
    color: var(--text);
    font: inherit;
    font-size: 1rem;
    line-height: 1.5;
    padding: 4px 2px;
  }

  #text::placeholder { color: #6b7287; }

  .send-btn {
    width: 46px;
    height: 46px;
    border: none;
    border-radius: 14px;
    display: grid;
    place-items: center;
    cursor: pointer;
    color: #fff;
    background: var(--customer);
    box-shadow: 0 10px 24px var(--accent-glow);
    transition: transform 0.18s ease, opacity 0.18s ease, box-shadow 0.18s ease;
    flex-shrink: 0;
  }

  .send-btn:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(109, 94, 252, 0.32);
  }

  .send-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
    transform: none;
  }

  .send-btn svg {
    width: 18px;
    height: 18px;
  }

  @media (max-width: 720px) {
    .app { padding: 0; }
    .shell {
      width: 100%;
      height: 100vh;
      border-radius: 0;
      border: none;
    }
    .header { padding: 18px 16px; }
    .chat { padding: 20px 14px 12px; }
    .composer { padding: 14px; }
    .bubble-wrap { max-width: 84%; }
    .status span:last-child { display: none; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="shell">
    <header class="header">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>
          </svg>
        </div>
        <div class="brand-copy">
          <h1>Support Reply Assistant</h1>
          <p>Seq2seq + attention &middot; trained on social support conversations</p>
        </div>
      </div>
      <div class="status" id="status">
        <span class="status-dot"></span>
        <span>Model online</span>
      </div>
    </header>

    <main class="chat" id="chat">
      <section class="welcome" id="welcome">
        <div class="welcome-badge">Live demo</div>
        <h2>How can we help today?</h2>
        <p>Type a customer message below and watch the model craft a contextual support reply in real time.</p>
        <div class="suggestions">
          <button type="button" class="suggestion" data-text="my internet has been down all day, please help">Internet outage</button>
          <button type="button" class="suggestion" data-text="my order still has not arrived and I need a refund">Missing order</button>
          <button type="button" class="suggestion" data-text="the app keeps crashing every time I try to log in">App crashing</button>
        </div>
      </section>
      <div class="messages" id="messages" hidden></div>
    </main>

    <footer class="composer">
      <form id="f" class="composer-inner">
        <div class="composer-top">
          <div class="mode-toggle" role="group" aria-label="Decoding mode">
            <button type="button" class="mode-btn active" data-mode="greedy">Greedy</button>
            <button type="button" class="mode-btn" data-mode="sample">Sample</button>
          </div>
          <span class="hint">Enter to send &middot; Shift+Enter for new line</span>
        </div>
        <div class="input-row">
          <textarea id="text" rows="1" placeholder="Describe the customer issue..." autocomplete="off" autofocus></textarea>
          <button class="send-btn" id="send" type="submit" aria-label="Send message">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>
            </svg>
          </button>
        </div>
      </form>
    </footer>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const welcome = document.getElementById('welcome');
const messages = document.getElementById('messages');
const text = document.getElementById('text');
const sendBtn = document.getElementById('send');
const statusEl = document.getElementById('status');
let decoding = 'greedy';
let busy = false;
let typingEl = null;

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}

function resizeInput() {
  text.style.height = 'auto';
  text.style.height = Math.min(text.scrollHeight, 140) + 'px';
}

function setBusy(next) {
  busy = next;
  sendBtn.disabled = next;
  text.disabled = next;
}

function hideWelcome() {
  if (!welcome.hidden) {
    welcome.hidden = true;
    messages.hidden = false;
  }
}

function scrollToBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function createRow(who, bodyHtml, extraClass = '') {
  const row = document.createElement('div');
  row.className = 'row ' + who + (extraClass ? ' ' + extraClass : '');
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = who === 'customer' ? 'YOU' : 'AI';
  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap';
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.innerHTML = '<span>' + (who === 'customer' ? 'Customer' : 'Support agent') + '</span><span>' + timeNow() + '</span>';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = bodyHtml;
  wrap.appendChild(meta);
  wrap.appendChild(bubble);
  row.appendChild(avatar);
  row.appendChild(wrap);
  return row;
}

function addMessage(textValue, who, extraClass = '') {
  hideWelcome();
  const safe = document.createElement('div');
  safe.textContent = textValue;
  messages.appendChild(createRow(who, safe.innerHTML, extraClass));
  scrollToBottom();
}

function showTyping() {
  hideWelcome();
  typingEl = createRow('support', '<span class="dot"></span><span class="dot"></span><span class="dot"></span>', 'typing');
  messages.appendChild(typingEl);
  scrollToBottom();
}

function hideTyping() {
  if (typingEl) {
    typingEl.remove();
    typingEl = null;
  }
}

document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    decoding = btn.dataset.mode;
  });
});

document.querySelectorAll('.suggestion').forEach(btn => {
  btn.addEventListener('click', () => {
    text.value = btn.dataset.text;
    resizeInput();
    text.focus();
  });
});

text.addEventListener('input', resizeInput);
text.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    document.getElementById('f').requestSubmit();
  }
});

async function checkHealth() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    const online = d.model_loaded;
    statusEl.innerHTML =
      '<span class="status-dot" style="background:' + (online ? 'var(--accent-2)' : 'var(--danger)') + ';box-shadow:0 0 0 4px ' +
      (online ? 'rgba(94,234,212,0.12)' : 'rgba(255,123,138,0.12)') + '"></span>' +
      '<span>' + (online ? 'Model online' : 'Model unavailable') + '</span>';
  } catch (_) {
    statusEl.innerHTML = '<span class="status-dot" style="background:var(--danger)"></span><span>Offline</span>';
  }
}

document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const msg = text.value.trim();
  if (!msg || busy) return;

  addMessage(msg, 'customer');
  text.value = '';
  resizeInput();
  setBusy(true);
  showTyping();

  try {
    const r = await fetch('/reply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, decoding })
    });
    const d = await r.json();
    hideTyping();
    if (d.reply) {
      addMessage(d.reply, 'support');
    } else {
      addMessage(d.error || 'Something went wrong.', 'support', 'error');
    }
  } catch (err) {
    hideTyping();
    addMessage('Network error: ' + err, 'support', 'error');
  } finally {
    setBusy(false);
    text.focus();
  }
});

checkHealth();
resizeInput();
</script>
</body>
</html>"""


@app.get("/")
def index():
    return INDEX_HTML


@app.get("/health")
def health():
    return jsonify({
        "status": "ok" if GENERATOR else "model_not_trained",
        "model_loaded": GENERATOR is not None,
        "detail": LOAD_ERROR,
    }), (200 if GENERATOR else 503)


@app.post("/reply")
def reply():
    if GENERATOR is None:
        return jsonify({"error": LOAD_ERROR or "model not trained"}), 503

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "field 'message' is required"}), 400

    decoding = data.get("decoding", config.DEFAULT_DECODING)
    if decoding not in ("greedy", "sample"):
        return jsonify({"error": "decoding must be 'greedy' or 'sample'"}), 400

    answer = GENERATOR.reply(message, decoding=decoding)
    return jsonify({"reply": answer, "decoding": decoding})


def main():
    if LOAD_ERROR:
        print(f"[api] WARNING: {LOAD_ERROR}")
    print(f"[api] serving on http://{config.API_HOST}:{config.API_PORT}")
    app.run(host=config.API_HOST, port=config.API_PORT, debug=False)


if __name__ == "__main__":
    main()
