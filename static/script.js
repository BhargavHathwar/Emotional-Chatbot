// ── Startup health check ──────────────────────────────────────────────────────
// On page load, ping /health. If model isn't ready, show a banner.
// If server is unreachable (cold start), keep retrying and inform the user.
function checkServerHealth() {
  const banner = document.getElementById('loading-banner');
  if (!banner) return;
  fetch('/health', { method: 'GET' })
    .then(res => res.json())
    .then(data => {
      if (!data.model_loaded) {
        banner.style.display = 'flex';
        banner.textContent = '⏳ AI model is warming up — please wait about 30 seconds...';
        setTimeout(checkServerHealth, 5000);
      } else {
        banner.style.display = 'none';
      }
    })
    .catch(() => {
      if (!banner) return;
      banner.style.display = 'flex';
      banner.textContent = '🔄 Server is waking up (cold start) — this takes up to 60 seconds. Hang tight!';
      setTimeout(checkServerHealth, 8000);
    });
}

const EMOTION_CONFIG = {
  joy:      { emoji: "😊", color: "#e8944a", label: "Happy"    },
  love:     { emoji: "💖", color: "#e8607a", label: "Love"     },
  sadness:  { emoji: "💙", color: "#5b90cc", label: "Sad"      },
  worry:    { emoji: "😟", color: "#9b72cf", label: "Stressed" },
  anger:    { emoji: "😠", color: "#d95555", label: "Angry"    },
  surprise: { emoji: "😲", color: "#f0b429", label: "Surprised"},
  neutral:  { emoji: "🙂", color: "#8a9a88", label: "Neutral"  },
  unknown:  { emoji: "🤔", color: "#aaaaaa", label: "Unknown"  },
};

let messageCount   = 0;
let emotionHistory = [];
let emotionCounts  = {};
let chart;

function initChart() {
  const ctx = document.getElementById("emotionChart").getContext("2d");
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        data: [],
        borderColor: "#c4703a",
        backgroundColor: "rgba(196,112,58,0.07)",
        borderWidth: 2,
        pointRadius: 5,
        pointBackgroundColor: [],
        pointBorderColor: "#fff",
        pointBorderWidth: 1.5,
        tension: 0.4,
        fill: true,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: { legend: { display: false }, tooltip: {
        callbacks: {
          label: ctx => {
            const e = emotionHistory[ctx.dataIndex];
            return e ? `${EMOTION_CONFIG[e.emotion]?.label} (${e.confidence}%)` : "";
          },
          title: ctx => `Message #${parseInt(ctx[0].label)}`,
        },
        backgroundColor: "rgba(44,36,22,0.88)",
        titleColor: "#faf8f5",
        bodyColor: "#c8b8a0",
        padding: 10,
        borderColor: "rgba(44,36,22,0.1)",
        borderWidth: 1,
      }},
      scales: {
        x: { display: false },
        y: {
          min: 0.5, max: 7.5,
          ticks: {
            stepSize: 1,
            callback: v => ({ 7:"😊", 6:"💖", 5:"🙂", 4:"😟", 3:"💙", 2:"😲", 1:"😠" }[v] || ""),
            color: "rgba(44,36,22,0.35)",
            font: { size: 14 },
          },
          grid: { color: "rgba(44,36,22,0.06)" },
          border: { display: false },
        }
      }
    }
  });
}

function updateChart(emotion, confidence) {
  const cfg = EMOTION_CONFIG[emotion] || EMOTION_CONFIG.neutral;
  emotionHistory.push({ emotion, confidence });
  const ds = chart.data.datasets[0];
  const EMOTION_Y = { joy: 7, love: 6, neutral: 5, worry: 4, sadness: 3, surprise: 2, anger: 1, unknown: 4 };
  chart.data.labels.push(String(emotionHistory.length));
  ds.data.push(EMOTION_Y[emotion] || 4);
  ds.pointBackgroundColor.push(cfg.color);
  if (ds.data.length > 20) {
    chart.data.labels.shift();
    ds.data.shift();
    ds.pointBackgroundColor.shift();
  }
  chart.update();
  document.getElementById("chart-empty").style.display = "none";
}

function updateStats(emotion) {
  emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
  document.getElementById("stat-messages").textContent = messageCount;
  const dominant = Object.entries(emotionCounts).sort((a, b) => b[1] - a[1])[0];
  if (dominant) {
    const cfg = EMOTION_CONFIG[dominant[0]];
    document.getElementById("stat-dominant").textContent = (cfg?.emoji || "") + " " + (cfg?.label || dominant[0]);
  }
}

function safeText(text) {
  const el = document.createElement("span");
  el.textContent = text;
  return el.innerHTML;
}

function appendMessage(html, cls) {
  const chatbox = document.getElementById("chatbox");
  const div = document.createElement("div");
  div.className = "message " + cls;
  div.innerHTML = html;
  chatbox.appendChild(div);
  chatbox.scrollTop = chatbox.scrollHeight;
}

function sendMessage() {
  const input = document.getElementById("message");
  const btn   = document.getElementById("send-btn");
  const msg   = input.value.trim();
  if (!msg) return;

  messageCount++;
  input.value = "";
  btn.disabled = true;

  appendMessage(`<span>${safeText(msg)}</span>`, "user");

  const typingEl = document.getElementById("typing");
  typingEl.style.display = "flex";
  document.getElementById("chatbox").scrollTop = 99999;

  fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: msg })
  })
  .then(res => res.json())
  .then(data => {
    setTimeout(() => {
      typingEl.style.display = "none";
      btn.disabled = false;

      if (data.error) {
        appendMessage(`<span>⚠️ ${safeText(data.error)}</span>`, "bot error-msg");
        return;
      }

      const { emotion, confidence, reply, top2 } = data;
      const cfg = EMOTION_CONFIG[emotion] || EMOTION_CONFIG.neutral;

      const badgeHtml = `
        <div class="emotion-badge" style="--badge-color:${cfg.color}">
          <span class="badge-emoji">${cfg.emoji}</span>
          <span class="badge-label">${cfg.label}</span>
          <span class="badge-conf">${confidence}%</span>
        </div>`;

      let top2Html = "";
      if (top2 && top2.length > 1) {
        top2Html = `<div class="top2">`;
        top2.forEach(t => {
          const c = EMOTION_CONFIG[t.emotion] || {};
          top2Html += `<span class="top2-pill" style="--pill-color:${c.color || '#888'}">${c.emoji || ""} ${c.label || t.emotion} <b>${t.confidence}%</b></span>`;
        });
        top2Html += `</div>`;
      }

      appendMessage(`${badgeHtml}<span class="bot-text">${safeText(reply)}</span>${top2Html}`, "bot");
      updateChart(emotion, confidence);
      updateStats(emotion);
    }, 800);
  })
  .catch(() => {
    typingEl.style.display = "none";
    btn.disabled = false;
    appendMessage(`<span>⚠️ Server unreachable — it may be waking up. Wait 30 seconds and try again.</span>`, "bot error-msg");
    checkServerHealth();
  });
}

function clearChat() {
  document.getElementById("chatbox").innerHTML = `
    <div class="welcome-msg">
      <div class="welcome-icon">👋</div>
      <p>Hi! I'm your Emotion AI assistant. Tell me how you're feeling and I'll do my best to understand and support you.</p>
    </div>`;
  messageCount = 0;
  emotionHistory = [];
  emotionCounts  = {};
  document.getElementById("stat-messages").textContent = "0";
  document.getElementById("stat-dominant").textContent = "—";
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.data.datasets[0].pointBackgroundColor = [];
  chart.update();
  document.getElementById("chart-empty").style.display = "block";
}

document.addEventListener("DOMContentLoaded", () => {
  initChart();
  checkServerHealth();
  document.getElementById("message").addEventListener("keypress", e => {
    if (e.key === "Enter") sendMessage();
  });
});
