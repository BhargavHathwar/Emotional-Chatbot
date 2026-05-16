const EMOTION_CONFIG = {
  happy:   { emoji: "😊", color: "#f59e0b", label: "Happy"   },
  sad:     { emoji: "💙", color: "#3b82f6", label: "Sad"     },
  stress:  { emoji: "😟", color: "#8b5cf6", label: "Stressed"},
  angry:   { emoji: "😠", color: "#ef4444", label: "Angry"   },
  neutral: { emoji: "🙂", color: "#6b7280", label: "Neutral" },
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
        borderColor: "#a78bfa",
        backgroundColor: "rgba(167,139,250,0.08)",
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
        backgroundColor: "rgba(15,15,25,0.85)",
        titleColor: "#e2e8f0",
        bodyColor: "#94a3b8",
        padding: 10,
        borderColor: "rgba(255,255,255,0.08)",
        borderWidth: 1,
      }},
      scales: {
        x: { display: false },
        y: {
          min: 0.5, max: 5.5,
          ticks: {
            stepSize: 1,
            callback: v => ({ 5:"😊", 4:"🙂", 3:"😟", 2:"💙", 1:"😠" }[v] || ""),
            color: "rgba(255,255,255,0.35)",
            font: { size: 14 },
          },
          grid: { color: "rgba(255,255,255,0.05)" },
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
  const EMOTION_Y = { happy: 5, neutral: 4, stress: 3, sad: 2, angry: 1 };
  chart.data.labels.push(String(emotionHistory.length));
  ds.data.push(EMOTION_Y[emotion] || 3);
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
    appendMessage(`<span>⚠️ Connection error. Please try again.</span>`, "bot error-msg");
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
  document.getElementById("message").addEventListener("keypress", e => {
    if (e.key === "Enter") sendMessage();
  });
});