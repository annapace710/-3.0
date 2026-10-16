// Параметры валидации загружаемого файла.
// Список расширений синхронизирован с ALLOWED_EXTENSIONS в main.py.
// HEVC (H.265) и H.264 — кодеки: они живут внутри контейнеров .mp4/.mkv/.mov,
// поэтому проверяем и расширение, и MIME-тип (включая MIME-типы кодеков).
const VALID_FORMATS = [
  "video/mp4",        // H.264 / HEVC (H.265) в контейнере MP4
  "video/webm",
  "video/x-m4v",
  "video/quicktime",  // .mov (H.264 / HEVC)
  "video/x-msvideo",  // .avi
  "video/x-matroska", // .mkv (H.264 / HEVC)
  "video/hevc",       // HEVC (H.265)
  "video/h265",       // HEVC (H.265)
  "video/h264",       // H.264
];
const VALID_EXTENSIONS = [".mp4", ".webm", ".m4v", ".avi", ".mov", ".mkv", ".hevc", ".h265", ".h264"];
const MAX_FILE_SIZE_MB = 50;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

// Ключ для хранения адреса API в localStorage
const API_URL_STORAGE_KEY = "gymnastics_api_url";
const DEFAULT_API_URL = "http://127.0.0.1:8000";

const form = document.getElementById("analysis-form");
const apiUrlInput = document.getElementById("api-url");
const fileInput = document.getElementById("video-file");
const validationMessage = document.getElementById("validation-message");
const analyzeBtn = document.getElementById("analyze-btn");
const resultSection = document.getElementById("result");
const resultContent = document.getElementById("result-content");
const statusText = document.getElementById("status-text");
const saveReportBtn = document.getElementById("save-report-btn");

// ID последнего отчёта (для PDF-карточки и ассистента)
let lastReportId = null;

// Восстанавливаем сохранённый адрес API при загрузке страницы
const savedApiUrl = localStorage.getItem(API_URL_STORAGE_KEY);
if (savedApiUrl) {
  apiUrlInput.value = savedApiUrl;
}

/** Возвращает актуальный базовый URL API из поля ввода. */
function getApiBaseUrl() {
  const url = apiUrlInput.value.trim().replace(/\/+$/, "");
  return url || DEFAULT_API_URL;
}

/**
 * Проверяет загруженный файл: наличие, формат и размер.
 * Возвращает сообщение об ошибке или пустую строку при успехе.
 */
function validateFile(file) {
  if (!file) {
    return "Пожалуйста, загрузите видеофайл.";
  }

  const fileName = (file.name || "").toLowerCase();
  const hasValidMime = VALID_FORMATS.includes(file.type);
  const hasValidExtension = VALID_EXTENSIONS.some((ext) => fileName.endsWith(ext));

  if (!hasValidMime && !hasValidExtension) {
    return "Недопустимый формат файла. Поддерживаются: MP4, WebM, M4V, AVI, MOV, MKV, HEVC (H.265), H.264.";
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return `Файл слишком большой. Максимальный размер — ${MAX_FILE_SIZE_MB} МБ.`;
  }

  return "";
}

/** Очищает вывод результатов и сообщение валидации. */
function resetOutput() {
  resultSection.classList.add("hidden");
  resultContent.innerHTML = "";
  statusText.textContent = "";
  saveReportBtn.classList.add("hidden");
  hideAgentChat();
  lastReportId = null;
}

/** Форматирует долю кадров с обнаруженной позой в проценты. */
function formatPercent(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

/** Создаёт HTML-карточку метрики. */
function metricCard(label, value) {
  return `
    <div class="metric">
      <span class="metric-label">${label}</span>
      <span class="metric-value">${value}</span>
    </div>`;
}

/** Создаёт HTML-блок найденных ошибок техники. */
function renderErrors(errors) {
  if (!errors || errors.length === 0) {
    return `<div class="feedback-success">✅ Ошибок не обнаружено. Отличная техника!</div>`;
  }

  const items = errors.map((err) => {
    const severityLabel = err.severity === "critical" ? "🔴 Критично" : "🟡 Замечание";
    return `
      <li class="error-item error-${err.severity}">
        <span class="error-severity">${severityLabel}</span>
        <span class="error-text">${err.message}</span>
      </li>`;
  }).join("");

  return `<ul class="error-list">${items}</ul>`;
}

/** Создаёт HTML-блок рекомендаций по технике. */
function renderRecommendations(recommendations) {
  if (!recommendations || recommendations.length === 0) {
    return "";
  }

  const items = recommendations.map((rec) => `<li>${rec}</li>`).join("");
  return `
    <div class="feedback-block">
      <h3>💡 Рекомендации</h3>
      <ul class="recommendation-list">${items}</ul>
    </div>`;
}

/** Формирует HTML с результатами анализа. */
function renderAnalysis(data) {
  const analysis = data.analysis || {};
  const feedback = analysis.feedback || {};
  const movement = analysis.movement || {};
  const athletes = analysis.athletes || [];
  const clubs = analysis.clubs || {};
  const format = analysis.format || "individual";
  const torso = movement.torso_tilt || {};
  const wrist = movement.wrist_height || {};
  const trajectory = movement.trajectory || {};
  const motion = movement.motion || {};
  const knee = movement.knee_angle || {};
  const legOpening = movement.leg_opening || {};
  const hipHeight = movement.hip_height || {};
  const torsoSigned = movement.torso_tilt_signed || {};
  const ankleHeight = movement.ankle_height || {};
  const ankleY = movement.ankle_y || {};
  const armSymmetry = movement.arm_symmetry || {};
  const rotationTurns = movement.rotation_turns || {};

  // Базовые метрики
  const baseCards = [
    metricCard("Всего кадров", analysis.total_frames ?? "—"),
    metricCard("Частота кадров (FPS)", analysis.fps ?? "—"),
    metricCard("Присутствие в кадре", formatPercent(analysis.pose_presence)),
    metricCard("Кадров с человеком", analysis.pose_detections ?? "—"),
  ].join("");

  // Углублённые метрики движения
  let movementHtml = "";

  if (torso.avg_deg !== undefined) {
    movementHtml += `<h3>🤸 Угол наклона корпуса (кувырок)</h3>
      <div class="metrics-grid">
        ${metricCard("Средний", `${torso.avg_deg}°`)}
        ${metricCard("Максимальный", `${torso.max_deg}°`)}
        ${metricCard("Минимальный", `${torso.min_deg}°`)}
      </div>`;
  }

  if (wrist.avg_y !== undefined) {
    movementHtml += `<h3>🙌 Высота рук (подброс булав)</h3>
      <div class="metrics-grid">
        ${metricCard("Средняя высота", wrist.avg_y)}
        ${metricCard("Макс. высота", wrist.highest_y)}
        ${metricCard("Досягаемость", formatPercent(wrist.reach_ratio))}
        ${metricCard("Нижняя точка (Y)", wrist.lowest_y !== undefined ? wrist.lowest_y : "—")}
      </div>`;
  }

  // Булавы: параметры полёта предметов (трекинг)
  if (clubs.clubs_detected !== undefined) {
    const throwH = clubs.throw_height || {};
    const sym = clubs.symmetry || {};
    const simult = clubs.simultaneity || {};
    const catchPos = clubs.catch_position || {};

    let clubsHtml = `<h3>🪄 Булавы (трекинг предметов)</h3>
      <div class="metrics-grid">
        ${metricCard("Обнаружено булав", clubs.clubs_detected)}
      </div>`;

    if (throwH.max !== undefined) {
      clubsHtml += `<div class="metrics-grid">
        ${metricCard("Высота броска (левая)", throwH.left !== undefined ? throwH.left : "—")}
        ${metricCard("Высота броска (правая)", throwH.right !== undefined ? throwH.right : "—")}
      </div>`;
    }

    if (sym.height_diff !== undefined || simult.peak_frame_diff !== undefined) {
      clubsHtml += `<div class="metrics-grid">
        ${metricCard("Симметрия (разница высот)", sym.height_diff !== undefined ? sym.height_diff : "—")}
        ${metricCard("Одновременность (кадры)", simult.peak_frame_diff !== undefined ? simult.peak_frame_diff : "—")}
      </div>`;
    }

    if (catchPos.left || catchPos.right) {
      const leftPos = catchPos.left ? `(${catchPos.left.x}, ${catchPos.left.y})` : "—";
      const rightPos = catchPos.right ? `(${catchPos.right.x}, ${catchPos.right.y})` : "—";
      clubsHtml += `<div class="metrics-grid">
        ${metricCard("Ловля (левая)", leftPos)}
        ${metricCard("Ловля (правая)", rightPos)}
      </div>`;
    }

    movementHtml += clubsHtml;
  }

  if (trajectory.total_distance !== undefined) {
    movementHtml += `<h3>📈 Траектория движения</h3>
      <div class="metrics-grid">
        ${metricCard("Общий путь", trajectory.total_distance)}
        ${metricCard("Размах X", trajectory.x_range)}
        ${metricCard("Размах Y", trajectory.y_range)}
      </div>`;
  }

  if (motion.avg_speed !== undefined) {
    movementHtml += `<h3>⚡ Скорость и плавность</h3>
      <div class="metrics-grid">
        ${metricCard("Средняя скорость", motion.avg_speed)}
        ${metricCard("Макс. скорость", motion.max_speed)}
        ${metricCard("Плавность", motion.smoothness)}
        ${metricCard("Рывков", motion.jerk_count)}
      </div>`;
  }

  // Признаки, специфичные для упражнений
  if (knee.left_avg_deg !== undefined || knee.right_avg_deg !== undefined) {
    const leftVal = knee.left_avg_deg !== undefined ? `${knee.left_avg_deg}°` : "—";
    const rightVal = knee.right_avg_deg !== undefined ? `${knee.right_avg_deg}°` : "—";
    movementHtml += `<h3>🦵 Угол в колене (равновесие)</h3>
      <div class="metrics-grid">
        ${metricCard("Левая нога (ср.)", leftVal)}
        ${metricCard("Правая нога (ср.)", rightVal)}
        ${metricCard("Левая (мин.)", knee.left_min_deg !== undefined ? `${knee.left_min_deg}°` : "—")}
        ${metricCard("Правая (мин.)", knee.right_min_deg !== undefined ? `${knee.right_min_deg}°` : "—")}
      </div>`;
  }

  if (legOpening.max_deg !== undefined) {
    movementHtml += `<h3>🦵 Раскрытие ног (прыжок)</h3>
      <div class="metrics-grid">
        ${metricCard("Макс. угол", `${legOpening.max_deg}°`)}
        ${metricCard("Средний угол", `${legOpening.avg_deg}°`)}
      </div>`;
  }

  if (hipHeight.min_y !== undefined) {
    movementHtml += `<h3>📐 Высота таза</h3>
      <div class="metrics-grid">
        ${metricCard("Верхняя точка (Y)", hipHeight.min_y)}
        ${metricCard("Нижняя точка (Y)", hipHeight.max_y !== undefined ? hipHeight.max_y : "—")}
        ${metricCard("Средняя (Y)", hipHeight.avg_y !== undefined ? hipHeight.avg_y : "—")}
      </div>`;
  }

  if (torsoSigned.avg_deg !== undefined) {
    movementHtml += `<h3>🎯 Наклон оси (пируэт)</h3>
      <div class="metrics-grid">
        ${metricCard("Средний наклон", `${torsoSigned.avg_deg}°`)}
        ${metricCard("Макс. отклонение", `${torsoSigned.abs_max_deg}°`)}
      </div>`;
  }

  if (ankleHeight.min_y !== undefined) {
    movementHtml += `<h3>🦶 Высота стоп (прыжок)</h3>
      <div class="metrics-grid">
        ${metricCard("Верхняя точка (Y)", ankleHeight.min_y)}
        ${metricCard("Средняя (Y)", ankleHeight.avg_y !== undefined ? ankleHeight.avg_y : "—")}
      </div>`;
  }

  if (ankleY.avg_diff !== undefined) {
    movementHtml += `<h3>🦵 Опорная нога (равновесие)</h3>
      <div class="metrics-grid">
        ${metricCard("Левая лодыжка (ср.)", ankleY.left_avg_y !== undefined ? ankleY.left_avg_y : "—")}
        ${metricCard("Правая лодыжка (ср.)", ankleY.right_avg_y !== undefined ? ankleY.right_avg_y : "—")}
        ${metricCard("Разница высот", ankleY.avg_diff)}
      </div>`;
  }

  if (rotationTurns.turns !== undefined) {
    movementHtml += `<h3>🔄 Витки (пируэт)</h3>
      <div class="metrics-grid">
        ${metricCard("Число витков", rotationTurns.turns)}
        ${metricCard("Суммарный угол", `${rotationTurns.total_deg}°`)}
      </div>`;
  }

  if (armSymmetry.avg_diff !== undefined) {
    movementHtml += `<h3>💪 Симметрия рук (пируэт)</h3>
      <div class="metrics-grid">
        ${metricCard("Средняя разница", armSymmetry.avg_diff)}
        ${metricCard("Макс. разница", armSymmetry.max_diff)}
      </div>`;
  }

  if (!movementHtml) {
    movementHtml = `<p class="error-message">⚠️ Недостаточно данных о позе для углублённого анализа.</p>`;
  }

  // Блок «Ошибки и рекомендации» (ядро продукта)
  let feedbackHtml = "";
  if (feedback.summary || feedback.errors || feedback.recommendations) {
    const scoreHtml =
      feedback.overall_score != null
        ? `<div class="score-value">${feedback.overall_score}</div>`
        : `<div class="score-value score-na">—</div>`;
    feedbackHtml = `
      <div class="feedback-block">
        <h3>📋 Оценка техники</h3>
        <div class="score-row">
          <div class="score-box">${scoreHtml}<div class="score-label">Баллы</div></div>
          <div class="score-summary">${feedback.summary || ""}</div>
        </div>
        ${renderErrors(feedback.errors)}
      </div>
      ${renderRecommendations(feedback.recommendations)}`;
  }

  // Групповой формат: карточки по каждой гимнастке
  let athletesHtml = "";
  if (format === "group" && athletes.length > 0) {
    const cards = athletes.map((athlete, i) => {
      const fb = athlete.feedback || {};
      const score = fb.overall_score != null ? fb.overall_score : "—";
      const errCount = (fb.errors || []).length;
      const errText = errCount
        ? `<span class="error-text">Ошибок: ${errCount}</span>`
        : `<span class="feedback-success">✅ Без ошибок</span>`;
      return `
        <div class="metric athlete-card">
          <span class="metric-label">Гимнастка ${i + 1}</span>
          <span class="metric-value">${score}</span>
          ${errText}
        </div>`;
    }).join("");
    athletesHtml = `
      <div class="feedback-block">
        <h3>👥 Результаты по гимнасткам</h3>
        <div class="metrics-grid">${cards}</div>
      </div>`;
  }

  // Командные замечания (групповой формат)
  let teamHtml = "";
  const teamFeedback = analysis.team_feedback || {};
  const teamMessages = teamFeedback.messages || [];
  if (format === "group" && teamMessages.length > 0) {
    const teamItems = teamMessages.map((msg) => {
      const severityLabel = msg.severity === "critical" ? "🔴" : "🟡";
      return `<li class="error-item error-${msg.severity}">
        <span class="error-severity">${severityLabel}</span>
        <span class="error-text">${msg.message}</span>
      </li>`;
    }).join("");
    teamHtml = `
      <div class="feedback-block">
        <h3>🤝 Командные замечания</h3>
        <ul class="error-list">${teamItems}</ul>
      </div>`;
  }

  resultContent.innerHTML = `
    <div class="metrics-grid">${baseCards}</div>
    <div class="movement-block">${movementHtml}</div>
    ${athletesHtml}
    ${feedbackHtml}
    ${teamHtml}
  `;
}

/** Показывает сообщение об ошибке в блоке результата. */
function renderError(message) {
  resultContent.innerHTML = `<p class="error-message">⚠️ ${message}</p>`;
}

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  resetOutput();
  validationMessage.textContent = "";

  const file = fileInput.files[0];
  const error = validateFile(file);

  if (error) {
    validationMessage.textContent = error;
    return;
  }

  const exercise = document.getElementById("exercise-select").value;
  const format = document.getElementById("format-select").value;
  const apiBaseUrl = getApiBaseUrl();

  // Сохраняем адрес API для следующего визита
  localStorage.setItem(API_URL_STORAGE_KEY, apiBaseUrl);

  // Готовим FormData с видео и выбранным упражнением
  const formData = new FormData();
  formData.append("video_file", file);
  formData.append("exercise", exercise);
  formData.append("format", format);

  // Блокируем кнопку на время запроса
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Анализ…";
  resultSection.classList.remove("hidden");
  statusText.textContent = "⏳ Загружаем и анализируем видео…";

  try {
    const response = await fetch(`${apiBaseUrl}/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || data.error || `Ошибка сервера (${response.status})`);
    }

    if (data.error) {
      renderError(data.error);
    } else {
      renderAnalysis(data);
      statusText.textContent = "✅ Анализ завершён.";
      // Отчёт сохранён на сервере — показываем PDF и ассистента
      lastReportId = data.report_id || null;
      if (lastReportId) {
        saveReportBtn.classList.remove("hidden");
        showAgentChat();
      }
    }
  } catch (err) {
    console.error("Ошибка запроса:", err);
    renderError("Не удалось подключиться к серверу анализа. Проверьте адрес API и что бэкенд запущен.");
    statusText.textContent = "❌ Ошибка.";
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Проанализировать";
  }
});

/** Скачивает PDF-карточку тренировки. */
saveReportBtn.addEventListener("click", async function () {
  if (!lastReportId) return;
  const apiBaseUrl = getApiBaseUrl();
  saveReportBtn.disabled = true;
  saveReportBtn.textContent = "⏳ Формируем PDF…";
  try {
    const response = await fetch(`${apiBaseUrl}/reports/${lastReportId}/pdf`);
    if (!response.ok) {
      throw new Error(`Ошибка сервера (${response.status})`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `training-report-${lastReportId.slice(0, 8)}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    saveReportBtn.textContent = "💾 Сохранить отчёт (PDF)";
  } catch (err) {
    console.error("Ошибка скачивания отчёта:", err);
    saveReportBtn.textContent = "❌ Не удалось сформировать PDF. Проверьте адрес API.";
  } finally {
    saveReportBtn.disabled = false;
  }
});

// === Ассистент тренера (LLM-агент) ===

const agentChatSection = document.getElementById("agent-chat");
const chatMessages = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSendBtn = document.getElementById("chat-send-btn");

// История диалога: [{role: "user"|"assistant", content: "..."}]
let chatHistory = [];

/** Добавляет сообщение в чат. */
function appendChatMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `chat-message chat-${role}`;
  bubble.textContent = text;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/** Показывает панель ассистента после успешного анализа; сбрасывает диалог. */
function showAgentChat() {
  chatHistory = [];
  chatMessages.innerHTML = "";
  if (agentChatSection) {
    agentChatSection.classList.remove("hidden");
  }
}

/** Скрывает панель ассистента (при сбросе результата). */
function hideAgentChat() {
  if (agentChatSection) {
    agentChatSection.classList.add("hidden");
  }
}

/** Блокирует/разблокирует отправку на время запроса. */
function setChatBusy(busy) {
  chatSendBtn.disabled = busy;
  chatInput.disabled = busy;
  chatSendBtn.textContent = busy ? "…" : "Отправить";
}

/** Отправляет сообщение агенту и показывает ответ. */
async function sendAgentMessage(text) {
  const message = (text || "").trim();
  if (!message || !lastReportId) return;

  appendChatMessage("user", message);
  chatHistory.push({ role: "user", content: message });

  setChatBusy(true);
  const thinking = document.createElement("div");
  thinking.className = "chat-message chat-assistant chat-thinking";
  thinking.textContent = "Ассистент думает…";
  chatMessages.appendChild(thinking);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const response = await fetch(
      `${getApiBaseUrl()}/agent/chat/${lastReportId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history: chatHistory.slice(0, -1) }),
      }
    );
    const data = await response.json();
    thinking.remove();

    if (!response.ok) {
      const explanations = {
        429: "Слишком много сообщений подряд — подождите минуту.",
        503: "LLM-ассистент не настроен: задайте LLM_API_KEY на сервере.",
        502: "Сервис LLM недоступен. Попробуйте позже.",
      };
      appendChatMessage(
        "assistant",
        explanations[response.status] || data.detail || `Ошибка сервера (${response.status})`
      );
      return;
    }

    const reply = data.reply || "Пустой ответ ассистента.";
    appendChatMessage("assistant", reply);
    chatHistory.push({ role: "assistant", content: reply });
  } catch (err) {
    console.error("Ошибка запроса к агенту:", err);
    thinking.remove();
    appendChatMessage(
      "assistant",
      "Не удалось связаться с ассистентом. Проверьте адрес API."
    );
  } finally {
    setChatBusy(false);
  }
}

chatForm.addEventListener("submit", function (event) {
  event.preventDefault();
  const text = chatInput.value;
  chatInput.value = "";
  sendAgentMessage(text);
});

// Кнопки-подсказки
document.querySelectorAll(".chat-suggest").forEach((btn) => {
  btn.addEventListener("click", function () {
    sendAgentMessage(this.dataset.q);
  });
});
