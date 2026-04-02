(function () {
  const form = document.getElementById("question-form");
  const questionInput = document.getElementById("question");
  const submitButton = document.getElementById("submit-button");
  const statusEl = document.getElementById("status");
  const resultsEl = document.getElementById("results");
  const resultsTitle = document.getElementById("results-title");
  const resultsSummary = document.getElementById("results-summary");
  const resultsPanelTitle = document.getElementById("results-panel-title");
  const resultsGrounding = document.getElementById("results-grounding");
  const panelNotesEl = document.getElementById("panel-notes");
  const synthesisCard = document.getElementById("synthesis-card");
  const synthesisText = document.getElementById("synthesis-text");
  const synthesisEvidence = document.getElementById("synthesis-evidence");
  const exampleButtons = Array.from(document.querySelectorAll("[data-question]"));

  let stageTimers = [];

  function clearStageTimers() {
    stageTimers.forEach((timer) => window.clearTimeout(timer));
    stageTimers = [];
  }

  function resetResults() {
    clearStageTimers();
    resultsEl.hidden = true;
    resultsTitle.textContent = "Your panel is in session.";
    resultsSummary.textContent = "";
    resultsPanelTitle.textContent = "";
    resultsGrounding.textContent = "";
    panelNotesEl.innerHTML = "";
    synthesisCard.classList.remove("is-revealed");
    synthesisText.textContent = "";
    synthesisEvidence.innerHTML = "";
  }

  function setLoadingState(isLoading, message) {
    submitButton.disabled = isLoading;
    questionInput.disabled = isLoading;
    statusEl.textContent = message || "";
  }

  function formatConfidence(value) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      return "Confidence unavailable";
    }
    return `Grounding confidence ${(value * 100).toFixed(0)}%`;
  }

  function createEvidenceTags(items) {
    const sources = Array.isArray(items) ? items.filter(Boolean) : [];
    if (!sources.length) {
      const span = document.createElement("span");
      span.className = "evidence-tag";
      span.textContent = "No explicit sources listed";
      return [span];
    }

    return sources.map((source) => {
      const span = document.createElement("span");
      span.className = "evidence-tag";
      span.textContent = source;
      return span;
    });
  }

  function createPanelCard(note, index) {
    const article = document.createElement("article");
    article.className = "panel-card";
    article.setAttribute("aria-label", `${note.agent_name} note ${index + 1}`);

    const status = document.createElement("p");
    status.className = "panel-card__status";
    status.textContent = `Expert note ${String(index + 1).padStart(2, "0")}`;

    const header = document.createElement("div");
    header.className = "panel-card__header";

    const identity = document.createElement("div");
    const name = document.createElement("p");
    name.className = "panel-card__name";
    name.textContent = note.agent_name || "Panel member";

    const role = document.createElement("p");
    role.className = "panel-card__role";
    role.textContent = note.role || "Research specialist";

    identity.append(name, role);

    const confidence = document.createElement("div");
    confidence.className = "panel-card__confidence";
    confidence.textContent = note.grounded ? formatConfidence(note.confidence) : "Ungrounded";

    header.append(identity, confidence);

    const content = document.createElement("p");
    content.className = "panel-card__content";
    content.textContent = note.content || "";

    const footer = document.createElement("div");
    footer.className = "panel-card__footer";
    createEvidenceTags(note.evidence_sources).forEach((tag) => footer.appendChild(tag));

    article.append(status, header, content, footer);
    return article;
  }

  function renderStageSequence(notes, synthesis, delayMs) {
    const cards = notes.map((note, index) => createPanelCard(note, index));
    panelNotesEl.innerHTML = "";
    cards.forEach((card) => panelNotesEl.appendChild(card));

    cards.forEach((card, index) => {
      const timer = window.setTimeout(() => {
        card.classList.add("is-revealed");
      }, delayMs * index);
      stageTimers.push(timer);
    });

    const synthTimer = window.setTimeout(() => {
      synthesisCard.classList.add("is-revealed");
    }, delayMs * cards.length + 180);
    stageTimers.push(synthTimer);
  }

  function renderResults(data) {
    const notes = Array.isArray(data.panel) ? data.panel : [];
    const synthesisSources = Array.isArray(data.synthesis_evidence_sources)
      ? data.synthesis_evidence_sources
      : [];

    resultsEl.hidden = false;
    resultsTitle.textContent = "Your panel is in session.";
    resultsSummary.textContent =
      data.question || "The panel is synthesizing your question with evidence-backed disagreement.";
    resultsPanelTitle.textContent = data.panel_title || "Research panel";
    resultsGrounding.textContent =
      typeof data.confidence === "number"
        ? formatConfidence(data.confidence)
        : data.grounded
          ? "Grounded synthesis"
          : "Synthesis still being checked";
    synthesisText.textContent = data.synthesis || "";

    synthesisEvidence.innerHTML = "";
    createEvidenceTags(synthesisSources).forEach((tag) => synthesisEvidence.appendChild(tag));

    renderStageSequence(notes, data.synthesis, 360);
  }

  async function submitQuestion(event) {
    event.preventDefault();

    const question = questionInput.value.trim();
    if (!question) {
      questionInput.focus();
      return;
    }

    resetResults();
    setLoadingState(true, "Running a private panel and staging the debate.");

    try {
      const response = await fetch("/debate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question }),
      });

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(payload.detail || `Request failed with status ${response.status}`);
      }

      renderResults(payload);
      statusEl.textContent = "Structured response received. Revealing panel notes.";
      resultsEl.scrollIntoView({ behavior: "smooth", block: "start" });

      const completionTimer = window.setTimeout(() => {
        statusEl.textContent = "Debate complete.";
      }, 360 * (Array.isArray(payload.panel) ? payload.panel.length : 0) + 900);
      stageTimers.push(completionTimer);
    } catch (error) {
      resultsEl.hidden = true;
      statusEl.textContent = `Unable to start the debate: ${error.message}`;
    } finally {
      submitButton.disabled = false;
      questionInput.disabled = false;
    }
  }

  exampleButtons.forEach((button) => {
    button.addEventListener("click", () => {
      questionInput.value = button.dataset.question || "";
      questionInput.focus();
      questionInput.setSelectionRange(questionInput.value.length, questionInput.value.length);
    });
  });

  form.addEventListener("submit", submitQuestion);
})();
