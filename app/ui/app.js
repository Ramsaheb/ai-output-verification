const queryInput = document.getElementById("query");
const contextInput = document.getElementById("context");
const answerInput = document.getElementById("answer");

const verifyBtn = document.getElementById("verifyBtn");
const exampleSupportBtn = document.getElementById("exampleSupportBtn");
const exampleHallucinationBtn = document.getElementById("exampleHallucinationBtn");
const clearBtn = document.getElementById("clearBtn");

const resultsCard = document.getElementById("resultsCard");
const placeholderCard = document.getElementById("placeholderCard");
const errorCard = document.getElementById("errorCard");
const metricsEl = document.getElementById("metrics");
const explanationEl = document.getElementById("explanation");
const errorMessageEl = document.getElementById("errorMessage");
const decisionPill = document.getElementById("decisionPill");
const apiStatusEl = document.getElementById("apiStatus");
const modeStatusEl = document.getElementById("modeStatus");

const VERIFY_LABEL = "Verify Output";

function setError(message) {
  errorMessageEl.textContent = message;
  errorCard.hidden = false;
}

function clearError() {
  errorCard.hidden = true;
  errorMessageEl.textContent = "";
}

function setBusy(isBusy) {
  verifyBtn.disabled = isBusy;
  verifyBtn.textContent = isBusy ? "Verifying..." : VERIFY_LABEL;
}

function formatNumber(value, digits) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return "-";
  }
  return n.toFixed(digits);
}

function clearForm() {
  queryInput.value = "";
  contextInput.value = "";
  answerInput.value = "";
  metricsEl.innerHTML = "";
  explanationEl.textContent = "";
  decisionPill.textContent = "-";
  decisionPill.classList.remove("decision-allow", "decision-flag", "decision-refuse");
  resultsCard.hidden = true;
  placeholderCard.hidden = false;
  clearError();
}

function setDecisionPill(decision) {
  decisionPill.textContent = decision;
  decisionPill.classList.remove("decision-allow", "decision-flag", "decision-refuse");

  if (decision === "ALLOW") {
    decisionPill.classList.add("decision-allow");
  } else if (decision === "FLAG") {
    decisionPill.classList.add("decision-flag");
  } else if (decision === "REFUSE") {
    decisionPill.classList.add("decision-refuse");
  }
}

function parseContext(raw) {
  return raw
    .split("\n")
    .map((line) => line.replace(/^[-*\u2022]\s*/, "").trim())
    .filter((line) => line.length > 0);
}

function addMetric(label, value) {
  const div = document.createElement("div");
  div.className = "metric";
  div.innerHTML = `<div class="metric-label">${label}</div><div class="metric-value">${value}</div>`;
  metricsEl.appendChild(div);
}

async function verifyOutput() {
  clearError();
  resultsCard.hidden = true;
  placeholderCard.hidden = false;
  metricsEl.innerHTML = "";

  const query = queryInput.value.trim();
  const context = parseContext(contextInput.value);
  const generatedAnswer = answerInput.value.trim();

  if (!query || !generatedAnswer || context.length === 0) {
    setError("Please provide query, at least one context chunk, and generated answer.");
    return;
  }

  setBusy(true);

  try {
    const response = await fetch("/api/v1/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        context,
        generated_answer: generatedAnswer,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`Request failed (${response.status}): ${text}`);
    }

    const data = await response.json();
    setDecisionPill(data.decision);

    addMetric("Decision", data.decision);
    addMetric("Confidence", formatNumber(data.confidence_score, 4));
    addMetric("Hallucination", data.hallucination_detected ? "Yes" : "No");
    addMetric("Coverage", data.verification_details.context_coverage);
    addMetric("Coverage %", formatNumber(data.verification_details.coverage_percent, 1));
    addMetric("Entailment Label", data.verification_details.entailment.label);
    addMetric(
      "Strict Mode",
      data.verification_details.strict_mode_applied ? "On" : "Off"
    );
    modeStatusEl.textContent = data.verification_details.strict_mode_applied
      ? "Strict"
      : "Balanced";

    explanationEl.textContent = data.explanation;

    resultsCard.hidden = false;
    placeholderCard.hidden = true;
  } catch (error) {
    setError(error instanceof Error ? error.message : "Unknown error occurred");
  } finally {
    setBusy(false);
  }
}

function loadSupportedExample() {
  queryInput.value = "What is the capital of France?";
  contextInput.value = "Paris is the capital city of France.\nFrance is in Western Europe.";
  answerInput.value = "The capital of France is Paris.";
}

function loadHallucinationExample() {
  queryInput.value = "Who developed the theory of relativity?";
  contextInput.value = "Albert Einstein developed the theory of relativity.";
  answerInput.value = "Isaac Newton developed the theory of relativity in 1687.";
}

async function checkApiHealth() {
  try {
    const response = await fetch("/api/v1/health");
    if (!response.ok) {
      throw new Error("health check failed");
    }
    const data = await response.json();
    apiStatusEl.textContent = data.status === "healthy" ? "Healthy" : data.status;
  } catch {
    apiStatusEl.textContent = "Unavailable";
    modeStatusEl.textContent = "Unknown";
    setError("API is currently unreachable. Please check if the server is running.");
  }
}

queryInput.addEventListener("input", clearError);
contextInput.addEventListener("input", clearError);
answerInput.addEventListener("input", clearError);

verifyBtn.addEventListener("click", verifyOutput);
exampleSupportBtn.addEventListener("click", loadSupportedExample);
exampleHallucinationBtn.addEventListener("click", loadHallucinationExample);
clearBtn.addEventListener("click", clearForm);

checkApiHealth();
