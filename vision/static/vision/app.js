async function fetchSimilarDetections(detectionId) {
  const response = await fetch("/api/search/similar/", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
      "X-CSRFToken": getCsrfToken(),
    },
    body: new URLSearchParams({ detection_id: detectionId, limit: "6" }),
  });

  if (!response.ok) {
    throw new Error("Similarity search failed.");
  }

  return response.json();
}

async function fetchRunStatus(runId) {
  const response = await fetch(`/api/runs/${runId}/`, {
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    throw new Error("Run status lookup failed.");
  }

  return response.json();
}

function getCsrfToken() {
  return document.cookie
    .split(";")
    .map((item) => item.trim())
    .find((item) => item.startsWith("csrftoken="))
    ?.split("=")[1];
}

function renderSimilarResults(results) {
  const target = document.getElementById("similar-results");
  if (!target) return;

  if (!results.length) {
    target.innerHTML = `
      <div class="empty-state">
        <strong>No similar plant patches found</strong>
        <span>Try a different finding or run scouting on more field imagery.</span>
      </div>
    `;
    return;
  }

  target.innerHTML = results
    .map(
      (item) => `
        <article class="similar-card">
          ${item.crop_url ? `<img src="${item.crop_url}" alt="${item.label}" class="crop-image" />` : ""}
          <strong>${humanizeToken(item.label)}</strong>
          <span class="muted">Confidence ${Number(item.confidence).toFixed(2)}</span>
          <span class="muted">Image #${item.image_id}</span>
        </article>
      `
    )
    .join("");
}

function humanizeToken(value) {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function selectDetectionById(detectionId, options = {}) {
  const targetId = `detail-${detectionId}`;
  const target = document.getElementById(targetId);
  if (!target) return;

  document.querySelectorAll(".inspector-card").forEach((card) => {
    card.hidden = true;
    card.classList.remove("inspector-card-active");
  });
  target.hidden = false;
  target.classList.add("inspector-card-active");

  document.querySelectorAll(".detection-row-active").forEach((row) => {
    row.classList.remove("detection-row-active");
  });
  const row = document.getElementById(`row-${detectionId}`);
  if (row) {
    row.classList.add("detection-row-active");
  }

  document.querySelectorAll(".bbox-active").forEach((bbox) => {
    bbox.classList.remove("bbox-active");
  });
  const bbox = document.querySelector(`.bbox[data-detection-id="${detectionId}"]`);
  if (bbox) {
    bbox.classList.add("bbox-active");
  }

  if (options.scroll !== false) {
    target.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

document.querySelectorAll(".bbox[data-target]").forEach((button) => {
  button.addEventListener("click", () => {
    selectDetectionById(button.dataset.detectionId);
  });
});

document.querySelectorAll(".detection-row[data-target]").forEach((button) => {
  button.addEventListener("click", () => {
    selectDetectionById(button.dataset.detectionId, { scroll: false });
  });
});

document.querySelectorAll("[data-file-input]").forEach((input) => {
  input.addEventListener("change", () => {
    const label = input.closest(".file-picker")?.querySelector("[data-file-name]");
    if (!label) return;
    label.textContent = input.files && input.files.length > 0 ? input.files[0].name : "No file selected";
  });
});

document.querySelectorAll(".similar-trigger").forEach((button) => {
  button.addEventListener("click", async () => {
    button.disabled = true;
    const previous = button.textContent;
    button.textContent = "Searching...";
    try {
      const payload = await fetchSimilarDetections(button.dataset.detectionId);
      renderSimilarResults(payload.results || []);
      selectDetectionById(button.dataset.detectionId, { scroll: false });
    } catch (error) {
      renderSimilarResults([]);
    } finally {
      button.disabled = false;
      button.textContent = previous;
    }
  });
});

function initializeRunMonitor() {
  const monitor = document.querySelector("[data-run-monitor]");
  if (!monitor) return;

  const runId = monitor.dataset.runId;
  const summary = document.querySelector("[data-run-summary]");
  const projectStatus = document.querySelector("[data-project-status]");
  const spinner = document.querySelector("[data-run-spinner]");
  if (!runId || !summary) return;

  let lastStatus = monitor.dataset.runTerminalStatus || "";

  const updateSummary = (run) => {
    summary.textContent = `${humanizeToken(run.current_stage)} · ${run.progress}% · ${String(run.status).toUpperCase()}`;
    if (projectStatus) {
      projectStatus.textContent = String(run.status === "succeeded" ? "ready" : run.status).toUpperCase();
    }
    if (spinner) {
      spinner.hidden = run.status === "succeeded" || run.status === "failed";
    }
  };

  const poll = async () => {
    try {
      const run = await fetchRunStatus(runId);
      updateSummary(run);

      if (run.status === "succeeded" || run.status === "failed") {
        window.location.reload();
        return;
      }

      lastStatus = run.status;
    } catch (_error) {
      if (lastStatus === "failed") return;
    }

    window.setTimeout(poll, 3000);
  };

  window.setTimeout(poll, 3000);
}

initializeRunMonitor();
