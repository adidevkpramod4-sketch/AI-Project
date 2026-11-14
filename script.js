
// The backend address comes from config.js (edit that file, not this one).
const API_BASE = (window.QA_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/, "");

const imageInput = document.getElementById("imageInput");
const dropArea = document.getElementById("dropArea");
const uploadPrompt = document.getElementById("uploadPrompt");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const selectedFileName = document.getElementById("selectedFileName");
const removeImage = document.getElementById("removeImage");
const searchButton = document.getElementById("searchButton");
const domainSelect = document.getElementById("domain");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const answerBox = document.getElementById("answerBox");
const answerText = document.getElementById("answerText");
const answerMeta = document.getElementById("answerMeta");

const backendStatus = document.getElementById("backendStatus");
const emptyState = document.getElementById("emptyState");
const resultsContent = document.getElementById("resultsContent");
const resultTitle = document.getElementById("resultTitle");
const resultBadge = document.getElementById("resultBadge");
const resultMessage = document.getElementById("resultMessage");
const resultsGrid = document.getElementById("resultsGrid");
const decisionBanner = document.getElementById("decisionBanner");
const decisionLabel = document.getElementById("decisionLabel");
const decisionReason = document.getElementById("decisionReason");
const evidenceRow = document.getElementById("evidenceRow");
const candidatesHeading = document.getElementById("candidatesHeading");
const evalContent = document.getElementById("evalContent");
const categoryFilter = document.getElementById("categoryFilter");
const categoryLine = document.getElementById("categoryLine");

let selectedFile = null;
let previewUrl = null;
let searchInProgress = false;

// Check whether the FastAPI backend is running.
async function checkBackend() {
    try {
        const response = await fetch(`${API_BASE}/health`);

        if (!response.ok) {
            throw new Error("Backend health check failed");
        }

        backendStatus.innerHTML =
            '<span class="dot"></span> Backend connected';

        backendStatus.classList.add("connected");
        backendStatus.classList.remove("disconnected");

    } catch (error) {
        backendStatus.innerHTML =
            '<span class="dot"></span> Backend offline';

        backendStatus.classList.add("disconnected");
        backendStatus.classList.remove("connected");

        console.warn("Could not connect to the backend:", error);
    }
}

// Display the chosen image in the preview panel.
function displayImage(file) {
    if (!file || !file.type.startsWith("image/")) {
        alert("Please select a valid image file.");
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert("Please choose an image smaller than 10 MB.");
        return;
    }

    selectedFile = file;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);
    imagePreview.src = previewUrl;
    selectedFileName.textContent = file.name;

    uploadPrompt.hidden = true;
    previewContainer.hidden = false;

    // Clear results belonging to the previous image.
    resultsContent.hidden = true;
    emptyState.hidden = false;
    resultsGrid.replaceChildren();
    answerBox.hidden = true;
    answerText.textContent = "";
    answerMeta.textContent = "";
    decisionBanner.hidden = true;

    const emptyHeading = emptyState.querySelector("h3");
    const emptyDescription = emptyState.querySelector("p");

    emptyHeading.textContent = "Your results will appear here";
    emptyDescription.textContent =
        "Upload an image to begin a visual search.";

    searchButton.disabled = searchInProgress;
    searchButton.innerHTML = "<span>⌕</span> Search visual index";
}

// Open the file picker when the upload area is clicked.
dropArea.addEventListener("click", (event) => {
    if (event.target.closest("button") ||
        event.target.closest("input")) {
        return;
    }

    imageInput.click();
});

// Handle file selection.
imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    if (file) {
        displayImage(file);
    }
});

// Drag-and-drop support.
dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropArea.classList.add("dragging");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragging");
});

dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    dropArea.classList.remove("dragging");

    const file = event.dataTransfer.files[0];

    if (file) {
        displayImage(file);
        imageInput.value = "";
    }
});

// Remove the currently selected image.
removeImage.addEventListener("click", (event) => {
    event.stopPropagation();

    selectedFile = null;
    imageInput.value = "";
    imagePreview.removeAttribute("src");
    selectedFileName.textContent = "";

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    previewContainer.hidden = true;
    uploadPrompt.hidden = false;
    decisionBanner.hidden = true;

    resultsContent.hidden = true;
    emptyState.hidden = false;
    resultsGrid.replaceChildren();
    answerBox.hidden = true;
    answerText.textContent = "";
    answerMeta.textContent = "";

    const emptyHeading = emptyState.querySelector("h3");
    const emptyDescription = emptyState.querySelector("p");

    emptyHeading.textContent = "Your results will appear here";
    emptyDescription.textContent =
        "Upload an image to begin a visual search.";

    searchButton.disabled = searchInProgress;
});

// Create a result card using actual backend data.
function addLine(parent, tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    el.textContent = text;
    parent.appendChild(el);
    return el;
}

function createResultCard(item, index, status) {
    const card = document.createElement("article");
    card.className = "result-card";
    if (index === 0) card.classList.add(status === "MATCH" ? "top-match" : "top-unconfirmed");

    // Visual evidence: the query photo next to the matched reference photo.
    const compare = document.createElement("div");
    compare.className = "compare";

    const queryFigure = document.createElement("figure");
    const queryImg = document.createElement("img");
    queryImg.src = previewUrl;
    queryImg.alt = "Your query image";
    queryFigure.append(queryImg);
    addLine(queryFigure, "figcaption", "", "Query");

    const refFigure = document.createElement("figure");
    const refImg = document.createElement("img");
    // The API returns paths such as /reference-images/fashion/shoe01__front.jpg
    refImg.src = `${API_BASE}${item.image_url}`;
    refImg.alt = `Reference image: ${item.filename}`;
    refImg.loading = "lazy";
    refImg.addEventListener("error", () => {
        refImg.alt = `Could not load ${item.filename}`;
        refImg.title = "Reference image could not be loaded";
    });
    refFigure.append(refImg);
    addLine(refFigure, "figcaption", "", "Best reference photo");

    compare.append(queryFigure, refFigure);

    const info = document.createElement("div");
    info.className = "result-info";
    addLine(info, "h4", "", `#${item.rank || index + 1} Instance ${item.instance_id}`);
    addLine(info, "p", "",
        `${item.domain} · ${item.category || "unlabelled"} · ${item.reference_photos} reference photo(s) · matched ${item.filename}`);

    // Cosine similarity is shown as a plain number, not a percentage,
    // because it is not a probability.
    addLine(info, "span", "demo-score",
        `Cosine similarity: ${item.similarity.toFixed(3)}`);
    const bar = document.createElement("div");
    bar.className = "score-bar";
    const fill = document.createElement("div");
    fill.style.width = `${Math.max(0, Math.min(1, item.similarity)) * 100}%`;
    bar.append(fill);
    info.append(bar);

    card.append(compare, info);
    return card;
}

function renderDecision(decision) {
    const isMatch = decision.status === "MATCH";
    const fmt = (v) => (v === null || v === undefined ? "n/a" : v.toFixed(3));

    decisionBanner.hidden = false;
    decisionBanner.className =
        `decision-banner ${isMatch ? "decision-match" : "decision-unknown"}`;
    decisionLabel.textContent = isMatch
        ? "POTENTIAL MATCH"
        : "UNKNOWN - no reliable match found";
    decisionReason.textContent = decision.reason;

    evidenceRow.replaceChildren();
    [
        ["Top score", fmt(decision.top_score)],
        ["Runner-up", fmt(decision.second_score)],
        ["Lead (gap)", fmt(decision.score_gap)],
        ["Min score", fmt(decision.min_score)],
        ["Min gap", fmt(decision.min_gap)],
        ["Thresholds", decision.calibrated ? "calibrated on validation" : "NOT calibrated (defaults)"]
    ].forEach(([label, value]) => {
        const chip = document.createElement("div");
        chip.className = "evidence-chip";
        addLine(chip, "span", "", label);
        addLine(chip, "strong", "", value);
        evidenceRow.append(chip);
    });
}

// Submit the image to the FastAPI search endpoint.
async function searchImages() {
    if (searchInProgress) {
        return;
    }

    if (!selectedFile) {
        alert("Please upload an image first.");
        return;
    }

    searchInProgress = true;
    searchButton.disabled = true;
    searchButton.textContent = "Searching...";
    decisionBanner.hidden = true;

    emptyState.hidden = false;
    resultsContent.hidden = true;
    resultsGrid.replaceChildren();

    const emptyHeading = emptyState.querySelector("h3");
    const emptyDescription = emptyState.querySelector("p");

    emptyHeading.textContent = "Searching the visual index...";
    emptyDescription.textContent =
        "The embedding model is comparing your image with the indexed reference images.";

    const formData = new FormData();

    // These field names must match the FastAPI /search endpoint.
    formData.append("file", selectedFile);
    formData.append("domain", domainSelect.value);
    formData.append("top_k", "12");
    formData.append("category_filter", categoryFilter.checked ? "true" : "false");

    try {
        const response = await fetch(`${API_BASE}/search`, {
            method: "POST",
            body: formData
        });

        // Read the response even if the server returns an error.
        const data = await response.json();

        if (!response.ok) {
            const detail = typeof data.detail === "string"
                ? data.detail
                : JSON.stringify(data.detail || data);

            throw new Error(detail || "The search request failed.");
        }

        if (!Array.isArray(data.results)) {
            throw new Error("The backend returned an unexpected response.");
        }

        emptyState.hidden = true;
        resultsContent.hidden = false;

        const decision = data.decision;
        const isMatch = decision.status === "MATCH";
        renderDecision(decision);
        const cat = data.category;
        categoryLine.textContent =
            `AI recognised: ${cat.label} (${(cat.confidence * 100).toFixed(0)}% confidence)` +
            (cat.filter_applied ? ` - showing only "${cat.label}" results.` : ".") +
            (cat.note ? ` ${cat.note}` : "");

        resultTitle.textContent = isMatch
            ? `Potential match: instance ${data.results[0].instance_id}`
            : "UNKNOWN - insufficient evidence";
        resultBadge.textContent = isMatch ? "MATCH ACCEPTED" : "UNKNOWN";
        candidatesHeading.textContent = isMatch
            ? "Ranked candidates"
            : "Closest candidates (NOT confirmed as a match)";

        const latency = data.latency_ms
            ? ` Latency: ${data.latency_ms.total} ms (embedding ${data.latency_ms.embedding} ms, search ${data.latency_ms.search} ms).`
            : "";
        resultMessage.textContent =
            `Searched ${data.indexed_images} reference photos of ` +
            `${data.indexed_instances} instances (domain: ${data.domain}). ` +
            `Query: ${data.query_filename}.${latency} ` +
            (data.hidden_unrelated
                ? `${data.hidden_unrelated} less similar item(s) hidden. `
                : "") +
            (data.message || "");

        if (data.results.length === 0) {
            resultMessage.textContent +=
                " No matching images were returned.";
        }

        data.results.forEach((item, index) => {
            resultsGrid.appendChild(createResultCard(item, index, decision.status));
        });

        // Remember this search for the History page (stored in this browser).
        if (window.QAHistory) {
            const queryFile = selectedFile;
            QAHistory.makeThumb(queryFile).then((thumb) => QAHistory.save({
                id: `${Date.now()}${Math.random().toString(36).slice(2, 6)}`,
                time: new Date().toISOString(),
                filename: queryFile.name,
                domain: data.domain,
                thumb,
                decision,
                category: cat,
                latency_ms: data.latency_ms,
                indexed_images: data.indexed_images,
                results: data.results.map((r) => ({
                    rank: r.rank, instance_id: r.instance_id, domain: r.domain, category: r.category,
                    filename: r.filename, image_url: r.image_url, similarity: r.similarity
                }))
            })).catch(() => {});
        }

    } catch (error) {
        console.error("QA search error:", error);

        emptyState.hidden = false;
        resultsContent.hidden = true;

        emptyHeading.textContent = "Search failed";
        emptyDescription.textContent =
            `${error.message} Check that your FastAPI server is running at ${API_BASE}.`;
    } finally {
        searchInProgress = false;
        searchButton.disabled = false;
        searchButton.innerHTML = "<span>⌕</span> Search visual index";
    }
}

// Connect the search button to the real API.
searchButton.addEventListener("click", searchImages);

// Ask BLIP a natural-language question about the currently selected image.
async function askAboutImage() {
    if (askButton.disabled) {
        return;
    }

    if (!selectedFile) {
        alert("Please upload an image first.");
        return;
    }

    const question = questionInput.value.trim();
    if (!question) {
        alert("Please type a question about the image.");
        questionInput.focus();
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("question", question);

    askButton.disabled = true;
    askButton.textContent = "Analyzing image...";
    answerBox.hidden = false;
    answerText.textContent = "BLIP is examining the image and generating an answer. The first request can take a little longer while the model loads.";
    answerMeta.textContent = "";

    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            body: formData
        });
        const data = await response.json();

        if (!response.ok) {
            const detail = typeof data.detail === "string"
                ? data.detail
                : JSON.stringify(data.detail || data);
            throw new Error(detail || "The question-answering request failed.");
        }

        answerText.textContent = data.answer || "No answer was returned.";
        answerMeta.textContent =
            `Model: ${data.model || "BLIP VQA"} · Device: ${data.device || "unknown"}. ` +
            (data.message || "");
    } catch (error) {
        console.error("QA VQA error:", error);
        answerText.textContent = `Could not answer: ${error.message}`;
        answerMeta.textContent = `Check that the FastAPI backend is running at ${API_BASE}.`;
    } finally {
        askButton.disabled = false;
        askButton.textContent = "Ask AI about image";
    }
}

askButton.addEventListener("click", askAboutImage);
questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        askAboutImage();
    }
});

// ---- Evaluation dashboard (reads backend/embeddings/evaluation_*.json) ----
function metricCell(value) {
    return value === null || value === undefined ? "n/a" : value.toFixed(3);
}

function makeTable(headers, rows) {
    const table = document.createElement("table");
    table.className = "eval-table";
    const head = table.createTHead().insertRow();
    headers.forEach((h) => addLine(head, "th", "", h));
    const body = table.createTBody();
    rows.forEach((cells) => {
        const tr = body.insertRow();
        cells.forEach((c) => addLine(tr, "td", "", c));
    });
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    wrap.append(table);
    return wrap;
}

function renderEvaluation(data) {
    evalContent.replaceChildren();
    const r = data.retrieval;
    const d = data.decision;
    const ks = [1, 3, 5, 10];

    addLine(evalContent, "p", "eval-meta",
        `Split: ${data.split} · model: ${data.model} · ` +
        `${data.index.photos} indexed photos / ${data.index.instances} instances · ` +
        `${d.n_queries} queries (${r.n_queries} known, ${d.n_unknown_queries} unknown-identity)`);

    addLine(evalContent, "h3", "", "Precision at different ranking depths");
    evalContent.append(makeTable(
        ["K", "P@K", "mP@K", "Recall@K", "Success@K"],
        ks.map((k) => [k, metricCell(r[`P@${k}`]), metricCell(r[`mP@${k}`]),
                       metricCell(r[`Recall@${k}`]), metricCell(r[`Success@${k}`])])
    ));

    addLine(evalContent, "h3", "", "UNKNOWN handling");
    evalContent.append(makeTable(
        ["Metric", "Value"],
        [["Unknown rejection rate", metricCell(d.unknown_rejection_rate)],
         ["False acceptance rate", metricCell(d.false_acceptance_rate)],
         ["Precision of accepted answers", metricCell(d.accept_precision)],
         ["False rejection rate", metricCell(d.false_rejection_rate)],
         ["Coverage (queries answered)", metricCell(d.coverage)],
         ["Thresholds", `min score ${data.thresholds.min_score}, min gap ${data.thresholds.min_gap}` +
             (data.thresholds.calibrated ? " (calibrated)" : " (NOT calibrated)")],
         ["Latency per query", `embedding ${data.latency_ms.embedding} ms, search ${data.latency_ms.search} ms`]]
    ));

    const groupRows = (obj) => Object.entries(obj).map(([name, m]) =>
        [name, m.n_queries, metricCell(m["P@1"]), metricCell(m["P@5"]), metricCell(m["Recall@5"])]);
    addLine(evalContent, "h3", "", "By domain");
    evalContent.append(makeTable(["Domain", "Queries", "P@1", "P@5", "Recall@5"], groupRows(data.by_domain)));
    addLine(evalContent, "h3", "", "By viewpoint / image-quality tag");
    evalContent.append(makeTable(["Tag", "Queries", "P@1", "P@5", "Recall@5"], groupRows(data.by_variation)));

    (data.warnings || []).forEach((w) => addLine(evalContent, "p", "eval-warning", `Warning: ${w}`));
}

async function loadEvaluation() {
    try {
        const response = await fetch(`${API_BASE}/evaluation`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "No evaluation results.");
        renderEvaluation(data);
    } catch (error) {
        evalContent.replaceChildren();
        addLine(evalContent, "p", "muted",
            "No evaluation results yet. Add val/test query images, then run " +
            "python backend/evaluate.py --calibrate and python backend/evaluate.py.");
    }
}

// Check the backend when the frontend loads.
checkBackend();
loadEvaluation();
