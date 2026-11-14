const API_BASE = (window.QA_API_BASE || "http://127.0.0.1:8000").replace(/\/+$/, "");

const listEl = document.getElementById("historyList");
const emptyEl = document.getElementById("historyEmpty");
const summaryEl = document.getElementById("historySummary");
const searchInput = document.getElementById("historySearch");
const filterButtons = document.getElementById("filterButtons");
const statusEl = document.getElementById("backendStatus");

const state = { filter: "all", query: "" };

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function num(value, digits = 3) {
    return value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);
}

function matchesFilter(entry) {
    const status = entry.decision && entry.decision.status;
    if (state.filter !== "all" && status !== state.filter) return false;
    if (!state.query) return true;
    const haystack = [
        entry.filename, entry.domain, entry.category && entry.category.label,
        ...(entry.results || []).map((r) => `${r.instance_id} ${r.category || ""}`),
    ].join(" ").toLowerCase();
    return haystack.includes(state.query.toLowerCase());
}

function renderSummary(all) {
    summaryEl.replaceChildren();
    const matches = all.filter((e) => e.decision && e.decision.status === "MATCH").length;
    [["Searches", all.length], ["Potential matches", matches], ["UNKNOWN", all.length - matches]]
        .forEach(([label, value]) => {
            const chip = el("div", "summary-chip");
            chip.append(el("strong", "", String(value)), el("span", "", label));
            summaryEl.append(chip);
        });
}

function renderEntry(entry) {
    const decision = entry.decision || {};
    const isMatch = decision.status === "MATCH";
    const card = el("article", `history-card ${isMatch ? "is-match" : "is-unknown"}`);

    const thumb = el("img", "h-thumb");
    thumb.src = entry.thumb || "";
    thumb.alt = `Query image ${entry.filename}`;

    const body = el("div", "h-body");

    const head = el("div", "h-head");
    const titleBox = el("div");
    titleBox.append(el("h3", "h-title", entry.filename || "image"),
        el("p", "h-time", new Date(entry.time).toLocaleString()));
    const del = el("button", "h-delete", "\u00d7");
    del.type = "button";
    del.title = "Remove this entry";
    del.addEventListener("click", () => { QAHistory.remove(entry.id); render(); });
    head.append(titleBox, del);

    const badges = el("div", "h-badges");
    badges.append(el("span", `badge ${isMatch ? "badge-match" : "badge-unknown"}`, isMatch ? "POTENTIAL MATCH" : "UNKNOWN"));
    if (entry.category) {
        badges.append(el("span", "chip", `AI: ${entry.category.label} (${Math.round(entry.category.confidence * 100)}%)`));
    }
    badges.append(el("span", "chip", `domain: ${entry.domain || "all"}`));

    const evidence = el("p", "h-evidence",
        `Top score ${num(decision.top_score)} \u00b7 gap ${num(decision.score_gap)} \u00b7 ` +
        `thresholds ${num(decision.min_score)} / ${num(decision.min_gap)}` +
        (entry.latency_ms ? ` \u00b7 ${entry.latency_ms.total} ms` : ""));
    const reason = el("p", "h-reason", decision.reason || "");

    const results = el("div", "h-results");
    (entry.results || []).forEach((r) => {
        const item = el("figure", "h-result");
        const img = el("img");
        img.src = `${API_BASE}${r.image_url}`;
        img.alt = `Result ${r.instance_id}`;
        img.loading = "lazy";
        img.addEventListener("error", () => { img.alt = "image unavailable"; img.classList.add("broken"); });
        item.append(img, el("figcaption", "", `#${r.rank} ${r.instance_id}`), el("small", "", num(r.similarity)));
        results.append(item);
    });
    if (!(entry.results || []).length) results.append(el("p", "h-reason", "No results were returned."));

    body.append(head, badges, reason, evidence, results);
    card.append(thumb, body);
    return card;
}

function render() {
    const all = QAHistory.load();
    renderSummary(all);
    const shown = all.filter(matchesFilter);
    listEl.replaceChildren(...shown.map(renderEntry));
    emptyEl.hidden = shown.length > 0;
    if (all.length && !shown.length) {
        emptyEl.querySelector("h3").textContent = "No entries match this filter";
        emptyEl.querySelector("p").textContent = "Change the filter or the search text above.";
    } else if (!all.length) {
        emptyEl.querySelector("h3").textContent = "No searches yet";
        emptyEl.querySelector("p").replaceChildren("Run a search on the ",
            Object.assign(document.createElement("a"), { href: "index.html#search", textContent: "Search page" }),
            " and it will appear here.");
    }
}

filterButtons.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    state.filter = button.dataset.filter;
    filterButtons.querySelectorAll(".seg").forEach((b) => b.classList.toggle("active", b === button));
    render();
});
searchInput.addEventListener("input", () => { state.query = searchInput.value.trim(); render(); });

document.getElementById("clearHistory").addEventListener("click", () => {
    if (QAHistory.load().length && confirm("Remove all saved searches from this browser?")) {
        QAHistory.clear();
        render();
    }
});
document.getElementById("downloadCsv").addEventListener("click", () => {
    const rows = QAHistory.load().filter(matchesFilter);
    if (!rows.length) return;
    const blob = new Blob([QAHistory.toCSV(rows)], { type: "text/csv;charset=utf-8" });
    const link = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "qa_search_history.csv" });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

async function checkBackend() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) throw new Error();
        statusEl.className = "status connected";
        statusEl.replaceChildren(el("span", "dot"), " Backend connected");
    } catch (error) {
        statusEl.className = "status disconnected";
        statusEl.replaceChildren(el("span", "dot"), " Backend offline");
    }
}

render();
checkBackend();
