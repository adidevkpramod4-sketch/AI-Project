// Search history, kept in this browser (localStorage). Used by index.html (saves) and history.html (shows).
(function () {
    const KEY = "qa_history_v1";
    const MAX_ENTRIES = 50;          // oldest entries are dropped beyond this

    function load() {
        try {
            const list = JSON.parse(localStorage.getItem(KEY) || "[]");
            return Array.isArray(list) ? list : [];
        } catch (error) {
            return [];
        }
    }

    function persist(list) {
        // If the browser quota is exceeded, drop the oldest entries until it fits.
        while (true) {
            try {
                localStorage.setItem(KEY, JSON.stringify(list));
                return true;
            } catch (error) {
                if (list.length <= 1) return false;
                list.pop();
            }
        }
    }

    function save(entry) {
        const list = load();
        list.unshift(entry);                       // newest first
        if (list.length > MAX_ENTRIES) list.length = MAX_ENTRIES;
        return persist(list);
    }

    function remove(id) {
        persist(load().filter((entry) => entry.id !== id));
    }

    function clear() {
        localStorage.removeItem(KEY);
    }

    // Small JPEG copy of the query image so the history stays light.
    function makeThumb(file, maxSide = 240) {
        return new Promise((resolve, reject) => {
            const url = URL.createObjectURL(file);
            const img = new Image();
            img.onload = () => {
                const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
                const canvas = document.createElement("canvas");
                canvas.width = Math.max(1, Math.round(img.width * scale));
                canvas.height = Math.max(1, Math.round(img.height * scale));
                canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(url);
                resolve(canvas.toDataURL("image/jpeg", 0.7));
            };
            img.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error("Could not read the image"));
            };
            img.src = url;
        });
    }

    function toCSV(list) {
        const cell = (value) => `"${String(value ?? "").replace(/"/g, '""')}"`;
        const header = ["time", "query_file", "domain", "ai_category", "ai_confidence", "decision",
            "top_instance", "top_score", "second_score", "gap", "reason", "results_shown"];
        const rows = list.map((e) => [
            e.time, e.filename, e.domain, e.category && e.category.label, e.category && e.category.confidence,
            e.decision && e.decision.status, e.results && e.results[0] && e.results[0].instance_id,
            e.decision && e.decision.top_score, e.decision && e.decision.second_score,
            e.decision && e.decision.score_gap, e.decision && e.decision.reason,
            e.results ? e.results.length : 0,
        ].map(cell).join(","));
        return [header.join(","), ...rows].join("\r\n");
    }

    window.QAHistory = { load, save, remove, clear, makeThumb, toCSV };
})();
