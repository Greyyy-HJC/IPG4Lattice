# presentation

The HTML progress report was moved to the **repository root** as `progress_report.html`.

Reason: Safari (WebKit) rejects `file://` navigation from a page under `presentation/` to PDFs under `../artifacts/` (“outside the sandbox”). With the report next to `artifacts/`, links stay as `artifacts/plots/...` and local PDF buttons work.

Alternative for any browser: from the clone root run `python3 -m http.server` and open `http://localhost:8000/progress_report.html`.
