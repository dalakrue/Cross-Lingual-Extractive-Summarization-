# frontend/ — "Briefly" React + Vite UI

## What it is

Paste an English article → get its key sentences → toggle English / မြန်မာ.

## How to run

```bash
cd frontend
npm install
npm run dev
# Then open the Vite URL. The dev server proxies /api to http://127.0.0.1:8000,
# so start the API first: uvicorn api.main:app --reload (from repo root).
```

Production build: `npm run build`, preview with `npm run preview`.

## Files

| File | What it is |
|---|---|
| `package.json` | Dependencies + scripts (React, Vite). |
| `vite.config.js` | Vite config. Proxies `/api` to the FastAPI backend. |
| `index.html` | Entry HTML, loads `/src/main.jsx`. |
| `src/main.jsx` | React entry, renders `<App />`. |
| `src/App.jsx` | The whole UI: article input, clean preview, summary length picker (5/7/10), summary output, EN/MY toggle, copy button. |
| `src/styles.css` | All styling. |

## How it talks to the backend

- `POST /api/clean` — "Preview cleaned text" button.
- `POST /api/summarize` — "Summarize article" button.
- `POST /api/translate` — မြန်မာ toggle. Only the generated summary is sent,
  to MyMemory via the backend. See `api/README.md`.

## Notes

- The original zip had duplicate `App.jsx` / `main.jsx` / `styles.css`
  at both the project root and `src/`. This repo keeps only the `src/`
  copies, which is what Vite loads.
- `node_modules/` and `dist/` are gitignored — run `npm install` locally.
