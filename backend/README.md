# CI Research Copilot Backend (Local Dev)

This is the local FastAPI backend for **CI Research Copilot + Research Watch**.  
This step implements the discovery store and endpoints only (no PDF analysis yet).

## 1. Setup (venv + install)

From the `backend/` directory:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Run the server locally

From `backend/` with the virtualenv activated:

```bash
uvicorn app.main:app --reload
```

The server will start on `http://127.0.0.1:8000`.

### Optional environment variables

- `DEV_MODE` (default: `true`)
- `FRONTEND_ORIGIN` (optional; CORS will always allow `http://localhost:3000` and `http://localhost:5173`)

Example (PowerShell):

```bash
$env:DEV_MODE="true"
uvicorn app.main:app --reload
```

## 3. Endpoints

### GET /health

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{ "status": "ok" }
```

### GET /discover

Returns only items with `status="new"`:

```bash
curl http://127.0.0.1:8000/discover
```

### POST /discover/refresh

Upsert discovery items (no abstract, just `paper_id`, `title`, `pdf_url`):

```bash
curl -X POST http://127.0.0.1:8000/discover/refresh ^
  -H "Content-Type: application/json" ^
  -d "[{\"paper_id\":\"ci-2024-001\",\"title\":\"Demo CI Paper\",\"pdf_url\":\"https://example.org/demo.pdf\"}]"
```

Example response:

```json
{
  "added": 1,
  "updated": 0,
  "skipped_ignored": 0,
  "total_new": 1
}
```

### POST /discover/ignore

```bash
curl -X POST http://127.0.0.1:8000/discover/ignore ^
  -H "Content-Type: application/json" ^
  -d "{\"paper_id\":\"ci-2024-001\"}"
```

Example response:

```json
{ "ok": true, "paper_id": "ci-2024-001", "status": "ignored" }
```

## 4. research_store.json

- Lives at `backend/research_store.json`.
- Created automatically on first access if missing.
- Schema (simplified):

```json
{
  "discovered": [],
  "ignored_ids": [],
  "analyzed": []
}
```

Each discovered item minimally includes:

- `paper_id` (string)
- `title` (string)
- `pdf_url` (string or null)
- `added_at` (ISO timestamp)
- `status`: `"new" | "ignored" | "analyzed"`

## 5. Manual “major test” script

The manual discovery roundtrip test is in `tests_manual/discovery_roundtrip.py`.

### Steps

1. **Start the server** in one terminal (from `backend/`):

   ```bash
   uvicorn app.main:app --reload
   ```

2. **Run the test script** in another terminal (from `backend/` with venv active):

   ```bash
   python tests_manual/discovery_roundtrip.py
   ```

### What the script does

1. **Step 1/4**: Deletes `research_store.json` if it exists (clean slate).
2. **Step 2/4**: Calls `POST /discover/refresh` with 3 sample discovery items and prints the response.
3. **Step 3/4**: Calls `GET /discover` and prints the list of `status="new"` items.
4. **Step 4/4**: Calls `POST /discover/ignore` on one item, then calls `GET /discover` again and prints the remaining items.

You should see:

- The initial `GET /discover` returning 3 items.
- After ignoring one, the final `GET /discover` should return 2 items.

