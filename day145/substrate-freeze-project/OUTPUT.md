# Substrate Freeze Service — Expected Output

## How to run

**Option A — Docker (recommended)**

```bash
cd /home/systemdr/git/aiml/day145/substrate-freeze-project
docker build -t substrate-freeze-app .
docker run -d --rm -p 8080:8080 --name substrate-freeze substrate-freeze-app
```

Then open: **http://localhost:8080/dashboard**

**Option B — Native Go**

```bash
cd src
go mod init substrate-freeze-service  # if not exists
go build -o ../substrate-freeze-app .
cd ..
./substrate-freeze-app
```

Then open: **http://localhost:8080/dashboard**

---

## API output (curl examples)

### Root

```bash
curl -s http://localhost:8080/
```

**Output:**

```
Welcome to the Substrate Freeze Service! Try /set, /get, /freeze, /unfreeze, /status, /dump, or /dashboard.
```

---

### GET /status

```bash
curl -s http://localhost:8080/status
```

**Output (example):**

```json
{"status":"running","timestamp":"2026-03-13T06:25:02Z"}
```

When frozen: `"status":"frozen"`.

---

### GET /dump

```bash
curl -s http://localhost:8080/dump
```

**Output when empty:**

```json
{}
```

**Output after SET demo=ok:**

```json
{"demo":"ok"}
```

---

### POST /set

```bash
curl -s -X POST "http://localhost:8080/set?key=demo&value=ok"
```

**Output:**

```
Set key 'demo' to 'ok'
```

---

### GET /get

```bash
curl -s "http://localhost:8080/get?key=demo"
```

**Output:**

```
Value for key 'demo': 'ok'
```

If key is missing: HTTP 404 and body `Key 'demo' not found`.

---

### POST /freeze

```bash
curl -s -X POST http://localhost:8080/freeze
```

**Output:**

```
Service is now FROZEN. New SET operations will be rejected, GET operations continue.
```

---

### POST /unfreeze

```bash
curl -s -X POST http://localhost:8080/unfreeze
```

**Output:**

```
Service is now UNFROZEN. SET operations can resume.
```

---

## Dashboard (http://localhost:8080/dashboard)

- **Status:** Shows `running` or `frozen` and a timestamp.
- **Store dump:** Shows `(empty)` or the current key/value JSON.
- **Buttons:** SET, GET, Freeze, Unfreeze, Refresh — each updates **Last result** and refreshes status/dump.
- **Last result:** Shows the response body of the last action.

Use the **same host and port** as the server (e.g. `http://localhost:8080`). Do not open the HTML from `file://` or another port, or the dashboard will not load data.
