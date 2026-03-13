package main

import (
    "encoding/json"
    "fmt"
    "log"
    "net/http"
    "strconv"
    "sync"
    "sync/atomic"
    "time"
)

var (
    dataStore = make(map[string]string)
    rwMutex   sync.RWMutex
    isFrozen  atomic.Bool
)

func main() {
    http.HandleFunc("/set", setHandler)
    http.HandleFunc("/get", getHandler)
    http.HandleFunc("/freeze", freezeHandler)
    http.HandleFunc("/unfreeze", unfreezeHandler)
    http.HandleFunc("/status", statusHandler)
    http.HandleFunc("/dump", dumpHandler)
    http.HandleFunc("/dashboard", dashboardHandler)
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Welcome to the Substrate Freeze Service! Try /set, /get, /freeze, /unfreeze, /status, /dump, or /dashboard.")
    })

    addr := "0.0.0.0:" + strconv.Itoa(8080)
    log.Printf("Server starting on %s", addr)
    log.Fatal(http.ListenAndServe(addr, nil))
}

func setHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost && r.Method != http.MethodPut {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    if isFrozen.Load() {
        log.Printf("SET request rejected: Service is frozen. Key: %s", r.URL.Query().Get("key"))
        http.Error(w, "Service is frozen for writes. Please unfreeze to perform SET operations.", http.StatusServiceUnavailable)
        return
    }

    key := r.URL.Query().Get("key")
    value := r.URL.Query().Get("value")

    if key == "" || value == "" {
        http.Error(w, "Key and value are required", http.StatusBadRequest)
        return
    }

    rwMutex.Lock()
    defer rwMutex.Unlock()

    dataStore[key] = value
    log.Printf("SET: Key='%s', Value='%s'", key, value)
    w.WriteHeader(http.StatusOK)
    fmt.Fprintf(w, "Set key '%s' to '%s'", key, value)
}

func getHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    key := r.URL.Query().Get("key")
    if key == "" {
        http.Error(w, "Key is required", http.StatusBadRequest)
        return
    }

    rwMutex.RLock()
    defer rwMutex.RUnlock()

    if val, ok := dataStore[key]; ok {
        log.Printf("GET: Key='%s', Value='%s'", key, val)
        w.WriteHeader(http.StatusOK)
        fmt.Fprintf(w, "Value for key '%s': '%s'", key, val)
    } else {
        log.Printf("GET: Key='%s' not found", key)
        http.Error(w, fmt.Sprintf("Key '%s' not found", key), http.StatusNotFound)
    }
}

func freezeHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    if isFrozen.Load() {
        log.Println("Already frozen. No action taken.")
        w.WriteHeader(http.StatusOK)
        fmt.Fprintln(w, "Service is already frozen.")
        return
    }

    log.Println("Attempting to freeze service...")
    rwMutex.Lock() // Acquire the write lock, blocking new writes. Existing writes complete.
    isFrozen.Store(true)
    log.Println("Service FROZEN. New SET operations will be rejected.")
    w.WriteHeader(http.StatusOK)
    fmt.Fprintln(w, "Service is now FROZEN. New SET operations will be rejected, GET operations continue.")
}

func unfreezeHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodPost {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    if !isFrozen.Load() {
        log.Println("Already unfrozen. No action taken.")
        w.WriteHeader(http.StatusOK)
        fmt.Fprintln(w, "Service is already unfrozen.")
        return
    }

    log.Println("Attempting to unfreeze service...")
    isFrozen.Store(false)
    rwMutex.Unlock() // Release the write lock, allowing new writes.
    log.Println("Service UNFROZEN. SET operations can resume.")
    w.WriteHeader(http.StatusOK)
    fmt.Fprintln(w, "Service is now UNFROZEN. SET operations can resume.")
}

func statusHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }

    status := "running"
    if isFrozen.Load() {
        status = "frozen"
    }
    
    response := map[string]string{"status": status, "timestamp": time.Now().Format(time.RFC3339)}
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}

func dumpHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    rwMutex.RLock() // Acquire read lock to ensure consistent dump
    defer rwMutex.RUnlock()

    log.Println("Dumping current dataStore state")
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(dataStore)
}

func dashboardHandler(w http.ResponseWriter, r *http.Request) {
    if r.Method != http.MethodGet {
        http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
        return
    }
    w.Header().Set("Content-Type", "text/html; charset=utf-8")
    fmt.Fprint(w, `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Substrate Freeze Dashboard</title>
<style>
body{font-family:sans-serif;max-width:800px;margin:1rem auto;padding:1rem;background:#1a1a2e;color:#eee;}
h1{color:#0f4;}
.card{background:#16213e;padding:1rem;margin:1rem 0;border-radius:8px;}
button{padding:0.5rem 1rem;margin:0.25rem;cursor:pointer;border-radius:4px;border:none;}
button.primary{background:#0f4;color:#000;}
button.danger{background:#c33;color:#fff;}
button.secondary{background:#444;color:#eee;}
pre{overflow:auto;font-size:12px;margin:0;}
#status{font-weight:bold;}
</style></head>
<body>
<h1>Substrate Freeze Service — Dashboard</h1>
<div class="card">Status: <span id="status">…</span> <span id="ts"></span></div>
<div class="card"><strong>Store dump:</strong><pre id="dump">…</pre></div>
<div class="card">
  <button type="button" class="primary" id="btn-set">SET key=demo value=ok</button>
  <button type="button" class="primary" id="btn-get">GET key=demo</button>
  <button type="button" class="danger" id="btn-freeze">Freeze</button>
  <button type="button" class="primary" id="btn-unfreeze">Unfreeze</button>
  <button type="button" class="secondary" id="btn-refresh">Refresh</button>
</div>
<div class="card"><strong>Last result:</strong><pre id="result">-</pre></div>
<script>
(function(){
  var api = window.location.origin + '/';
  var statusEl = document.getElementById('status');
  var tsEl = document.getElementById('ts');
  var dumpEl = document.getElementById('dump');
  var resultEl = document.getElementById('result');

  function showResult(s) { resultEl.textContent = s; }

  function refresh() {
    fetch(api + 'status')
      .then(function(r) { return r.ok ? r.json() : Promise.reject(new Error(r.status)); })
      .then(function(d) {
        statusEl.textContent = d.status;
        tsEl.textContent = d.timestamp || '';
      })
      .catch(function() {
        statusEl.textContent = 'error';
        tsEl.textContent = '';
      });

    fetch(api + 'dump')
      .then(function(r) { return r.ok ? r.text() : Promise.reject(new Error(r.status)); })
      .then(function(t) {
        try {
          var data = JSON.parse(t);
          dumpEl.textContent = Object.keys(data).length === 0 ? '(empty)' : JSON.stringify(data, null, 2);
        } catch (e) {
          dumpEl.textContent = t;
        }
      })
      .catch(function() { dumpEl.textContent = '(error loading)'; });
  }

  function run(method, path, body) {
    var opts = { method: method || 'GET' };
    fetch(api + path, opts)
      .then(function(r) { return r.text(); })
      .then(function(text) {
        showResult(text);
        refresh();
      })
      .catch(function(err) {
        showResult('Error: ' + err.message);
        refresh();
      });
  }

  document.getElementById('btn-set').onclick = function() { run('POST', 'set?key=demo&value=ok'); };
  document.getElementById('btn-get').onclick = function() { run('GET', 'get?key=demo'); };
  document.getElementById('btn-freeze').onclick = function() { run('POST', 'freeze'); };
  document.getElementById('btn-unfreeze').onclick = function() { run('POST', 'unfreeze'); };
  document.getElementById('btn-refresh').onclick = refresh;

  setInterval(refresh, 3000);
  refresh();
})();
</script>
</body></html>`)
}
