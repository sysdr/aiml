# Fix: ERR_CONNECTION_REFUSED

**Cause:** The server is not running, or the browser cannot reach it.

## 1. Start the server first

From the project folder run:

```bash
cd /home/systemdr/git/aiml/day145/substrate-freeze-project
./run-docker.sh
```

Wait until you see **"Server is up"** or the dashboard URL.

## 2. Use this URL in the browser

Open exactly (same host and port as the server):

- **http://127.0.0.1:8080/dashboard**

or

- **http://localhost:8080/dashboard**

## 3. If you use WSL2 and the browser is on Windows

- Try **http://127.0.0.1:8080/dashboard** (Windows usually forwards this to WSL2).
- If it still fails, in WSL run: `hostname -I | awk '{print $1}'` and try **http://THAT_IP:8080/dashboard** (replace THAT_IP).

## 4. Check that the server is running

In a terminal:

```bash
curl -s http://127.0.0.1:8080/status
```

You should see JSON like `{"status":"running",...}`. If you get "Connection refused", start the server with `./run-docker.sh`.

## 5. Stop and start again

```bash
docker stop substrate-freeze
./run-docker.sh
```

Then open **http://127.0.0.1:8080/dashboard** again.
