import threading
import time
from pyngrok import ngrok
import uvicorn

# --- Start ngrok tunnel ---
def start_ngrok():
    public_url = ngrok.connect(8000, "http")
    print(f" * ngrok tunnel running at: {public_url.public_url}")
    print(" * You can now access your FastAPI app via the above public URL.")

# --- Start FastAPI server ---
def start_uvicorn():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    # Start ngrok in a separate thread
    ngrok_thread = threading.Thread(target=start_ngrok, daemon=True)
    ngrok_thread.start()

    # Give ngrok a moment to initialize
    time.sleep(2)

    # Start FastAPI (blocking call)
    start_uvicorn()
