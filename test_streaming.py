
import requests
import json
import sys

def test_streaming():
    url = "http://localhost:8000/chat"
    payload = {
        "message": "Explain the concept of pneumothorax briefly.",
        "stream": True,
        "session_id": "test_stream_session"
    }
    
    print(f"Connecting to {url}...")
    try:
        with requests.post(url, json=payload, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return

            print("Stream started:")
            print("-" * 20)
            
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    text = chunk.decode('utf-8')
                    print(text, end='', flush=True)
            
            print("\n" + "-" * 20)
            print("Stream finished.")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_streaming()
