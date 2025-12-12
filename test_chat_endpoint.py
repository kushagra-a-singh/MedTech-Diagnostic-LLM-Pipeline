
import requests
import time
import json
import sys

def test_chat():
    print("Waiting for server to start...")
    # Wait for checking health
    for i in range(60):
        try:
            response = requests.get('http://localhost:8000/health')
            if response.status_code == 200:
                print("\nServer is ready!")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(2)
            sys.stdout.write(".")
            sys.stdout.flush()
    else:
        print("\nServer failed to start in time.")
        return

    print("\nTesting Chatbot with MedGemma model...")
    
    payload = {
        "message": "What are the common symptoms of pneumonia?",
        "session_id": "test-session-1",
        "stream": False
    }
    
    try:
        start_time = time.time()
        response = requests.post('http://localhost:8000/chat', json=payload)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ Chat Response Received:")
            print("-" * 50)
            print(data.get('response'))
            print("-" * 50)
            print(f"Time taken: {duration:.2f} seconds")
        else:
            print(f"\n❌ Error: Status {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"\n❌ Request failed: {e}")

if __name__ == "__main__":
    test_chat()
