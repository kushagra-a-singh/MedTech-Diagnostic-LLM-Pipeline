
import requests
import json
import time

def test_dicom_qa():
    url = "http://localhost:8000/chat"
    
    # Mock context simulating what comes from the pipeline after DICOM analysis
    context = {
        "imaging_data": "CT Chest with contrast. Findings: Irregular mass lesion in the Right Upper Lobe (RUL) measuring 3.5 x 3.2 cm. Spiculated margins observed. No pleural effusion. Mediastinal lymphadenopathy present.",
        "clinical_context": "Patient: 65-year-old male. History: Chronic smoker (40 pack-years). Complaint: Persistent cough and weight loss.",
        "segmentation_findings": "Segmented region: Right Upper Lobe. Volume: 14.2 cc. Density: Mixed.",
        "similar_cases": "Case 1: Adenocarcinoma (85% similarity). Case 2: Squamous cell carcinoma (60% similarity)."
    }
    
    # Question asking specifically about the imaging findings
    payload = {
        "message": "Based on the imaging findings, what are the concerning features of the mass?",
        "context": context,
        "history": []
    }
    
    print("Sending request to /chat endpoint...")
    print(f"Context provided: RUL mass, 3.5cm, spiculated, lymphadenopathy")
    print(f"Question: {payload['message']}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            print("Response Status: 200 OK")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print("-" * 50)
            print("LLM Response:")
            print(data.get("response", "No response field found"))
            print("-" * 50)
            
            # Simple validation
            response_text = data.get("response", "").lower()
            if "spiculated" in response_text or "margin" in response_text or "right upper lobe" in response_text or "3.5" in response_text:
                print("✅ TEST PASSED: Response references specific imaging context.")
            else:
                print("⚠️ TEST WARNING: Response might be generic. Check context usage.")
                
        else:
            print(f"❌ Error: Status Code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("Ensure the server is running on localhost:8000")

if __name__ == "__main__":
    # Wait loop to ensure server is ready
    print("Waiting for server to be ready...")
    for i in range(30):
        try:
            r = requests.get("http://localhost:8000/health")
            if r.status_code == 200:
                print("Server is ready!")
                break
        except:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print("\n")
    
    test_dicom_qa()
