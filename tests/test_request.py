import requests

url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}
data = {"text": "AI technology is advancing rapidly in the healthcare sector, with new breakthroughs in medical diagnosis and treatment planning."}

try:
    response = requests.post(url, json=data, headers=headers)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", str(e))
