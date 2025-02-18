import requests
import json
import base64
import os

def save_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f:
        f.write(base64.b64decode(data))

url = "http://192.168.1.87:5480/woosh/map/Download"

payload = json.dumps({
   "sceneName": "IA San Jose Office"
})
headers = {
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
response_data = json.loads(response.text)

for file_data in response_data['body']['fileDatas']:
    filename = file_data['name']
    data = file_data['data']
    save_file(data, filename)
    print(f"Saved {filename}")