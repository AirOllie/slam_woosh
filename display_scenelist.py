import requests
import json

url = "http://192.168.1.87:5480/woosh/map/SceneList"

payload = json.dumps({})
headers = {
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)