import requests
import json

url = "http://192.168.1.87:5480/woosh/robot/SetRobotPose"

payload = json.dumps({
   "pose": {
      "x": 0,
      "y": 0,
      "theta": 1.57
   }
})
headers = {
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)