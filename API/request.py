import requests

url = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint
body = {"X": 8, "Y": 5, "month": "oct", "day": "sun", "FFMC": 70.6, "DMC": 35.4, "DC": 669.1, "ISI": 6.7, "temp": 18.0, "RH": 33, "wind": 0.9, "rain": 0.0}
response = requests.post(url, data=body)
print(response.json())
