# A small program to send test messages to the api server 
import requests
import json

# Define the endpoint URL
endpoint_url = "http://localhost:8080/data"

# Define the JSON data to be sent in the request
# --- Example 1: Test a rescaling ----
print("Attempting rescaling...")
json_data = {
    "do_rescale": True,
    "Dnu_target": 50,
    "epsilon_target": 0.5,
    "d0l_target": [-1.5,-1.5,-1.5],
    "fl0":[100,200,300,400],
    "fl1":[151,251,351,451],
    "fl2":[95,195,295,395],
    "fl3":[]
}
# Convert the JSON data to a string
json_data_str = json.dumps(json_data)
# Define the headers for the request
headers = {
    "Content-Type": "application/json"
}
# Send the request
response = requests.post(endpoint_url, headers=headers, data=json_data_str)

# Print the response
print(response.text)

# --- Example 2: Test a decomposition ----
print("Attempting decomposition...")
json_data = {
    "do_rescale": False,
    "Dnu_target": -1,
    "epsilon_target": -1,
    "d0l_target": [],
    "fl0":[100,200,300,400],
    "fl1":[151,251,351,451],
    "fl2":[95,195,295,395],
    "fl3":[]
}
# Convert the JSON data to a string
json_data_str = json.dumps(json_data)
# Define the headers for the request
headers = {
    "Content-Type": "application/json"
}
# Send the request
response = requests.post(endpoint_url, headers=headers, data=json_data_str)

# Print the response
print(response.text)
