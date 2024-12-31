import requests


data = {
    "age": 20,
    "workclass": "State-gov",
    "fnlgt": 0,
    "education": "Bachelors",
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Sales",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "hoursPerWeek": 40,
    "nativeCountry": "United-States"
}

r = requests.post('https://udacity-mlops-pipeline-deploy.herokuapp.com//predict',
                  json=data)

assert r.status_code == 200

print(f"Response code: {r.status_code}")
print(f"Response body: {r.json()}")