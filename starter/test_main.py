from main import app
from fastapi.testclient import TestClient
import os
print(os.getcwd())

client = TestClient(app)


def test_get():
    """
    Test GET() on the root for giving a welcome message.
    """

    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "greeting": "Welcome to Udacity MLDevops Project, this app predicts whether income exceeds $50k/yr based on personal census data."}


def test_predict_endpoint():

    response = client.post("/predict/", json={
        "age": "30",
        "workclass": "Private",
        "fnlgt": "0",
        "education": "Bachelors",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"}, follow_redirects=True)
    print(response.json())

    assert response.status_code == 200, response.json()


def test_validation_fields():

    response = client.post("/predict/", json={
        "fnlgt": "0",
        "education": "Bachelors",
        "education_num": "13",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 1000,
        "capital-loss": 0,
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"})

    assert response.status_code == 422
    assert len(response.json()['detail']) == 2


def test_post_less_50k():
    """
    Test the predict output for salary >=50k.
    """
    input_dict = {
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
    response = client.post("/predict/", json=input_dict, follow_redirects=True)
    assert response.status_code == 200
    assert response.json()["predictions"] == '<=50K'


def test_post_greater_50k():
    input_dict = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "hoursPerWeek": 50,
        "nativeCountry": "United-States"
    }
    response = client.post("/predict/", json=input_dict, follow_redirects=True)
    assert response.status_code == 200
    assert response.json()["predictions"] == ">50K"


if __name__ == "__main__":
    test_get()
    test_predict_endpoint()
    test_validation_fields()
    test_post_less_50k()
    test_post_greater_50k()
