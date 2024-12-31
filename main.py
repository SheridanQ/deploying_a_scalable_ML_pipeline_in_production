# Put the code for your API here.
import logging
import pandas as pd
from starter.train_model import categorical_features
from starter.ml.model import inference, load_model
from starter.ml.data import process_data
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI
from typing import Literal
import os
print(os.getcwd())


# Configure logging for debugging and tracking purposes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()


class ModelInput(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    fnlgt: int
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college', 'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    maritalStatus: Literal[
        "Never-married", "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Separated", "Married-AF-spouse", "Widowed"]
    occupation: Literal[
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
    relationship: Literal[
        "Wife", "Own-child", "Husband",
        "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal[
        "White", "Asian-Pac-Islander",
        "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    hoursPerWeek: int
    nativeCountry: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

    class Config:
        schema_extra = {
            "example": {
                "age": 33,
                "workclass": 'Private',
                "fnlgt": 77516,
                "education": 'Bachelors',
                "maritalStatus": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Male",
                "hoursPerWeek": 40,
                "nativeCountry": 'United-States'
            }
        }


@app.get("/")
async def get_root():
    return {"greeting": "Welcome to Udacity MLDevops Project, this app predicts whether income exceeds $50k/yr based on personal census data."}

# Define the model inference endpoint


@app.post("/predict")
def predict(data: ModelInput):
    # Convert the input data into a DataFrame for processing
    input_df = pd.DataFrame([{"age": data.age,
                              "workclass": data.workclass,
                              "fnlgt": data.fnlgt,
                              "education": data.education,
                              "marital-status": data.maritalStatus,
                              "occupation": data.occupation,
                              "relationship": data.relationship,
                              "race": data.race,
                              "sex": data.sex,
                              "hours-per-week": data.hoursPerWeek,
                              "native-country": data.nativeCountry}])

    # Logging the input data for debugging purposes
    print(input_df.to_dict())
    logger.info(f" input_data: {input_df.to_dict()}")

    # Process the input data using the predefined process_data function
    trained_model, encoder, lb = load_model()
    X, _, _, _ = process_data(
        input_df, categorical_features=categorical_features, label=None, training=False, encoder=encoder, lb=lb
    )

    # Perform inference
    predictions = inference(trained_model, X)

    # Convert predictions back to original labels
    preds = lb.inverse_transform(predictions)[0]

    logging.info(f" predictions: {preds}")

    # Return the predictions in JSON format
    return {"predictions": preds}


# Run the FastAPI app if the script is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
