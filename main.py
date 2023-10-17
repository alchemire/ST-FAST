from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pickle

# instance
app = FastAPI()

# define input data type
class iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# load model
model = pickle.load(open("models/model_iris", "rb"))

# top page
@app.get("/")
def index():
    return{"Iris": "iris_prediction"}

# when post
@app.post("/predict")
def make_predictions(features: iris):
    return({"prediction":str(model.predict([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])[0])})