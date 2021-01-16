from flask import Flask , render_template 
from flask import request
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open("model.pkl",rb))

@app.route("/")
def heart():
    return render_template("index.html")

@app.route("/predict",method=["POST"])   
def pred():
    input_features= [float(x) for x in request.form.value()]
    features_values=[np.array(input_features)]

    features_name=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang",
                    "oldpeak","slope","ca","thal"]

    df=pd.DataFrame(features_values,columns=features_name)
    output = model.predict(df)


    if output ==1:
        res="Heart disease"
    else:
        res="You are fine"

if __name__ == "__main__":
    app.run()
