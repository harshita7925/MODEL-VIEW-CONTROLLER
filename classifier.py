import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL.ImageOps
from flask import Flask,jsonify,request

X=np.load("image.npz")["arr_0"]
y=pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())

classes=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nclasses=len(classes)

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=9,train_size=3500,test_size=500)

classifier=LogisticRegression(random_state=9).fit(X_train_scale,y_train)
def get_prediction(image):
    im_pil=Image.open(image)
    image_bw=im_pil.convert("L")
    image_bw_resized=image_bw.resize((22,30),Image.ANTIALIAS)
    pixel_filter=20
    min_pixel=np.percentile(image_bw_resized,pixel_filter)
    image_bw_resized_inverted_scaled=np.clip(image_bw_resized-min_pixel,0,255)
    max_pixel=np.max(image_bw_resized)
    image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,660)
    test_pred=classifier.predict(test_sample)
    return test_pred[0]

from classifier import get_predictions

app=Flask(__name__)
@app.route("/predict-alphabet",methods=["POST"])

def predict_data():
    image=request.files.get("alphabet")
    prediction=get_prediction(image)
    return jsonify({
        "prediction":prediction
    },200)

if __name__=="__main__":
    app.run(debug=True)