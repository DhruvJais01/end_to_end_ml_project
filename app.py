from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application


## Route for the home page
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict_datapoint():
    if request.method == "GET":
        logging.info("GET request received for prediction")
        return render_template("home.html")
    else:
        logging.info("POST request received for prediction")
        logging.info("Extracting input features from the request")
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score")),
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        logging.info(
            "Input features DataFrame created for prediction and predict pipeline initializing"
        )
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.info("Prediction completed")
        print(results)
        logging.info("Rendering prediction result on the home page")
        return render_template(
            "home.html",
            results=results[0],
        )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
