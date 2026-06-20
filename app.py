import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_wtf import FlaskForm
from wtforms import SelectField, IntegerField, DecimalField, StringField, SubmitField
from wtforms.validators import DataRequired, NumberRange, InputRequired

from config import SECRET_KEY, FLASK_ENV, PRICE_CURRENCY, PRICE_CONVERSION_FACTOR
from services.model_service import load_model_registry, load_best_model, load_model_artifact, ModelLoadingError
from services.prediction_service import run_prediction
from services.validation_service import PredictionFormValidator
from services.analytics_service import load_analytics_artifacts
from services.formatting_service import format_currency, format_number
from services.explanation_service import generate_value_observations

app = Flask(
    __name__,
    static_folder="public",
    static_url_path="/",
    template_folder="templates"
)
app.config["SECRET_KEY"] = SECRET_KEY
app.jinja_env.filters["currency"] = format_currency
app.jinja_env.globals["price_currency"] = PRICE_CURRENCY
app.jinja_env.globals["price_factor"] = PRICE_CONVERSION_FACTOR


class CarPredictionForm(FlaskForm):
    brand = SelectField("Brand", validators=[DataRequired(message="Brand is required")], choices=[])
    model = SelectField("Model", validators=[DataRequired(message="Model is required")], choices=[])
    year = IntegerField("Manufacturing Year", validators=[InputRequired(), NumberRange(min=1980, max=datetime.now().year)])
    fuel_type = SelectField("Fuel Type", validators=[DataRequired()], choices=[])
    transmission = SelectField("Transmission", validators=[DataRequired()], choices=[])
    mileage_km = IntegerField("Mileage (km)", validators=[InputRequired(), NumberRange(min=0, max=1000000)])
    engine_size_cc = IntegerField("Engine Size (cc)", validators=[InputRequired(), NumberRange(min=500, max=10000)])
    owner_count = IntegerField("Previous Owners", validators=[InputRequired(), NumberRange(min=0, max=10)])
    model_choice = SelectField("Model Selection", validators=[DataRequired()], choices=[("best", "Best available model"), ("linear_regression", "Linear Regression"), ("random_forest", "Random Forest")])
    submit = SubmitField("Predict Price")


def load_global_artifacts():
    analytics = {}
    model_registry = {}
    best_model = {}
    try:
        analytics = load_analytics_artifacts()
    except Exception:
        analytics = {}
    try:
        model_registry = load_model_registry()
    except ModelLoadingError:
        model_registry = {}
    try:
        best_model = load_best_model()
    except Exception:
        best_model = {}
    return analytics, model_registry, best_model


@app.route("/")
def index():
    analytics, model_registry, best_model = load_global_artifacts()
    return render_template(
        "index.html",
        analytics=analytics,
        model_registry=model_registry,
        best_model=best_model,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    analytics, model_registry, best_model = load_global_artifacts()
    feature_meta = analytics.get("feature_metadata", {})
    form = CarPredictionForm()
    
    def populate_choices():
        form.brand.choices = [(brand, brand) for brand in feature_meta.get("brands", [])]
        form.fuel_type.choices = [(fuel, fuel) for fuel in feature_meta.get("fuel_types", [])]
        form.transmission.choices = [(trans, trans) for trans in feature_meta.get("transmissions", [])]
        form.model_choice.choices = [
            ("best", "Best available model"),
            ("linear_regression", "Linear Regression"),
            ("random_forest", "Random Forest"),
        ]
    
    populate_choices()

    if request.method == "POST":
        selected_model = request.form.get("model_choice", "best")
        raw_data = {
            "brand": request.form.get("brand", "").strip(),
            "model": request.form.get("model", "").strip(),
            "year": request.form.get("year", ""),
            "fuel_type": request.form.get("fuel_type", "").strip(),
            "transmission": request.form.get("transmission", "").strip(),
            "mileage_km": request.form.get("mileage_km", ""),
            "engine_size_cc": request.form.get("engine_size_cc", ""),
            "owner_count": request.form.get("owner_count", ""),
        }
        validator = PredictionFormValidator(feature_meta)
        is_valid, errors = validator.validate(raw_data)
        if not is_valid:
            for error in errors:
                flash(error, "danger")
            return render_template("predict.html", form=form, analytics=analytics, feature_meta=feature_meta, selected_model=selected_model)

        try:
            prediction_result = run_prediction(raw_data, selected_model)
        except Exception as exc:
            flash(str(exc), "danger")
            return render_template("predict.html", form=form, analytics=analytics, feature_meta=feature_meta, selected_model=selected_model)

        observations = generate_value_observations(raw_data, analytics)
        return render_template(
            "result.html",
            result=prediction_result,
            observations=observations,
            analytics=analytics,
        )

    return render_template("predict.html", form=form, analytics=analytics, feature_meta=feature_meta, selected_model="best")


@app.route("/dashboard")
def dashboard():
    analytics, model_registry, best_model = load_global_artifacts()
    return render_template("dashboard.html", analytics=analytics, model_registry=model_registry, best_model=best_model)


@app.route("/analysis")
def analysis():
    analytics, model_registry, best_model = load_global_artifacts()
    return render_template("analysis.html", analytics=analytics, model_registry=model_registry, best_model=best_model)


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/health")
def health():
    _, model_registry, best_model = load_global_artifacts()
    environment = "vercel" if request.headers.get("x-vercel-proxy-region") else "local"
    return jsonify(
        status="healthy",
        models_available=bool(model_registry.get("models")),
        best_model=best_model.get("display_name", "Unknown") if best_model else None,
        currency=PRICE_CURRENCY,
        environment=environment,
    )


@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(error):
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=FLASK_ENV.lower() == "development")
