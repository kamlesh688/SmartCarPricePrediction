# Smart Car Price Prediction System

A professional machine-learning web application that estimates used car resale prices using Linear Regression and Random Forest models.

## Overview

This application accepts vehicle information, generates dataset-based price predictions, and displays an empirical prediction interval, price category, and model comparison dashboard.

## Features

- Predict indicative used-car resale prices in INR
- Linear Regression and Random Forest comparison
- Dataset analysis and visualizations
- Model selection with saved artifacts
- Flask web application deployable to Vercel
- Local dataset preparation and model training workflow

## Requirements

- Python 3.12
- A public used-car price dataset placed at `dataset/raw/car_price_dataset.csv`
- At least 500 valid records after cleaning

## Expected dataset columns

The project accepts flexible dataset columns, including:

- Brand: `brand`, `make`, `manufacturer`, `company`, `car_brand`
- Model: `model`, `car_model`, `vehicle_model`, `name`
- Year: `year`, `model_year`, `manufacturing_year`, `manufacture_year`
- Fuel type: `fuel`, `fuel_type`, `fueltype`
- Transmission: `transmission`, `transmission_type`, `gearbox`
- Engine size: `engine`, `engine_size`, `engine_cc`, `engine_capacity`, `displacement`
- Mileage: `mileage`, `km_driven`, `kilometers_driven`, `odometer`, `distance_driven`
- Owner count: `owner`, `owners`, `owner_count`, `previous_owners`, `number_of_owners`
- Selling price: `selling_price`, `price`, `resale_price`, `sale_price`, `listed_price`, `car_price`

## Setup

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/inspect_dataset.py
python scripts/prepare_dataset.py
python train_models.py
python scripts/verify_artifacts.py
pytest
python app.py
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python scripts/inspect_dataset.py
python scripts/prepare_dataset.py
python train_models.py
python scripts/verify_artifacts.py
pytest
python app.py
```

Local URL: http://127.0.0.1:5000

## Deployment

Follow `VERCEL_DEPLOYMENT.md` for Vercel deployment steps.

## Privacy

Vehicle information submitted on the prediction page is processed only to generate the current estimate and is not stored by the application.

## Limitations

- Requires a local dataset at `dataset/raw/car_price_dataset.csv`
- Model training is performed locally and artifacts must be committed
- Predictions are dataset-based estimates, not official valuations
