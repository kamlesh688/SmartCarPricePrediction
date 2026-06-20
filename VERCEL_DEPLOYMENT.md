# Vercel Deployment Guide

## Predeployment Steps

1. Place the dataset at `dataset/raw/car_price_dataset.csv`.
2. Inspect the dataset: `python scripts/inspect_dataset.py`
3. Prepare the dataset: `python scripts/prepare_dataset.py`
4. Train both models: `python train_models.py`
5. Verify artifacts: `python scripts/verify_artifacts.py`
6. Run tests: `pytest`
7. Confirm model artifacts and required JSON files are committed.
8. Confirm raw dataset files and reports are excluded from deployment.

## Deploy to Vercel

```bash
npm install -g vercel
vercel login
vercel
vercel --prod
```

## Environment Variables

Set the following in Vercel:

- SECRET_KEY
- FLASK_ENV=production
- PRICE_CURRENCY=INR
- PRICE_SYMBOL=₹
- PRICE_CONVERSION_FACTOR=85.0
- PRICE_CONVERSION_FACTOR=1.0

## Validation

- Open `/health` and confirm the app is healthy.
- Test `/`, `/predict`, `/dashboard`, and `/analysis`.

## Notes

- Models must be trained locally.
- The deployed app uses pretrained artifacts only.
- No dataset download or file generation occurs at runtime.
