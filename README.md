# Car Price Prediction – From Scraping to Deployment

This project is an end-to-end data science pipeline that predicts used car prices based on various features like brand, mileage, fuel type, and more. It covers everything from data collection to model deployment.

## Project Structure

```
car_price_predictor/
│
├── app/
│   └── streamlit_app.py        # Streamlit app for deployment
│
├── data/
│   ├── Audi.csv
│   ├── Benz.csv
│   ├── BMW.csv
│   ├── Volkswagen.csv
│   ├── merged_cars_data.csv
│   └── processed_cars_data.csv
│
├── notebooks/
│   ├── 01_data_collection.ipynb           # Web scraping with BeautifulSoup
│   ├── 02_data_preparation__analysis.ipynb # Cleaning, EDA, and Feature Engineering
│   └── 03_model_training.ipynb            # Model training and tuning
│
├── saved_models/
│   └── model.pkl                # Final trained Random Forest model
```

## Features

- Data scraping from multiple car brand pages using BeautifulSoup
- Data cleaning and preprocessing (e.g., date parsing, mileage formatting)
- Feature engineering (e.g., Car Age at Sale)
- EDA using Seaborn and Matplotlib
- Model training (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
- Hyperparameter tuning with GridSearchCV
- Streamlit deployment with a responsive UI

## Best Model

- Model: Random Forest Regressor
- R² Score (Test Set): ~0.66
- MAE: ~€5,018  
- RMSE: ~€8,375

## How to Run

1. Clone this repository or download the folder
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Train the model or use the saved model at `saved_models/model.pkl`
4. Run the Streamlit app:
   ```
   streamlit run app/streamlit_app.py
   ```

## Author

Jivesh Dhakate  
MSc Computer Science, University College Dublin
# Car-Price-Prediction-From-Data-Scraping-to-Deployment
