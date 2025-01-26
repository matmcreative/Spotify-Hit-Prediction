# Spotify-Hit-Prediction

This project predicts the likelihood of a song becoming a hit based on its Spotify audio features. Additionally, it evaluates trends in song characteristics over time and provides an interactive Streamlit app for user-friendly predictions.

## **Features**

- **Data Exploration**:
  - Analyze audio features like `danceability`, `energy`, `tempo`, and their relationship with Billboard chart positions.
  - Identify correlations between Spotify audio features and a song’s success.

- **Classification Task**:
  - Predict whether a song ranks in the **Top 10** on the Billboard charts (binary classification).

- **Regression Task**:
  - Predict `chart_position` (lower is better) or `weeks_on_chart` to estimate song longevity.

- **Evaluation Metrics**:
  - **Regression**: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).
  - **Classification**: Accuracy, precision, recall, F1-score, and confusion matrix.

- **Interactive App**:
  - A **Streamlit** app allows users to predict song success interactively by providing audio feature inputs.

## **Project Structure**

- **Data Preprocessing**:
  - Merge Spotify and Billboard datasets.
  - Create derived features (e.g., `top_10` for binary classification).
  - Handle class imbalance using techniques like SMOTE or class weights.

- **Modeling**:
  - Train models like **Random Forest Regressor**, **XGBoost Classifier**, and **Logistic Regression**.
  - Perform hyperparameter tuning using `GridSearchCV` for optimal performance.

- **Visualization**:
  - Feature importance analysis.
  - Correlation heatmaps.
  - Trends in features over time (e.g., danceability vs. chart date).

## **Requirements**

- Python 3.8+
- Libraries:
  - `pandas`, `numpy`: Data manipulation.
  - `matplotlib`, `seaborn`: Visualization.
  - `scikit-learn`: Machine learning models and evaluation metrics.
  - `xgboost`: Gradient Boosting Classifier.
  - `streamlit`: Interactive app development.
  - `imblearn`: Oversampling techniques for handling class imbalance.

## **How to Run**

### 1. **Data Preparation**
- Ensure the following datasets are available:
  - `spotify_songs.csv`: Contains Spotify audio features.
  - `billboard_data.csv`: Contains Billboard chart rankings.

### 2. **Install Dependencies**
Install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn streamlit
```

### 3. **Run the Notebook**
- Open the Jupyter Notebook: `hit_prediction.ipynb`.
- Run all cells sequentially to:
  - Prepare data.
  - Train models.
  - Evaluate performance.
  - Visualize trends and predictions.

### 4. **Launch Streamlit App**
Run the following command to start the Streamlit app:
```bash
streamlit run hit_prediction_app.py
```
Interact with the app to predict song success by adjusting sliders for audio features like tempo, energy, and loudness.

## **Key Results**

- **Baseline Model**:
  - Compared the Random Forest model to a baseline (mean prediction) for benchmarking.

- **Classification Performance**:
  - Improved prediction of `top_10` hits using oversampling and hyperparameter tuning.

- **Feature Importance**:
  - Identified key features influencing song success, such as `danceability` and `energy`.

## **Future Improvements**

1. **Advanced Feature Engineering**:
   - Add interaction terms like `tempo_to_energy_ratio`.
   - Normalize features for better model performance.

2. **Explainable AI**:
   - Use SHAP to explain individual predictions.

3. **Additional Models**:
   - Test other classifiers like Support Vector Machines (SVM) or Gradient Boosting.

4. **Enhanced Visualizations**:
   - Include rolling averages for trends analysis.
   - Plot precision-recall curves for classification tasks.

## **Acknowledgments**

- **Spotify Web API** for providing song audio features.
- **Billboard Charts** for chart performance data.

---
Feel free to extend or adapt this project for other predictive tasks related to music analytics!

