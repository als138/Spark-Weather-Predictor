# 🌦️ Spark Weather Predictor

A distributed machine learning project using **Apache Spark** and **PySpark** to predict **rain/no rain** conditions from weather data, leveraging **Logistic Regression**. The dataset is stored and processed in **HDFS**, and the model is optimized using **Cross-Validation**.

---

## 📌 Project Overview

This project uses **Apache Spark** for distributed data processing and **PySpark MLlib** for building a classification model that predicts whether it will rain based on meteorological features. The workflow covers:

- Loading weather data from **HDFS**
- Preprocessing and feature engineering
- Training a **Logistic Regression** model
- Model evaluation with accuracy, F1-score, confusion matrix, and ROC curve
- Saving the model to both **HDFS** and **local file system**
- Optimizing the model with **Cross-Validation**
- Visualizing model performance

---

## 📊 Dataset

The dataset (`weather_forecast_data.csv`) contains the following columns:

| Feature        | Type   | Description |
|----------------|--------|-------------|
| `Temperature`  | double | Air temperature |
| `Humidity`     | double | Relative humidity (%) |
| `Wind_Speed`   | double | Wind speed |
| `Cloud_Cover`  | double | Cloud cover percentage |
| `Pressure`     | double | Atmospheric pressure |
| `Rain`         | string | Target label: `"rain"` or `"no rain"` |

---

## 🛠️ Technologies Used

- **Apache Spark** (PySpark)
- **HDFS** (Hadoop Distributed File System)
- **Python** (3.x)
- **Spark MLlib** for machine learning
- **Matplotlib** & **scikit-learn** for visualization and metrics
- **CrossValidator** & **ParamGridBuilder** for hyperparameter tuning

---

## 📂 Project Structure

```
Spark-Weather-Predictor/
│
├── weather_forecast_data.csv      # Weather dataset
├── spark_weather_predictor.py     # Main PySpark script
├── local_output/
│   ├── models/                    # Saved model (local)
│   └── plots/                      # ROC curve plot (local)
├── README.md                      # Project documentation
```

---

## ⚙️ How to Run

### 1️⃣ Prerequisites
- Apache Spark installed and configured
- Hadoop and HDFS running
- Python 3.x with required dependencies:
  ```bash
  pip install pyspark matplotlib scikit-learn
  ```

### 2️⃣ Upload Data to HDFS
```bash
hdfs dfs -mkdir -p /user/hduser/weather_data
hdfs dfs -put weather_forecast_data.csv /user/hduser/weather_data/
hdfs dfs -ls /user/hduser/weather_data
```

### 3️⃣ Run the PySpark Script
```bash
spark-submit --master local[*] spark_weather_predictor.py
```

---

## 📈 Model Training Workflow

1. **Load Data** from HDFS  
2. **Clean Missing Values**  
3. **Convert Target Variable** (`Rain`) to numerical labels using `StringIndexer`  
4. **Assemble Features** into a vector with `VectorAssembler`  
5. **Split Data** into training (80%) and testing (20%) sets  
6. **Train Logistic Regression Model**  
7. **Evaluate Model**:
   - Accuracy
   - F1-Score
   - Confusion Matrix
   - ROC Curve & AUC
8. **Save Model** to both HDFS and local filesystem  
9. **Optimize Model** with 5-fold Cross-Validation  

---

## 📊 Results

**Base Model Performance:**
- **Accuracy:** 0.9376
- **F1-Score:** 0.9340
- **AUC:** 0.9710

**Confusion Matrix:**
| Label (Actual) | Prediction | Count |
|----------------|------------|-------|
| rain           | rain       | 25    |
| no rain        | rain       | 9     |
| rain           | no rain    | 19    |
| no rain        | no rain    | 396   |

**Feature Coefficients:**
- Temperature: -0.1539  
- Humidity: +0.0823  
- Wind Speed: -0.0021  
- Cloud Cover: +0.0488  
- Pressure: -0.0002  

Interpretation:
- Higher **humidity** and **cloud cover** increase the probability of rain.
- Higher **temperature** decreases the probability of rain.
- **Wind speed** and **pressure** have minimal impact.

---

## 📉 ROC Curve

The ROC curve shows an **AUC ≈ 0.9710**, indicating excellent model performance. The curve is close to the top-left corner, meaning the model achieves high TPR with low FPR.

---

## 💾 Model and Output Storage

- **Model saved to HDFS:**  
  `hdfs://localhost:9000/user/hduser/models/logistic_weather_model`  
- **Model saved locally:**  
  `./local_output/models/logistic_weather_model`
- **ROC Curve Image:**  
  - Local: `./local_output/plots/roc_curve.png`  
  - HDFS: `/user/hduser/plots/roc_curve.png`

---

## 🚀 Future Improvements

- Add more meteorological and geographical features.
- Handle class imbalance with oversampling/undersampling or class weighting.
- Compare with advanced models like **Random Forest** or **Gradient Boosted Trees**.
- Implement **Spark Streaming** for real-time weather prediction.
- Continuously retrain and monitor model performance with new data.

---

## 👨‍💻 Author
**Ali Salimi**  
📧 Contact: [alisalimi6205@yahoo.com]  
📂 GitHub: [als138](https://github.com/als138)

---
