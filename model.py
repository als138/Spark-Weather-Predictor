import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ===========================
# 1. Define Output Paths
# ===========================
# 1.1 HDFS paths for model and plot
HDFS_MODEL_DIR = "hdfs://localhost:9000/user/hduser/models/logistic_weather_model"
HDFS_PLOT_DIR  = "/user/hduser/plots"  # Directory in HDFS to copy the plot

# 1.2 Local paths for model and plot
LOCAL_MODEL_DIR = "./local_output/models/logistic_weather_model"
LOCAL_PLOT_FILE = "./local_output/plots/roc_curve.png"

# Ensure local directories exist (create them if they don't)
os.makedirs(os.path.dirname(LOCAL_MODEL_DIR), exist_ok=True)
os.makedirs(os.path.dirname(LOCAL_PLOT_FILE), exist_ok=True)

# ===========================
# 2. Initialize SparkSession
# ===========================
spark = SparkSession.builder \
    .appName("WeatherClassification") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

# ===========================
# 3. Load data from HDFS
# ===========================
hdfs_path = "hdfs://localhost:9000/user/hduser/weather_data/weather_forecast_data.csv"
df = spark.read.csv(hdfs_path, header=True, inferSchema=True)

# Print schema and preview to verify data is loaded correctly
df.printSchema()
df.show(5)

# ======================================
# 4. Data Preparation for Modeling
# ======================================
# 4.1 Index the 'Rain' column to create labels
indexer = StringIndexer(inputCol="Rain", outputCol="label")
df = indexer.fit(df).transform(df)

# 4.2 Assemble feature columns into a single vector
feature_cols = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df = assembler.transform(df)

# 4.3 Split data into training and testing sets
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# ===============================
# 5. Train Logistic Regression Model
# ===============================
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.01)
model = lr.fit(train_data)

# ===============================
# 6. Evaluate Base Model
# ===============================
predictions = model.transform(test_data)

evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="label", predictionCol="prediction")
evaluator_f1  = MulticlassClassificationEvaluator(metricName="f1",       labelCol="label", predictionCol="prediction")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

print("=== Base Model Evaluation ===")
print("Accuracy: {:.4f}".format(accuracy))
print("F1-Score: {:.4f}".format(f1_score))
predictions.groupBy("label", "prediction").count().show()

print("Intercept:", model.intercept)
print("Coefficients:", model.coefficients)

# ======================================
# 7. Save Model to HDFS and Local
# ======================================

# 7.1 Save to HDFS
model.write().overwrite().save(HDFS_MODEL_DIR)
print(f"✅ Model saved to HDFS:\n  {HDFS_MODEL_DIR}")

# 7.2 Save to Local filesystem
model.write().overwrite().save(LOCAL_MODEL_DIR)
print(f"✅ Model saved locally:\n  {LOCAL_MODEL_DIR}")

# ========================================
# 8. Plot ROC Curve using Matplotlib
# ========================================
# 8.1 Extract probabilities and labels as Pandas DataFrame
prob_df = predictions.select("probability", "label") \
    .rdd.map(lambda row: (float(row.probability[1]), float(row.label))) \
    .toDF(["prob", "label"]) \
    .toPandas()

# 8.2 Calculate FPR, TPR, AUC
fpr, tpr, _ = roc_curve(prob_df["label"], prob_df["prob"])
roc_auc = auc(fpr, tpr)

# 8.3 Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# 8.4 Save the plot locally as PNG
plt.savefig(LOCAL_PLOT_FILE)
print(f"✅ ROC Curve saved locally:\n  {LOCAL_PLOT_FILE}")

# To display the plot in a GUI environment:
# plt.show()

# =========================================
# 9. Upload the Plot to HDFS
# =========================================
# Ensure destination directory exists in HDFS
os.system(f"hdfs dfs -mkdir -p {HDFS_PLOT_DIR}")

# Upload plot file from local to HDFS (overwrite if exists)
os.system(f"hdfs dfs -put -f {LOCAL_PLOT_FILE} {HDFS_PLOT_DIR}/roc_curve.png")
print(f"✅ ROC Curve saved to HDFS:\n  {HDFS_PLOT_DIR}/roc_curve.png")

# ===================================
# 10. Cross-Validation to Tune Model
# ===================================
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

cv = CrossValidator(estimator=lr,
                    estimatorParamMaps=paramGrid,
                    evaluator=BinaryClassificationEvaluator(labelCol="label"),
                    numFolds=5)

cv_model = cv.fit(train_data)
best_model = cv_model.bestModel

# Evaluate the tuned model on test data
final_predictions = best_model.transform(test_data)
final_accuracy = evaluator_acc.evaluate(final_predictions)

print("=== Tuned Model Evaluation (Cross-Validation) ===")
print(f"Tuned Accuracy: {final_accuracy:.4f}")

# ===============================
# End: Stop SparkSession
# ===============================
spark.stop()
