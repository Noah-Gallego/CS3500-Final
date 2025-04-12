# 🧠 CMPS 3500 - Crime Status Prediction using Deep Learning

Welcome to our final project for **CMPS 3500: Introduction to Deep Learning and Neural Networks**. This repository contains a full pipeline to clean real-world crime data from Los Angeles and train a Neural Network to predict the **status** of a crime incident.

---

## 📌 Project Description

You are working as a data scientist for a global finance company analyzing crime data in Los Angeles from **2020 to present**. The goal is to predict the **Status** of each crime using a Neural Network trained on rich data containing time, location, victim, and weapon details.

Crime status codes include:
- `IC` - Investigation Continues
- `AA` - Adult Arrest
- `AO` - Adult Other
- `JA` - Juvenile Arrest
- `JO` - Juvenile Other
- `CC` - Unknown

---

## 📊 The Dataset

Data source: `/home/fac/walter/public_html/courses/cs3500/03_2025_spring/proj/data/`

- `LA_Crime_Data_2023_to_Present_data.csv` – Training dataset (~80,000 rows, 28 columns)
- `LA_Crime_Data_2023_to_Present_test1.csv` – Reserved testing dataset

The dataset includes columns like:
- `DR_NO`, `DATE OCC`, `AREA NAME`, `Crm Cd`, `Vict Age`, `Weapon Desc`, `Status`, etc.

Note: Some data may include missing, duplicate, or malformed entries.

---

## ⚙️ Requirements

Before running the script, ensure you have the following:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

---

## 🧹 Data Cleaning Tasks

- Drop unnecessary columns
- Handle missing or malformed values
- Remove duplicates and outliers
- Encode categorical variables appropriately
- Convert data types as needed

---

## 🧠 Model Implementation

We trained a feedforward Neural Network using TensorFlow/Keras. The goal was to minimize **Root Mean Square Error (RMSE)** and maximize accuracy, precision, recall, and F1 score.

Metrics evaluated:
- Model Accuracy
- Model Precision
- Model Recall
- F1 Score
- Confusion Matrix

---

## 🖥️ Run Description (Text-Based Menu)

```
(1) Load training data
(2) Process (Clean) data
(3) Train NN
(4) Load testing data
(5) Generate Predictions
(6) Print Accuracy
(7) Quit
```

Example terminal output:
```
[14:03:21] Loading training data...
[14:03:25] Total Rows Read: 81234
[14:03:27] Performing Data Clean Up...
[14:03:32] Model Accuracy: 87.2%
...
```

---

## 🧪 Output Format

After running the model, the final predictions are saved to:
```
predictionClassProject1.csv
```

Columns:
- `DR_NO`
- `Status` (predicted)

---

## 🧾 Deliverables

Submit to Canvas:
- ✅ PDF Report (data pipeline, model architecture, error handling, RMSE)
- ✅ PDF Slide Deck (max 12 slides)
- ✅ Source Code: `ClassProjectGroup1.py`

---

## 📋 Grading Breakdown

| Component           | Weight |
|---------------------|--------|
| Progress Reports    | 10%    |
| Written Report      | 5%     |
| Source Code         | 10%    |
| Slide Deck          | 5%     |
| Presentation        | 10%    |
| Project Demo        | 45%    |
| Model Performance   | 15%    |

---

## ✍️ Contributors

```python
# course: cmps3500
# CLASS Project
# PYTHON IMPLEMENTATION: Crime Status Prediction
# date: 04/10/2025
# Student 1: Noah Gallego
# Student 2: John Smith
# Student 3: Jane Doe
# Student 4: Alice Johnson
# description: Implementation of data cleaning, deep learning model, and predictions for LA crime data.
```

---

## 📚 References

- [Pandas Library Overview](https://pandas.pydata.org/docs/)
- [30 Best Practices for Software Development](https://stackify.com/software-development-best-practices/)
- [What is Good Code?](https://medium.com/@fagnerbrack/what-is-good-code-4a4f56bdb500)
- [TensorFlow Docs](https://www.tensorflow.org/)
