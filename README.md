# 🎓 Student Performance Prediction Using SVM

This project analyzes and predicts student performance using machine learning (Support Vector Machines). The pipeline is built using Object-Oriented Programming (OOP) concepts in Python.

## 📁 Dataset

We use the `student-mat.csv` dataset (from UCI Machine Learning Repository) which contains student-related data such as grades (G1, G2, G3), parental education, study time, etc.

## 📊 Workflow

### 1. Data Collection
- Load CSV using pandas with `sep=';'`.
- Show shape, column names, and preview data.

### 2. Data Understanding
- Check data types, null values, unique values.
- Generate summary statistics.

### 3. Data Preprocessing
- Handle missing values using mean.
- Normalize specific numeric columns using Min-Max scaling.

### 4. Data Analysis
#### Univariate Analysis
- Histograms, boxplots, violin plots.
#### Bivariate Analysis
- Scatter plots and correlation heatmaps.

### 5. Data Splitting
- Split dataset into training and testing sets using sklearn.

### 6. Model Training
- Use `SVR (Support Vector Regression)` from sklearn.
- Evaluate model using Mean Squared Error and R² score.

### 7. Model Storage
- Save the trained model using `pickle`.

## 🔧 Technologies Used
- Python
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Pickle

## 📂 How to Run
1. Ensure `student-mat.csv` is present in the same directory.
2. Run the Python file containing all class definitions and method calls.
3. The model will be saved as `student_model.pkl`.

✨ Project by: [Jahanzaib Zafar]
