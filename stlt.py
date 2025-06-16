import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Data Splitter Class
class DataSplitter:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column

    def split(self, test_size=0.2, random_state=42):
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

# Model Trainer Class with Save Method
class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = None

    def train_svm(self):
        if self.X_train.empty or self.y_train.empty:
            raise ValueError("Training data is empty. Please check preprocessing.")
        self.model = SVR()
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return mse, r2

    def save_model(self, filename="student_model.pkl"):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)
        return filename

# ----------- Streamlit App Starts Here -------------

st.title("🎓 Student Performance Prediction")

uploaded_file = st.file_uploader("Upload your CSV file (e.g., student-mat.csv)", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df_raw.head())

    target_col = st.selectbox("🎯 Select the target column", df_raw.columns)

    # Encode non-numeric columns
    df_encoded = pd.get_dummies(df_raw)

    # Check if target still exists after encoding
    if target_col not in df_encoded.columns:
        st.error(f"❌ Target column '{target_col}' disappeared during encoding. Please select a numeric target.")
        st.stop()

    df = df_encoded

    # Input filename for saving
    pickle_filename = st.text_input("💾 Enter Pickle filename", value="student_model.pkl")

    if st.button("🚀 Split, Train & Save Model"):
        try:
            splitter = DataSplitter(df, target_col)
            X_train, X_test, y_train, y_test = splitter.split()

            st.success("✅ Data split successfully!")
            st.write("📊 Training Data Shape:", X_train.shape)
            st.write("📊 Testing Data Shape:", X_test.shape)

            trainer = ModelTrainer(X_train, y_train)
            model = trainer.train_svm()

            mse, r2 = trainer.evaluate(X_test, y_test)
            st.subheader("📈 Model Evaluation")
            st.write(f"🔹 Mean Squared Error (MSE): `{mse:.2f}`")
            st.write(f"🔹 R² Score: `{r2:.2f}`")

            # Save model
            filename = trainer.save_model(pickle_filename)
            st.success(f"📦 Model saved as `{filename}`")

        except Exception as e:
            st.error(f"⚠️ Error during model training or saving: {e}")
