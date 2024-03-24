# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_resource
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return data.copy()

data = load_data()

# Sidebar for user input
st.sidebar.title("Employee Attrition Prediction")
selected_radio = st.sidebar.radio("Select", ("Background", "Visualization", "Prediction"))

# Display project description and problem statement
st.title("Employee Attrition Prediction")

# Show relevant sections based on selected radio button
if selected_radio == "Background":
    st.subheader("Background")
    st.write("Regression is a supervised learning technique used to model the relationship between a dependent variable and one or more independent variables. It is widely used for prediction and forecasting tasks. There are various types of regression algorithms, including:")
    st.write("- Linear Regression")
    st.write("- Polynomial Regression")
    st.write("- Ridge Regression")
    st.write("- Lasso Regression")
    st.write("- Logistic Regression (for binary classification)")
    st.write("")
    st.write("Project Title: Employee Attrition Prediction")
    st.write("In this project, we aim to predict employee attrition using logistic regression.")
    st.write("")
    st.write("Background Problem")
    st.write("Employee attrition, or turnover, refers to the rate at which employees leave an organization. High employee attrition can be costly for businesses due to the expenses associated with hiring and training new employees, as well as the potential loss of productivity and morale within the organization.")
    st.write("")
    st.write("Objective")
    st.write("The objective of this project is to build a predictive model that can identify employees who are likely to leave the organization. By predicting employee attrition, organizations can take proactive measures to retain valuable employees and improve overall employee satisfaction and retention rates.")

elif selected_radio == "Visualization":
    st.subheader("Visualization")

    selected_viz = st.radio("Select Visualization", ("Histogram", "Count Plot", "Box Plot"))

    if selected_viz == "Histogram":
        selected_column = st.selectbox("Select column for histogram", data.columns)
        st.subheader(f"Histogram of {selected_column}")
        plt.hist(data[selected_column], bins=20, color='skyblue', edgecolor='black')
        st.pyplot(plt.gcf())

    elif selected_viz == "Count Plot":
        selected_column = st.selectbox("Select categorical column for count plot", data.select_dtypes(include='object').columns)
        st.subheader(f"Count Plot of {selected_column}")
        sns.countplot(data=data, x=selected_column, palette='viridis')
        st.pyplot(plt.gcf())

    elif selected_viz == "Box Plot":
        selected_column = st.selectbox("Select numerical column for box plot", data.select_dtypes(include=['int', 'float']).columns)
        st.subheader(f"Box Plot of {selected_column}")
        sns.boxplot(data=data, y=selected_column, color='lightblue')
        st.pyplot(plt.gcf())

else:  # Prediction
    st.subheader("Prediction")
    st.write("Employee Feature Input")
    user_input = {}
    for column in data.columns:
        if column != 'Attrition':
            if data[column].dtype == object:
                user_input[column] = st.selectbox(f"Select {column}", data[column].unique())
            else:
                user_input[column] = st.number_input(f"Enter {column}", value=0)
    if st.button("Predict"):
        # Data preprocessing
        data.dropna(inplace=True)
        label_encoder = LabelEncoder()
        for column in data.columns:
            if data[column].dtype == object:
                data[column] = label_encoder.fit_transform(data[column])

        # Model training
        X = data.drop('Attrition', axis=1)
        y = data['Attrition']
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        # Make prediction
        input_df = pd.DataFrame([user_input])
        for column in input_df.columns:
            if input_df[column].dtype == object:
                input_df[column] = label_encoder.transform(input_df[column])
        prediction = model.predict(input_df)
        if prediction[0] == 0:
            st.write("Employee is likely to stay.")
        else:
            st.write("Employee is likely to leave.")
