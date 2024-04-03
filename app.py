# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_resource
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    columns_to_keep = ['Age', 'MonthlyIncome', 'Department', 'DistanceFromHome', 
                       'Education', 'EducationField', 'JobRole', 'MaritalStatus']
    return data[columns_to_keep].copy()

data = load_data()

# Define function to encode categorical variables
def encode_categorical(data):
    categorical_cols = ['Department', 'Education', 'EducationField', 'JobRole', 'MaritalStatus']
    label_encoder = LabelEncoder()
    encoded_cols = {}
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])
        encoded_cols[col] = label_encoder
    return data, encoded_cols

# Define function to decode categorical variables
def decode_categorical(data):
    decode_map = {
        'Department': {0: 'Human Resources', 1: 'Research & Development', 2: 'Sales'},
        'Education': {0: 'Below College', 1: 'College', 2: 'Bachelor', 3: 'Master', 4: 'Doctor'},
        'EducationField': {0: 'Human Resources', 1: 'Life Sciences', 2: 'Marketing', 3: 'Medical', 4: 'Other', 5: 'Technical Degree'},
        'JobRole': {0: 'Healthcare Representative', 1: 'Human Resources', 2: 'Laboratory Technician', 3: 'Manager', 4: 'Manufacturing Director', 5: 'Research Director', 6: 'Research Scientist', 7: 'Sales Executive', 8: 'Sales Representative'},
        'MaritalStatus': {0: 'Divorced', 1: 'Married', 2: 'Single'}
    }
    
    for col in decode_map.keys():
        data[col] = data[col].map(decode_map[col])
    
    return data

# Sidebar for user input
st.sidebar.title("Employee Salary Prediction")
selected_radio = st.sidebar.radio("Select", ("Background", "Visualization", "Prediction"))

# Display project description and problem statement
st.title("Employee Salary Prediction")

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
    st.subheader("Project Title: Employee Salary Prediction")
    st.write("In this project, we aim to predict employee salary using linear regression.")
    st.write("")
    # Background Problem
    st.write("")
    st.subheader("**Background Problem:**")

    st.write("Key Challenges:")
    st.write("1. Ensuring fair compensation practices")
    st.write("2. Budgeting and resource allocation")
    st.write("3. Talent acquisition and retention")
    st.write("4. Performance management and career development")

# Project Objective
    st.write("")
    st.subheader("Objective:")


    st.write("The objective of this project is to:")
    st.write("1. Develop a predictive model to estimate employee salaries using linear regression.")
    st.write("2. Optimize recruitment processes by leveraging salary predictions for candidate selection and negotiation.")
    st.write("3. Ensure fair compensation practices by benchmarking salaries and identifying disparities.")
    st.write("4. Retain top talent by offering competitive compensation packages informed by the predictive model.")
    st.write("5. Support strategic decision-making through accurate salary forecasts and insights into salary-related trends.")



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
    st.write("Data Head:")
    st.write(data.head())

    user_input = {}
    for column in data.columns:
        if column != 'MonthlyIncome' and data[column].dtype == object:
            user_input[column] = st.selectbox(f"Select {column}", data[column].unique())
        elif column != 'MonthlyIncome' and data[column].dtype != object:
            user_input[column] = st.number_input(f"Enter {column}", value=0)

    st.write("Selected Input Features:")
    input_df = pd.DataFrame(user_input, index=[0])
    st.table(input_df)

    if st.button("Predict Salary"):
        # Decode categorical variables
        decoded_input_df = decode_categorical(input_df)

        # Encode categorical variables
        input_df_encoded, _ = encode_categorical(decoded_input_df)

        # Data preprocessing
        data.dropna(inplace=True)
        label_encoder = LabelEncoder()
        for column in data.columns:
            if data[column].dtype == object:
                data[column] = label_encoder.fit_transform(data[column])

        # Model training
        X = data.drop('MonthlyIncome', axis=1)
        y = data['MonthlyIncome']

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X, y)

        # Make prediction
        prediction = model.predict(input_df_encoded)
        st.write(f"Predicted Salary: ${prediction[0]:,.2f}")
