import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import streamlit as st 
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('diabetes.csv')

# as we can see there are field with 0 values which is not possible like BMI, Glucose, BloodPressure etc.
# we will replace those 0 values with mean of the respective columns
imputer = SimpleImputer(missing_values=0, strategy='mean')
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = imputer.fit_transform(data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']])
data.describe()

X=data.iloc[:,0:8]
y=data.iloc[:,8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf=RandomForestClassifier(max_depth=3)
rf.fit(X_train,y_train)

train_y_pred=rf.predict(X_train)
test_y_pred=rf.predict(X_test)

def app():
    img = Image.open(r"diabetes_-scaled.jpg")
    img = img.resize((900,900))
    st.image(img,caption="Diabetes Image",width=200)


    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # make a prediction based on the user input
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)
    prediction = rf.predict(reshaped_input_data)

    # display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if prediction == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # display some summary statistics about the dataset
    st.header('Dataset Summary')
    st.write(data.describe())

    #st.header('Distribution by Outcome')
    #st.write(diabetes_mean_df)

    # display the model accuracy
    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {accuracy_score(y_train,train_y_pred)}')
    st.write(f'Test set accuracy: {accuracy_score(y_test,test_y_pred)}')

if __name__ == '__main__':
    app()