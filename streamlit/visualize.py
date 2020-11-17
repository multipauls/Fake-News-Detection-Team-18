from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import streamlit as st
import pandas as pd
# import numpy as np
# import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go

from joblib import dump, load

# set title
st.title("Fake News Detection")

# load data
test_data = ".data/test_data.joblib"
model = ".data/tfidf_logistic_regression_model.joblib"
vectorizer = ".data/tf_idf_vectorizer.joblib"

svm_model = load(model)
test = load(test_data)
tf_idf = load(vectorizer)


print(svm_model)

x_test = test['data']

y_test = test['labels']
for j in y_test[:10]:
    print(j)

df = pd.DataFrame(list(zip(x_test, y_test)))

if st.checkbox('Show dataframe'): 
     st.write(df)


# x_train = vectorizer.fit_transform(x_test)
article = "Hello world. This is a sample article with a sample source"
sample_label = 0

article = tf_idf.transform([article])
print(article)

if st.checkbox('Predict on sample text'):
    pred = svm_model.predict(article)
    st.write(pred)

if st.checkbox('Predict on custom text'):
    user_input = st.text_area("Enter text", "This is sample text.")
    text = tf_idf.transform([user_input])
    pred = svm_model.predict(text)
    prediction = pred[0]
    if prediction == 0:
        st.write("reliable")
    elif prediction == 1:
        st.write("mixed")
    elif prediction == 2:
        st.write("unreliable")
    elif prediction == 3:
        st.write("cannot tell")
