import streamlit as st
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

st.title('Customer Segmentation Analysis')
data = pd.read_csv(r'H:\Mall_Customers.csv')
st.write('Head of The Data:')
st.write(data.head())
st.write('Tail of The Data')
st.write(data.tail())
st.write('Get Information About The Data')
buffer = io.StringIO()
data.info(buf=buffer)
s = buffer.getvalue()
st.text(s)
st.write('Get statistic Information About The Data')
st.write(data.describe())
st.write('Check For Nulls')
st.write(data.isnull().sum())
st.write('Check For Duplictes')
duplicated=data[data.duplicated]
st.write('The number Of duplictes :',len(duplicated))
st.markdown('<h3 style="font-weight:bold;">Histograme Visulaizations </h3>', unsafe_allow_html=True)
cols=['Age','Annual Income (k$)','Spending Score (1-100)']
for col in cols:
    sns.histplot(data=data,x=col,kde=True,element='step')
    plt.xlabel(col)
    plt.title(f"Histograme Of {col} Column")
    st.pyplot(plt)
    plt.clf()

st.markdown('<h3 style="font-weight:bold;">Correlation Heatmap </h3>', unsafe_allow_html=True)
correlation = data.select_dtypes('int','float').corr()
sns.heatmap(correlation,annot=True,cmap='Blues')
plt.title('Correlation Heatmap')
st.pyplot(plt)  
plt.clf() 

st.markdown('<h3 style="font-weight:bold;">Scatter Plot Of Data</h3>', unsafe_allow_html=True)
sns.scatterplot(data=data,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Gender')
plt.title("Annual Income vs. Spending Score by Gender")
st.pyplot(plt)
plt.clf()

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
data.drop('CustomerID',axis=1,inplace=True)
st.markdown('<h3 style="font-weight:bold;">Elbow Method For Optimal Number Of Clusters</h3>', unsafe_allow_html=True)
inertia = []
K_values=range(1,11)
for i in K_values:
    model = KMeans(n_clusters=i)
    model.fit_predict(data)
    inertia.append(model.inertia_)
    
plt.plot(K_values,inertia)  
plt.title("Elbow Diagram ")
plt.xlabel("Number Of Clusters")
plt.ylabel("Inertia")
st.pyplot(plt)
plt.clf()

st.write('The best number of clusters is 5')
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(data)
st.write(clusters)

st.markdown('<h3 style="font-weight:bold;">Scatter Plot Of Clusters</h3>', unsafe_allow_html=True)
sns.scatterplot(data=data,x='Annual Income (k$)',y='Spending Score (1-100)',hue=clusters)
plt.title("Customer Segmentation Using KMeans")
st.pyplot(plt)
plt.clf()


# User input section
st.markdown('<h2 style="font-weight:bold;">Input Features for Segmentation</h2>', unsafe_allow_html=True)
age = st.number_input('Age', min_value=0, max_value=100, value=30)
annual_income = st.number_input('Annual Income (k$)', min_value=0, value=50)
spending_score = st.number_input('Spending Score (1-100)', min_value=1, max_value=100, value=50)
gender=st.selectbox("Gender",['Male','Female'])
if st.button('Segment Customer'):
    input_data = pd.DataFrame({
        'Gender':[gender],
        'Age': [age],
        'Annual Income (k$)': [annual_income],
        'Spending Score (1-100)': [spending_score]
        
    })
    le = LabelEncoder()
    input_data['Gender'] = le.fit_transform(input_data['Gender'])    
    predicted_cluster = kmeans.predict(input_data)
    st.write(f'The predicted cluster for the input features is: {predicted_cluster[0]}')