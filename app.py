import streamlit as st
import pickle 
import numpy as np
pipe1 = pickle.load(open('pipe1.pkl','rb'))
df1 = pickle.load(open('df1.pkl','rb'))
st.title("Laptop predictor")

#brand
company = st.selectbox('Brand',df1['Company'].unique())

#type
type = st.selectbox('Type',df1['TypeName'].unique())

#Ram
ram= st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

#weight
weight = st.number_input('Weight of the laptop')

#TouchScreen
touchscreen = st.selectbox('Touchscreen',['NO','Yes'])

#IPS
ips = st.selectbox('IPS',['No','Yes'])

#Screensize
screen_size = st.number_input('Screen Size')

#resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df1['Cpu brand'].unique())

#harddrive
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

#ssd
ssd = st.selectbox('SSD(in GB)',[0,128,256,512,1024])

gpu = st.selectbox('GPU',df1['Gpu brand'].unique())

os = st.selectbox('OS',df1['os'].unique())

if st.button('Predict Price'):
    #pass
    ppi=None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'Yes':
        ips=1
    else:
        ips=0
    
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size

    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    st.title(int(np.exp(pipe1.predict(query)[0])))