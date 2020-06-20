import numpy as np

import pandas as pd
#from flasgger import Swagger
import streamlit as st 
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
import statistics as sts
import math
from sklearn.metrics import mean_squared_error,mean_absolute_error
#visualisation pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from keras.backend import manual_variable_initialization 





from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import LeakyReLU
import tensorflow
leaky_relu = keras.layers.LeakyReLU(alpha=0.1)




json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")
loaded_model.compile(loss='mean_squared_error',optimizer='adam')




st.title("Delhi pm2.5")



def main():
    
    st.title('Air Quality index ML APP')
    
    activities=['EDA','PLOT','MODEL']
    choice= st.sidebar.selectbox('Select activity',activities)
    st.write("""
    # Delhi pollution index
    Shown are the pm2.5 values!
    """)

    
    if choice=='EDA':
        st.subheader('Exploratory Data Analysis')
        data=st.file_uploader('upload_data', type=['csv','txt'])
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
        if st.checkbox('show shape'):
            st.write(df.shape)
        if st.checkbox('show columns'):
            all_columns= df.columns
            st.write(all_columns)
        if st.checkbox('mean'):
            all_columns= df.columns
            st.write(df[all_columns].mean())
        if st.checkbox('median'):
            all_columns= df.columns
            st.write(df[all_columns].median())
        if st.checkbox('mode'):
            all_columns= df.columns
            st.write(df[all_columns].mode())
        if st.checkbox('variance'):
            all_columns= df.columns
            st.write(np.var(df[all_columns]))
    
    elif choice=='PLOT':
        st.subheader('Data Visualisation')
        data=st.file_uploader('upload_data', type=['csv','txt'])
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())
        st.write("""
            # Data Visualisation
             """)
        st.line_chart(df.rename(columns={'date':'index'}).set_index('index'))
        df1=df['pm25_sea_ma']
        df=np.array(df1).reshape(-1,1)
        training_size=int(len(df)*0.70)
        test_size=len(df)-training_size
        train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]
        if st.checkbox('Train data plot'):
            st.write("""
            # Train data
            """)
            st.line_chart(pd.DataFrame(train_data))
        if st.checkbox('Test data plot'):
            st.write("""
            # Test data
             """)
            st.line_chart(pd.DataFrame(test_data))
        
        if st.checkbox('Histogram'):
            df2=pd.DataFrame(df1)
            all_colunms= df2.columns
            columns_to_plot=st.selectbox('select pm value', all_colunms)
            st.write(plt.hist(df2[columns_to_plot], color='g', bins=100))
            st.pyplot()
            
        if st.checkbox('barplot'):
            df2=pd.DataFrame(df1)
            all_colunms= df2.columns
            columns_to_plot=st.selectbox('open', all_colunms)
            st.write(plt.boxplot(df2[columns_to_plot]))
            st.pyplot()
   
    elif choice=='MODEL':
        html_temp = """<div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Forcasting ML App </h2>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)
        data=st.file_uploader('upload_data', type=['csv','txt'])
        if data is not None:
            df2=pd.read_csv(data)
            st.dataframe(df2.head())
        import numpy
        def create_dataset(dataset, time_step=100):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return numpy.array(dataX), numpy.array(dataY)
        scaler=MinMaxScaler(feature_range=(0,1))
        df1=scaler.fit_transform(np.array(df2).reshape(-1,1))
        
        
        
        def pre_process():
           scaler=MinMaxScaler(feature_range=(0,1))
           df1=scaler.fit_transform(np.array(df2).reshape(-1,1))
           time_step=100
           dataX, dataY=create_dataset(df1, time_step)
           dataX =dataX.reshape(dataX.shape[0],dataX.shape[1] , 1)

           return dataX,dataY
       
        def predict_note_authentication(df2):
           dataX,dataY= pre_process()
           prediction1=loaded_model.predict(dataX)
           prediction2=scaler.inverse_transform(prediction1)
           prediction= pd.DataFrame(prediction2).abs()
           print(prediction)
           print(dataY)
           return prediction
       
            


        
        st.subheader('Selection & Visualisation')    
        result=""
        if st.button("Predict"):
          result=predict_note_authentication(df2)
          st.success(st.table(result))
          st.write('Visualisation of predicted output')
          st.line_chart(pd.DataFrame(result))
        
          
        def RMSE(df2):
           dataX,dataY= pre_process()
           prediction1=loaded_model.predict(dataX)
           prediction2=scaler.inverse_transform(prediction1)
           prediction= pd.DataFrame(prediction2).abs()
           data1Y=scaler.inverse_transform(dataY.reshape(-1, 1))
           Root_mean_sqaure=math.sqrt(mean_squared_error(data1Y,prediction))
           Root_absolute_mean_sqaure=math.sqrt(mean_absolute_error(data1Y,prediction))
           print(Root_mean_sqaure)
           print(Root_absolute_mean_sqaure)
           return Root_mean_sqaure,Root_absolute_mean_sqaure
        
        st.subheader('Selection & Visualisation')    
        result1=""
        if st.checkbox("RMSE_Score"):
          result1=RMSE(df2)
          st.success('The output is {}'.format(result1))
         
        if st.button("About"):
          st.text("Lets LEarn")
          st.text("Built with Streamlit")
          
       
        
        

if __name__=='__main__':
    main()
    
    


         


