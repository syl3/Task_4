import streamlit as st
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
import io
import base64

model_1 =lgb.Booster(model_file= 'model_10_0.txt')
model_2 =lgb.Booster(model_file= 'model_10_1.txt')
model_3 =lgb.Booster(model_file= 'model_10_2.txt')
model_4 =lgb.Booster(model_file= 'model_10_3.txt')
model_5 =lgb.Booster(model_file= 'model_10_4.txt')
datasetA = pd.read_csv('data_sample.csv')

def make_prediction(dataframe):
	if(dataframe.columns.tolist()!=datasetA.columns.tolist()):
		st.write('Please make sure your column names are the same as example dataset.')
	else:
		prediction_=pd.DataFrame((model_1.predict(dataframe.drop('ID',axis=1))+model_2.predict(dataframe.drop('ID',axis=1))+model_3.predict(dataframe.drop('ID',axis=1))+model_4.predict(dataframe.drop('ID',axis=1))+model_5.predict(dataframe.drop('ID',axis=1)))/5)
		prediction_.rename(columns={0:'Predicted_Probability'},inplace=True)
		prediction_['ID']=dataframe.ID
		prediction_=prediction_[['ID','Predicted_Probability']]
		prediction_['Prediction']=(prediction_.Predicted_Probability>0.5)*1

	return prediction_
		


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="Prediction.csv">Download csv file</a>'



def main():
	st.title('Beyond Prediction System')
	st.markdown('Wellcome to the Beyond Prediction System! The system is built by LightGBM and the performance is in the following:  \n  \nAccuracy = 0.9142  \nPrecision = 0.9052  \nRecall = 0.7061')
	st.markdown('Example dataset')
	st.dataframe(datasetA)

	uploaded_file = st.file_uploader("Choose a CSV file")

	if uploaded_file is not None:
		dataframe = pd.read_csv(uploaded_file)

		st.markdown('This is your uploaded file.')
		st.write(dataframe)
		
		st.markdown('**Congratulation!** This is the result.')
		prediction=make_prediction(dataframe)

		st.write(prediction)
		st.markdown(get_table_download_link(prediction), unsafe_allow_html=True)


if __name__=='__main__':
	main()



