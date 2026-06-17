import joblib
import pickle
import pandas as pd
import seaborn as sns
import plotly.express as px

import streamlit as st
from streamlit_option_menu import option_menu
st.set_page_config(page_title='Energy ForeCast',page_icon='⚡')

model = joblib.load('engery_model.pkl')
df = pd.read_csv('AEP_hourly.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.set_index('Datetime')

menu = option_menu(
    menu_title='',
    options=['Home', 'Prediction', 'Analysis'],
    icons=['house', 'activity', 'bar-chart'],
    orientation='horizontal'
)

if menu == 'Home':
    st.title('⚡ Energy Consumption Forecast')
    st.text('')
    st.subheader('About..')
    st.text('This app predicts energy consumption (in Megawatts) for the AEP region of the United states using historical hourly data from 2004 to 2018')
    
    st.text('')
    st.header('How it Works')
    st.text("""- Enter a date and hour
- Model predicts expected energy consumption
- Compare with historical patterns""")
    
    st.text('')
    st.subheader('Model')
    st.text("""- Algorithm: XGBoost Regressor
- R2 Score: 0.99
- Dataset: 121,000+ hourly records""")
elif menu == 'Prediction':
    st.title('⚡ Energy Forecast')
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date')
    with col2:
        end_date = st.date_input('End Date')

    if st.button('Forecast'):
        date_range = pd.date_range(start=start_date, end=end_date, freq='h')
        
        forecast_df = pd.DataFrame({'Datetime': date_range})
        forecast_df['hour'] = forecast_df['Datetime'].dt.hour
        forecast_df['dayofweak'] = forecast_df['Datetime'].dt.dayofweek
        forecast_df['month'] = forecast_df['Datetime'].dt.month
        forecast_df['year'] = forecast_df['Datetime'].dt.year
        forecast_df['quarter'] = forecast_df['Datetime'].dt.quarter
        forecast_df['dayofyear'] = forecast_df['Datetime'].dt.dayofyear
        forecast_df['weekofyear'] = forecast_df['Datetime'].dt.isocalendar().week.astype(int)
        forecast_df['lag_1'] = df['AEP_MW'].mean()
        forecast_df['lag_24'] = df['AEP_MW'].mean()
        forecast_df['lag_168'] = df['AEP_MW'].mean()

        features = ['hour','dayofweak','month','year','quarter','dayofyear','weekofyear','lag_1','lag_24','lag_168']
        forecast_df['Predicted_MW'] = model.predict(forecast_df[features])

        fig = px.line(forecast_df, x='Datetime', y='Predicted_MW', title='Hourly Energy Forecast')
        st.plotly_chart(fig)

        st.metric('Peak Consumption', f"{forecast_df['Predicted_MW'].max():.2f} MW")
        st.metric('Average Consumption', f"{forecast_df['Predicted_MW'].mean():.2f} MW")
        st.metric('Total Hours Forecasted', len(forecast_df))
elif menu == 'Analysis':
    Yearly = df.resample('YE').mean()
    fig1 = px.line(Yearly, y='AEP_MW', title = 'Yearly Average Energy Consumption')
    st.plotly_chart(fig1)

    monthly = df.resample('ME').mean()
    monthly['Month'] = monthly.index.strftime('%b %Y')
    fig2 = px.line(monthly, x='Month', y='AEP_MW', title='Monthly Average Energy Consumption')
    st.plotly_chart(fig2)

    df['hour'] = df.index.hour
    hourly = df.groupby('hour')['AEP_MW'].mean()
    fig3=px.bar(hourly, title='Average Consumption by Hour of Day')
    st.plotly_chart(fig3)