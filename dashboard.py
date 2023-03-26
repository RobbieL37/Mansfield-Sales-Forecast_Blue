import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

import plotly.express as px
import plotly.graph_objs as go

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse, meanabs, rmspe
from sklearn.metrics import mean_absolute_percentage_error

import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler
import pickle
from PIL import Image


header = st.container()
dataset = st.container()
features = st.container()
model = st.container()




with header: 
    ## Mansfield logo
    img = Image.open("mansfield_logo.jpg")
    st.image(img)
    
    st.title('Welcome To Our Mansfield Sales Forecast Dashboard!')
    st.text('In This Project We Will Forcast Mansfield Alto Sales For Three Months Ahead')
    
#------------------------------------------------------------------------------------------------------------------------------------

with dataset:
    st.header('Mansfield Alto Sales From 2018 To 2022')
    AltoSales = pd.read_csv('Scaled_sales_Mansfield_WH_SW_Alto_2022.csv').rename(columns={'Date':'Month'})
#     st.write(AltoSales.head())
    # Create a plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(AltoSales['Month'], AltoSales['Orders'], label='orders')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlabel('Month')
    ax.legend()
    st.pyplot(fig)
    
#-----------------------------------------------------------------------------------------------------------------------------------
     
    
with features:
    st.header('Macro Variables We Used')
    
    macro_var = pd.read_csv('macro_variables_clean.csv')
    macro_var
    
    standardized_data = pd.read_csv('standardized_data.csv')
    standardized_data['Month'] = pd.to_datetime(standardized_data['Month'], format='%m/%d/%Y').dt.date
    

#-------------------------------------------------------------------------------------------------------------------------------------


with model:
    st.header('Model Performance')
    st.subheader('These Are The Predicted Orders')
    
    st.sidebar.header("Choice of Model")

    # Create a slider for the prediction period (default is 3)
    prediction_period = st.sidebar.slider('Select the number of months to predict:', 1, 6, 3)

    # Create a dropdown menu for the user to select a model
    model_choice = st.sidebar.selectbox(
        'Select a model:',
        ('Holt-Winters', 'SARIMA')
    )

    # Use an if/else statement to load the selected model and make a prediction
    if model_choice == 'Holt-Winters':
        # Load the Holt-Winters model

        with open('holt_winters_model.pkl', 'rb') as h:
            model = pickle.load(h)

        # Load your data
        standardized_data = pd.read_csv("standardized_data.csv")
        standardized_data['Month'] = pd.to_datetime(standardized_data['Month'])
        

        # Make predictions with your Holt-Winters model using the selected prediction period
        start_date = standardized_data.index[0]
        end_date = standardized_data.index[-1] 
        predictions = model.predict(start=start_date, end=end_date)
        
        
           # 3 months ahead 
        print('\n\nNext Three Months Forecast:',)
        hw_order_pred = round(model.predict(60, 59+prediction_period),3)
        #st.write(hw_order_pred)
        
        
        # Create a new DataFrame with the predicted values
        pred_data = pd.DataFrame(hw_order_pred)
        pred_data.columns = ['Orders']
        pred_data.index = pd.date_range(start='2023-01-01', periods=len(pred_data), freq='MS')
        
        pred_data = pred_data.reset_index(drop=True)
        pred_data.index = pred_data.index + 60
        pred_data['Month'] = pd.date_range(start='2023-01-01', periods=len(pred_data), freq='MS')
        pred_data['Month'] = pd.to_datetime(pred_data['Month'], format='%m/%d/%Y').dt.date
        pred_data = pred_data[['Month'] + [col for col in pred_data.columns if col != 'Month']]

        st.write(pred_data)

#         # Plot the original data and predicted values
#         standardized_data['HW'] = predictions
#         fig = px.line(standardized_data, x='Month', y=['Orders', 'HW'], title='Lines of Holts Winters Forecasting and Original',
#         color_discrete_map={'Orders': 'light blue', 'HW': 'red'})
        
#         
#      # Display the prediction using st.write()
#         st.plotly_chart(fig)


        # Create traces for original and predicted data
        trace1 = go.Scatter(x=standardized_data['Month'], y=standardized_data['Orders'], mode='lines', name='Original Data')
        trace2 = go.Scatter(x=pred_data['Month'], y=pred_data['Orders'], mode='lines', name='Predicted Data', line=dict(color='red'))

        # Set layout options
        layout = go.Layout(
            title='Holts Winters Forecasting and Original Data',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Orders'),
            hovermode='closest'
        )

        # Calculate x-axis range
        min_date = min(standardized_data['Month'])
        max_date = max(pred_data['Month']) + pd.DateOffset(months=1)

        # Add vertical line to indicate end of original data
        shapes = [dict(
            type='line',
            xref='x',
            yref='y',
            x0=max_date,
            x1=max_date,
            y0=min(standardized_data['Orders']),
            y1=max(standardized_data['Orders']),
            line=dict(
                color='grey',
                width=1,
                dash='dash'
            )
        )]

        # Combine traces and layout options into a Figure object
        fig = go.Figure(data=[trace1, trace2], layout=layout)

        # Add vertical line to figure
        fig.add_vline(x='2022-12-02', line_width=3, line_dash="dash", line_color="grey")

        # Set x-axis range
        fig.update_xaxes(range=[min_date, max_date])

        # Display figure in Streamlit app
        st.plotly_chart(fig)


        
        
        # Calculate the RMSE of the predictions
        RMSE_HW = np.sqrt(mean_squared_error(standardized_data['Orders'], predictions))
        # Calculate the MAD of the predictions
        MAD_HW = mean_absolute_error(standardized_data['Orders'], predictions)
        # Calculate the MAPE of the predictions
        MAPE_HW = np.mean(np.abs((standardized_data['Orders'] - predictions) / standardized_data['Orders'])) * 100
        # Extract the AIC of the model
        AIC_HW = model.aic
        # Extract the BIC of the model
        BIC_HW = model.bic
       

        st.write("RMSE:", round(RMSE_HW, 4))
        st.write("MAD:", round(MAD_HW, 4))
        st.write("MAPE:", round(MAPE_HW, 4))
        st.write("AIC:", round(AIC_HW, 4))
        st.write("BIC:", round(BIC_HW, 4))
        
        pass
#--------------------------------------------------------------------------------------------------------------------'        
        
    else:
        # Load the SARIMA model
        with open('Sarima_model.pkl', 'rb') as s:
            model = pickle.load(s)
            
            
        # Load your data
        standardized_data = pd.read_csv("standardized_data.csv")
        standardized_data['Month'] = pd.to_datetime(standardized_data['Month'])
        
        # Make a prediction using the model
        
        start_date = standardized_data.index[0]
        end_date = standardized_data.index[-1] 
        predictions = model.predict(start=start_date, end=end_date)
        
        # 3 months ahead 
        print('\n\nNext Three Months Forecast:',)
        sarima_order_pred = round(model.predict(60, 59+prediction_period),3)
        #st.write(sarima_order_pred)
        
        
        
        # Display the prediction
        
         # Create a new DataFrame with the predicted values
        pred_data = pd.DataFrame(sarima_order_pred)
        pred_data.columns = ['Orders']
        pred_data.index = pd.date_range(start='2023-01-01', periods=len(pred_data), freq='MS')
        
        pred_data = pred_data.reset_index(drop=True)
        pred_data.index = pred_data.index + 60
        pred_data['Month'] = pd.date_range(start='2023-01-01', periods=len(pred_data), freq='MS')
        pred_data['Month'] = pd.to_datetime(pred_data['Month'], format='%m/%d/%Y').dt.date
        pred_data = pred_data[['Month'] + [col for col in pred_data.columns if col != 'Month']]

        st.write(pred_data)
        
        
        
#         standardized_data['SARIMA'] = predictions
#         fig = px.line(standardized_data, x='Month', y=['Orders', 'SARIMA'], title='Lines of SARIMA Forecasting and Original',
#         color_discrete_map={'Orders': 'light blue', 'SARIMA': 'orange'})
           
#         st.plotly_chart(fig)


  # Create traces for original and predicted data
        trace1 = go.Scatter(x=standardized_data['Month'], y=standardized_data['Orders'], mode='lines', name='Original Data')
        trace2 = go.Scatter(x=pred_data['Month'], y=pred_data['Orders'], mode='lines', name='Predicted Data',             line=dict(color='mediumseagreen'))

        # Set layout options
        layout = go.Layout(
            title='SARIMA Forecasting and Original Data',
            xaxis=dict(title='Month'),
            yaxis=dict(title='Orders'),
            hovermode='closest'
        )

        # Calculate x-axis range
        min_date = min(standardized_data['Month'])
        max_date = max(pred_data['Month']) + pd.DateOffset(months=1)

        # Add vertical line to indicate end of original data
        shapes = [dict(
            type='line',
            xref='x',
            yref='y',
            x0=max_date,
            x1=max_date,
            y0=min(standardized_data['Orders']),
            y1=max(standardized_data['Orders']),
            line=dict(
                color='grey',
                width=1,
                dash='dash'
            )
        )]

        # Combine traces and layout options into a Figure object
        fig = go.Figure(data=[trace1, trace2], layout=layout)

        # Add vertical line to figure
        fig.add_vline(x='2022-12-02', line_width=3, line_dash="dash", line_color="grey")

        # Set x-axis range
        fig.update_xaxes(range=[min_date, max_date])

        # Display figure in Streamlit app
        st.plotly_chart(fig)

        
        
        
        # Calculate the RMSE of the predictions
        RMSE_SARIMA = np.sqrt(mean_squared_error(standardized_data['Orders'], predictions))
        # Calculate the MAD of the predictions
        MAD_SARIMA = mean_absolute_error(standardized_data['Orders'], predictions)
        # Calculate the MAPE of the predictions
        MAPE_SARIMA = np.mean(np.abs((standardized_data['Orders'] - predictions) / standardized_data['Orders'])) * 100
        # Extract the AIC of the model
        AIC_SARIMA = model.aic
        # Extract the BIC of the model
        BIC_SARIMA = model.bic

        st.write('RMSE:', round(RMSE_SARIMA, 4))
        st.write('MAD:',  round(MAD_SARIMA, 4))
        st.write('MAPE:', round(MAPE_SARIMA, 4))
        st.write('AIC:', round(AIC_SARIMA, 4))
        st.write('BIC:', round(BIC_SARIMA, 4))
        
        pass


    
    