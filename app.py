import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import ipywidgets as wd
from IPython.display import display, clear_output
from ipywidgets import interactive, interactive

st.title('Sales Prediction-RealWare')

#Cleaning the csv before the analysis
df=pd.read_excel('RW_2016-2021_WS.xlsx')

df=df.iloc[5:,:] #Selecting only the relevent rows
df.reset_index(drop=True, inplace=True) #reset the index

df.columns = df.iloc[0] #Renaming the columns

df=df.iloc[1:,] #
df['Sales']=df['Sales'].astype(float).round(3)
df=df[df['Sales']>0]
df.reset_index(drop=True, inplace=True) #reset the index


if st.checkbox('Preview Dataset'):
    data=df
if st.button('Head'):
    st.write(data.head(10))
if st.button('Tail'):
    st.write(data.tail(10))

# Selecting the region nad category for the anlysis   
st.title('Selecting a Region for Analysis')

Select_Region=st.selectbox('Select a Region', ('All Region','EMEA', 'Americas','APAC'))
Select_Category=st.selectbox('Select a Category', ('All Category','Direct Customer', 'Indirect Customer','Partner'))

#Creating a function to get data based on selected region and category

def section(reg, catg):
    reg=str(reg)
    catg=str(catg)
    if (reg=='All Region') & (catg=='All Category'):
        sec=pd.DataFrame(df[df['Geography'].isnull()!=True])
        sec.reset_index(drop=True, inplace=True)
        return sec
    if (reg=='All Region') & (catg!='All Category'):
        sec=pd.DataFrame(df[df['Customer Category: Name']==catg])
        sec.reset_index(drop=True, inplace=True)
        return sec
    if (catg=='All Category') & (reg!='All Region'):
        sec=pd.DataFrame(df[df['Geography']==reg])
        sec.reset_index(drop=True, inplace=True)
        return sec
    else:
        sec=pd.DataFrame(df[(df['Geography']==reg) & (df['Customer Category: Name']==catg)])
        sec.reset_index(drop=True, inplace=True)
        return sec


df_new=section(Select_Region,Select_Category)
data=df_new

st.title('{} & {} Sales Data'.format(Select_Region,Select_Category))
st.write(data.head(10))

st.title('Visualizing the Sales data for the selection')

df2=df_new[['Customer', 'Sales', 'Week', 'Geography', 'Customer Category: Name']].groupby('Week').sum().reset_index()
df2[['Year','Week']]=df2['Week'].str.split('-', expand=True)
df2['DT'] = pd.to_datetime(df2.Week.astype(str)+df2.Year.astype(str).add('-1') ,format='%V%G-%u')
df2['DT'] = pd.to_datetime(df2['DT'])
df3=df2[['DT','Sales']]

#Visualizing the Weekly Sales data
fig=px.line(df3, x=df3['DT'], y=df3["Sales"])
fig.update_layout(showlegend=True, height=500, width=1000,title_text="Weekly Sales Data")


# Plot!
st.plotly_chart(fig, use_container_width=True)

#Visualizing the Monthly Sales data
y=df3.set_index('DT')
y=pd.DataFrame(y['Sales'].resample('M').sum())
y=y[y['Sales']>0]

fig1=px.line(y, x=y.index, y="Sales")
fig1.update_layout(showlegend=True, height=500, width=1000,title_text="Monthly Sales Data")


# Plot!
st.plotly_chart(fig1, use_container_width=True)

#Model
k=y.copy()
k=k[:-1]

#Auto Arima
import pmdarima as pm
model = pm.auto_arima(k['Sales'], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)


st.title('Forecasting')

# Forecast
n_periods = 15
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(k.index[-1], periods = n_periods, freq='M')


# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

fig3=make_subplots(rows=1,cols=1, shared_xaxes=True)

fig3.append_trace(go.Scatter(x=k.index, y=k.Sales, name='Actual Sales'), row=1, col=1)
fig3.append_trace(go.Scatter(x=fc_series.index,y=fc_series,name='Forecasted Sales'), row=1, col=1)

# Update axis properties
fig3.update_yaxes(title_text='Sales', row=1, col=1)
fig3.update_xaxes(title_text='Years', row=1, col=1)

fig3.update_layout(showlegend=True,height=500, width=1500,title_text="Sales vs Forecast")


# Plot!
st.plotly_chart(fig3, use_container_width=True)
