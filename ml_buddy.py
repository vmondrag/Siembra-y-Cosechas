#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 05:46:35 2023

@author: renukhandelwal

purpose:
Boston housing
Linear regression, KNN Regressor, Random Forest regressor
Main Page: 
upload the file
view the data in data editor
save the edited data
clean the data

Analyze the data:
plot the correaltion, outliers, bar charts

Build and Evaluate te Model:
Select the model and its hyperparameters
display model metrics
enter a new data set and display the predicted value against the actula value of test dataset
"""

from xgboost import plot_importance
import xgboost as xgb
import streamlit as st
import pandas as pd
import sqlite3
import re
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def random_forest_feature_imp():
    if 'df' in st.session_state:
        df=st.session_state['df']
   
        X_train, y_train, X_test, y_test=create_test_train_data(df.columns[:-1])
            
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
       
        fig = plt.figure(figsize = (10, 5))     
        # creating the bar plot
        plt.bar(df.columns[:-1], model.feature_importances_, color ='maroon', 
        width = 0.4)
        # Print the feature importances
        st.pyplot(fig)
def random_forest_model():
    
    # Train a random forest model
    if 'df' in st.session_state:
        df=st.session_state['df']
        model = RandomForestRegressor()
        selected_imp_columns_rf = st.multiselect('Select columns for training Random Forest:', df.columns[:-1],  key='rf')
        if selected_imp_columns_rf !=[]:
            X_train, y_train, X_test, y_test=create_test_train_data(selected_imp_columns_rf)
            model.fit(X_train, y_train)
            y_pred= model.predict(X_test)
            st.metric(" Mean Squared Error",np.round(mean_squared_error(y_test, y_pred),2))
            st.metric(" R Square",np.round(r2_score(y_test, y_pred),2))
            fig1 = plt.figure(figsize = (10, 5))
            df_pred= pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})
            plt.plot(df_pred['actual'], label='Actual')
            plt.plot(df_pred['predictions'], label='predictions')
            plt.legend()
            st.pyplot(fig1)
def linear_regression_model():
    if 'df' in st.session_state:
        df=st.session_state['df']
        selected_imp_columns_lr = st.multiselect('Select columns for training Linear Regression:', df.columns[:-1], key='lr')
        if selected_imp_columns_lr !=[]:
            X_train, y_train, X_test, y_test=create_test_train_data(selected_imp_columns_lr)
            
            model =LinearRegression()
            
            model.fit(X_train, y_train) 
            
            
            y_pred=model.predict(X_test)
            st.metric(" Mean Squared Error",np.round(mean_squared_error(y_test, y_pred),2))
            st.metric(" R Square",np.round(r2_score(y_test, y_pred),2))
            fig1 = plt.figure(figsize = (10, 5))
            df_pred= pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})
            plt.plot(df_pred['actual'], label='Actual')
            plt.plot(df_pred['predictions'], label='predictions')
            plt.legend()
            st.pyplot(fig1)
def create_test_train_data(selected_imp_columns):
    if 'df' in st.session_state:
        df= st.session_state['df']
        df_len=len(df)
        y_col=df.columns[-1]
        #st.write("in test train",selected_imp_columns)
        #df_col=selected_imp_columns
        train_len=int(.7*df_len)
        X_train=df.loc[:train_len, selected_imp_columns].copy()
        y_train=df.loc[:train_len,y_col].copy()
        X_test=df.loc[train_len:, selected_imp_columns].copy()
        y_test=df.loc[train_len:,y_col].copy()
        #st.write(X_test, y_test)
        return X_train, y_train, X_test, y_test
def lime_feature_imp():
    if 'df' in st.session_state:
        
        df= st.session_state['df']
        X_train, y_train, X_test, y_test=create_test_train_data(df.columns[:-1])
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        explainer=LimeTabularExplainer(training_data=np.array(X_train),
                              mode="regression",
                              feature_names=list(X_train.columns),
                              training_labels=np.array(y_train),
                              random_state=0)
        exp=explainer.explain_instance(X_test.iloc[0], model.predict, num_features=len(X_train.columns))
        #fig = plt.figure(figsize = (10, 5))
        
        
        
        #plt.tight_layout()
        st.pyplot(exp.as_pyplot_figure())
def xgboost_feature_imp():
    if 'df' in st.session_state:
        df=st.session_state['df']
        
        X_train, y_train, X_test, y_test=create_test_train_data(df.columns[:-1])
        # Create the XGBoost model
        model = xgb.XGBRegressor(objective ='reg:linear', 
                                 eval_metric='rmsle', 
                                 seed = 123)
        
        # Train the model
        model.fit(X_train, y_train)
       
        st.pyplot(plot_importance(model).figure)
def xgboost_pred():
    #Allow the user to play with hyperparameters and then predict
    
    if 'df' in st.session_state:
        df=st.session_state['df']
        
        #Sliders for hyperparameter selection
        max_depth = st.slider('Select a value for max_depth',4, 7, 5)
        n_estimators = st.slider('Select a value for n_estimator',300, 700, 400)
        learning_rate = st.slider('Select a value for learning rate',0.01, 0.05, 0.01)
        # select columns based on feature importances
        selected_imp_columns_xgb = st.multiselect('Select columns for training:', df.columns[:-1],  key='xgb')
        if selected_imp_columns_xgb !=[]:
            model = xgb.XGBRegressor(objective ='reg:linear', 
                                     learning_rate = learning_rate,
                                     n_estimators  = n_estimators,
                                     max_depth     = max_depth,
                                     eval_metric='rmsle', 
                                     seed = 123)
            X_train, y_train, X_test, y_test=create_test_train_data(selected_imp_columns_xgb)
            # Train the model
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            st.metric(" Mean Squared Error",np.round(mean_squared_error(y_test, y_pred),2))
            st.metric(" R Square",np.round(r2_score(y_test, y_pred),2))
            fig1 = plt.figure(figsize = (10, 5))
            df_pred= pd.DataFrame(data={'predictions': y_pred, 'actual': y_test})
            #st.dataframe(df_pred)
            plt.plot(df_pred['actual'], label='Actual')
            plt.plot(df_pred['predictions'], label='predictions')
            plt.legend()
            st.pyplot(fig1)
def color_negative_red(val):
    color = 'red' if val is None else 'white'
    return f'color: {color}'
def get_book_names():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('./books.db')
    cur = conn.cursor()
    cur.execute("SELECT book_name FROM books_2023")
    result=cur.fetchall()
    cur.close()
    return [row[0] for row in result]

# Function to validate email
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    return False

def detect_outliers(df, field):
    Q1 = df[field].quantile(0.25)
    Q3 = df[field].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[field] < lower_bound) | (df[field] > upper_bound)]

def plot_data_with_outliers(df, field):
    outliers = detect_outliers(df, field)
    fig, ax = plt.subplots()
    ax.boxplot(df[field])
    ax.scatter(outliers.index, outliers[field], color='red', label='Outliers')
    ax.set_title(f"Outliers in {field}")
    ax.legend()
    return fig

def load_data_2_dataframe(file):
    '''
     Input Parameters
    ----------
    file : the file needs to be of type CSV

    Description
    -------
    Loads the data from the CSV file into a pandas dataframe
    
    Returns
    -------
    session_state containing dataframe df

    '''
    
    if 'df' not in st.session_state:
        df= pd.read_csv(file)
        st.session_state['df'] = df
        st.dataframe(df)
        
    else:
        st.dataframe(st.session_state['df'])
    
    return st.session_state['df']
def clean_data():
    if 'df' in st.session_state:
        df= st.session_state['df']
        count_rows_with_nan = df.isna().any(axis=1).sum()
        st.session_state['null_count']=count_rows_with_nan
        st.subheader("No. of rows containing null value",)
        # Display Null count
        st.metric("Null value", count_rows_with_nan, int(len(df)-count_rows_with_nan))    
        field = st.selectbox('Select Field to Analyze for Outliers', df.columns)
        # Plot and display in Streamlit
        if field:
            st.dataframe(detect_outliers(df, field))
#using diferent pages
# Define your page functions
def load_data():
    '''
    loads the selected file into a dataframe
    stores the selected file and dataframe in st.session_state
    '''
    # check if the dataframe df in st.session_state and is not blank
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df=load_data_2_dataframe(st.session_state['selected_file'])
        clean_data()
    #if the df does not exist in sesssion state then populate it       
    else:
        file = st.file_uploader("Upload a file", type=['csv'])
        if file is not None:
            st.session_state['selected_file'] = file
            df=load_data_2_dataframe(st.session_state['selected_file'])
            clean_data()           
    
        if 'null_count' in st.session_state:
            if st.session_state["null_count"] >0:
                null_action = st.radio(
                    'Select the action for handling Null Values',
                    ['Drop NA', 'Impute with 0', 'Impute with Mean', ])
                if null_action=='Drop NA':
                    #drop NA values in place
                    df= df.dropna(inplace=True)
                elif null_action=='Impute with 0':
                    # Fill missing values with a specific value
                    df = df.fillna(0, inplace=True)
                elif null_action=='Impute with 0':
                    # Fill missing values with mean of the column
                    df = df.fillna(df.mean(), inplace=True)
                    st.write(df)
    
   
def data_page():
    '''
     Display different graphs to understand the data
    '''
    if 'df' in st.session_state and st.session_state['df'] is  not None:
        df= st.session_state['df']
        graph_type= st.checkbox('Correlation graph')
        if graph_type:
            fig, ax = plt.subplots(figsize=(10,10))
            # Create the heatmap
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
            # Add a legend
            ax.legend()
            # Display the plot
            st.pyplot(fig)
        selected_columns = st.multiselect('Select columns to plot:', df.columns)
        if selected_columns is not None:
            # Create a figure
            df= st.session_state['df']
            fig, ax = plt.subplots()
            
            # Plot the data
            for column in selected_columns:
                ax.plot(df[column], label=column)
            
            # Set the axis labels
            ax.set_xlabel('X Axis')
            ax.set_ylabel('Y Axis')
            
            # Set the title
            ax.set_title('Chart')
            
            # Add a legend
            ax.legend()
            
            # Display the plot
            st.pyplot(fig)
    
def feature_imp():
    # check if df exists in session_state
    if 'df' in st.session_state:
        df=st.session_state['df']
        selected_columns=df.columns[:-1]
    #Feature Importance using Random Forest Regressor
    if st.checkbox("Random Forest Regressor Feature Importance", 
                   value=True):
        random_forest_model(selected_columns)
    # Feature Importnace using Lime   
    elif st.checkbox("lime", value=True):
        lime_feature_imp()

def model_selection():
    
    
       
    #st.write(selected_imp_columns[0])
    
    linear, RF, xgboost=st.tabs(['Linear Regression', 'Random Forest', 'XGBoost'])
     
    
    with linear:
        feature_imp, pred= st.columns(2) 
        with feature_imp:
            st.subheader("Lime Feature Importances")
            lime_feature_imp()
        with pred:
            st.subheader("Prediction with Linear Regression")
            linear_regression_model()
    with RF:
        feature_imp, pred= st.columns(2) 
        with feature_imp:
            st.subheader("Random Forest Feature Importance")
            random_forest_feature_imp()
        with pred:
            st.subheader("Prediction with Random Forest")
            random_forest_model()
    with xgboost:
        feature_imp, pred= st.columns(2) 
        with feature_imp:
            st.subheader("XGBoost Feature Importance")
            xgboost_feature_imp()
        with pred:
            st.subheader("Prediction with XGBoost")
            xgboost_pred()
        
        
        
        
# Title of the web app
st.title("Machine learning Buddy:high_brightness:")    
# Page Layout
st.sidebar.title("Navigation Options")

# Display different functionalities 
choice = st.sidebar.radio("Go to", ("Data View", "Data Analysis",  "Model Prediction"))

# Allow user to select a CSV file for Data Analysis
if choice == "Data View":
    load_data()
elif choice == "Data Analysis":
    data_page()
elif choice=="Model Prediction":
    model_selection()

