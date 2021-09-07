import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split

def preprocess_function(dataset_, params):
    
    # Create the target based on the return date
    dataset_['target'] = np.where((dataset_['returnDate']!='00000000') & (~dataset_['returnDate'].isna()), 1, 0)
    
    # Transform shipDate, orderDate, seasonYear to datetime
    dataset_['shipDate'] = pd.to_datetime(dataset_['shipDate'], format='%Y%m%d', errors = 'coerce')
    dataset_['orderDate'] = pd.to_datetime(dataset_['orderDate'], format='%Y%m%d', errors = 'coerce')
    dataset_['seasonYear'] = pd.to_datetime(dataset_['seasonYear'], errors = 'coerce')

    dataset_['order_date_year'] = dataset_['orderDate'].dt.year
    dataset_['season_year'] = dataset_['seasonYear'].dt.year    

    dataset_['collection'] = np.where((dataset_['season_year']>dataset_['order_date_year']) & (dataset_['season_year'] != 9999), 'New', 'Current')
    dataset_['collection'][dataset_['season_year']<dataset_['order_date_year']] = 'Old'    
    
    
    return dataset_

def split_preprocess_dataset(dataset_, params):
    
    # Split dataset to X_train, X_test, y_train, y_test in a ratio 80/20
    x = dataset_
    y = dataset_['target']
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=123)
    
    
    # Creation of mean target encodings
    materailSize_enc = X_train[['materailSize', 'target']].groupby(['materailSize']).mean()
    X_train = X_train.merge(materailSize_enc['target'].to_frame('materailSize_encoded'), how = 'left', on = ['materailSize'])
    X_test = X_test.merge(materailSize_enc['target'].to_frame('materailSize_encoded'), how = 'left', on = ['materailSize'])
    
    business_enc = X_train[['business', 'target']].groupby(['business']).mean()
    X_train = X_train.merge(business_enc['target'].to_frame('business_encoded'), how = 'left', on = ['business'])
    X_test = X_test.merge(business_enc['target'].to_frame('business_encoded'), how = 'left', on = ['business'])
    
    productGroup_enc = X_train[['productGroup', 'target']].groupby(['productGroup']).mean()
    X_train = X_train.merge(productGroup_enc['target'].to_frame('productGroup_encoded'), how = 'left', on = ['productGroup'])
    X_test = X_test.merge(productGroup_enc['target'].to_frame('productGroup_encoded'), how = 'left', on = ['productGroup'])
    
    division_enc = X_train[['division', 'target']].groupby(['division']).mean()
    X_train = X_train.merge(division_enc['target'].to_frame('division_encoded'), how = 'left', on = ['division'])
    X_test = X_test.merge(division_enc['target'].to_frame('division_encoded'), how = 'left', on = ['division'])
    
    mainColor_enc = X_train[['mainColor', 'target']].groupby(['mainColor']).mean()
    X_train = X_train.merge(mainColor_enc['target'].to_frame('mainColor_encoded'), how = 'left', on = ['mainColor'])
    X_test = X_test.merge(mainColor_enc['target'].to_frame('mainColor_encoded'), how = 'left', on = ['mainColor'])
    
    country_enc = X_train[['country', 'target']].groupby(['country']).mean()
    X_train = X_train.merge(country_enc['target'].to_frame('country_encoded'), how = 'left', on = ['country'])
    X_test = X_test.merge(country_enc['target'].to_frame('country_encoded'), how = 'left', on = ['country'])
    
    webBrowser_enc = X_train[['webBrowser', 'target']].groupby(['webBrowser']).mean()
    X_train = X_train.merge(webBrowser_enc['target'].to_frame('webBrowser_encoded'), how = 'left', on = ['webBrowser'])
    X_test = X_test.merge(webBrowser_enc['target'].to_frame('webBrowser_encoded'), how = 'left', on = ['webBrowser'])
    
    webCategory_enc = X_train[['webCategory', 'target']].groupby(['webCategory']).mean()
    X_train = X_train.merge(webCategory_enc['target'].to_frame('webCategory_encoded'), how = 'left', on = ['webCategory'])
    X_test = X_test.merge(webCategory_enc['target'].to_frame('webCategory_encoded'), how = 'left', on = ['webCategory'])
    
    webSource_enc = X_train[['webSource', 'target']].groupby(['webSource']).mean()
    X_train = X_train.merge(webSource_enc['target'].to_frame('webSource_encoded'), how = 'left', on = ['webSource'])
    X_test = X_test.merge(webSource_enc['target'].to_frame('webSource_encoded'), how = 'left', on = ['webSource'])
    
    subProuctGroup_enc = X_train[['subProuctGroup', 'target']].groupby(['subProuctGroup']).mean()
    X_train = X_train.merge(subProuctGroup_enc['target'].to_frame('subProuctGroup_encoded'), how = 'left', on = ['subProuctGroup'])
    X_test = X_test.merge(subProuctGroup_enc['target'].to_frame('subProuctGroup_encoded'), how = 'left', on = ['subProuctGroup'])
    
    # Drop target from X_train
    X_train = X_train.drop('target', 1)
    
    #Columns to drop
    drop_columns_train = list(set(X_train.columns.to_list()) - set(params['continuous_vars'] + params['categorical_vars']))
    
    # Drop columns
    X_train = X_train.drop(drop_columns_train, 1)
    
    # One hot encode the categorical features
    X_train = pd.get_dummies(X_train, columns = params['categorical_vars'])
    
    # Drop target from X_test
    X_test = X_test.drop('target', 1)    
    
    #Columns to drop
    drop_columns_test = list(set(X_test.columns.to_list()) - set(params['continuous_vars'] + params['categorical_vars']))
    
    # Drop columns
    X_test = X_test.drop(drop_columns_test, 1)
    
    # One hot encode the categorical features
    X_test = pd.get_dummies(X_test, columns = params['categorical_vars'])
    
    return X_train, y_train, X_test, y_test