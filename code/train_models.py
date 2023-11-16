from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss
import pandas as pd
# assuming crossvalidation and bootstrap are custom modules
import crossvalidation
#import bootstrap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import numpy as np
#import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.externals import joblib
# Import necessary keras modules

#from keras.models import Sequential
#from keras.layers import Dense, Dropout
#from keras.scikit_learn import KerasClassifier
#from scikeras.wrappers import KerasClassifier
#from keras.utils import to_categorical

'''
def support_vector_classifier(df):
    
    # Data Preprocessing
    start_date = pd.to_datetime('2003-02-02')
    end_date = pd.to_datetime('2016-01-02')
    threshold = 0.7 
    
    df = df.drop(columns=['touch_lower', 'touch_upper'])
    df = df.dropna(how='all')
    df = df[60:]
    
    # Splitting data
    train_datasets, test_datasets, weights = crossvalidation.run_split_process(df)
    # train_datasets = bootstrap.sequential_bootstrap_with_rebalance(train_datasets)    
    
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'SpreadOC', 'SpreadLH']
    target_col = "label"
    
    all_predictions = []
    all_actuals = []
    all_preds = []
    n_components = 6
    scaler = StandardScaler()
    
    # Define a parameter grid for GridSearchCV
    
    param_grid = {
        'C': [ 50], 
        'gamma': [ 'auto'], 
        'kernel': ['linear']  
    }
    
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']} 

    # Training and Predicting for each split
    for train, test, weights in zip(train_datasets, test_datasets, weights):
        train = train_datasets
        test = test_datasets
        #weight = weights[-1] 
        
        
        
        X_train = train[feature_cols]
        y_train = train[target_col]
        X_test = test[feature_cols]
        y_test = test[target_col]

        # Standardize the data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Initialize GridSearchCV
        clf = SVC(probability=True, C=50)
        #grid_search = GridSearchCV(clf, param_grid,refit=True, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train)

        # Use the best estimator to predict
        #best_svm = grid_search.best_estimator_
        #print('best svm:',best_svm)
        probas = clf.predict_proba(X_test)

        #y_pred = (probas[:, 1] >= threshold).astype(int)

        max_proba_indices = np.argmax(probas, axis=1)

        predicted_classes = clf.classes_[max_proba_indices]
        y_pred = predicted_classes

        # Print and store results
        print('######################')
        print('probas:', probas)
        print(classification_report(y_test, y_pred, zero_division=1))
        print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
        predicted_probabilities = probas[:, 1]
        print('predicted_probs:', predicted_probabilities)
        print('y_test:', y_test)
        brier_score = brier_score_loss(y_test, predicted_probabilities)
        print('Brier Score:', brier_score)

        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predictions': y_pred
        })
        all_predictions.append(predictions_df)

        all_actuals.extend(y_test.tolist())
        all_preds.extend(y_pred.tolist())
        print('###########################')

        print('classes---> ',clf.classes_)


    # After processing all splits, compute overall metrics
    '''
   # joblib.dump(clf, 'models/EURUSD/random_forest_model_up_SPY.pkl')
   # joblib.dump(pca, 'models/EURUSD/pca_transformation_up_SPY.pkl')
    #joblib.dump(scaler, 'models/EURUSD/scaler_SPY.pkl')

#    file_input = "/mnt/volume_nyc1_02"
'''
    #joblib.dump(clf, f'{file_input}/models/EURUSD/support_vector_classifier_up_EURUSD.pkl')
    #joblib.dump(pca, f'{file_input}/models/EURUSD/pca_transformation_up_EURUSD.pkl')
    #joblib.dump(scaler, f'{file_input}/models/EURUSD/scaler_EURUSD.pkl')
    
    print(predictions_df)
    print("\nOverall Classification Report:")
    print(classification_report(all_actuals, all_preds, zero_division=1))
    print('Overall Confusion Matrix:', confusion_matrix(all_actuals, all_preds))

    
    # Combining all predictions and saving
    final_predictions_df = pd.concat(all_predictions)
    final_predictions_df.to_csv('predictions.csv', index=False)
'''

def support_vector_classifier(df):
    
    # Data Preprocessing
    start_date = pd.to_datetime('2003-02-02')
    end_date = pd.to_datetime('2016-01-02')
    
    df = df.drop(columns=['touch_lower', 'touch_upper'])
    df = df.dropna(how='all')
    df = df[60:]
    
    # Assuming crossvalidation is a custom module you've defined
    train_datasets, test_datasets, weights = crossvalidation.run_split_process(df)
    
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'SpreadOC', 'SpreadLH']
    target_col = "label"
    
    all_predictions = []
    all_actuals = []
    all_preds = []
    
    n_components = 6
    scaler = StandardScaler()
    
    # Define a parameter grid for GridSearchCV
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
                  'kernel': ['rbf']} 

    # Training and Predicting for each split
    for train_data, test_data, weight_data in zip(train_datasets, test_datasets, weights):
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Standardize the data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # Initialize the SVC model and GridSearchCV
        clf = SVC(probability=True, C=0.1,gamma=1, kernel='rbf')
        #clf = GridSearchCV(svc, param_grid, refit=True, verbose=3, n_jobs=-1)
        
        # Fit the model
        clf.fit(X_train_pca, y_train)

        # Get the best estimator
        #best_svc = clf.best_estimator_
        
        # Predictions
        #probas = clf.predict_proba(X_test_pca)
        #y_pred = clf.predict(X_test_pca)

        probas = clf.predict_proba(X_test_pca)

        #y_pred = (probas[:, 1] >= threshold).astype(int)

        
        max_proba_indices = np.argmax(probas, axis=1)
        predicted_classes = clf.classes_[max_proba_indices]
        y_pred = predicted_classes

        # Print and store results
        print('######################')
        print('probas:', probas)
        print(classification_report(y_test, y_pred, zero_division=1))
        print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
        #predicted_probabilities = probas[:, 2]
        #print('predicted_probs:', predicted_probabilities)
        #print('y_test:', y_test)
        print(f'Y_true:{y_test} Y_pred:{y_pred}' )

        comparison_df = pd.DataFrame({'Y_true': y_test, 'Y_pred': y_pred})

        print(comparison_df)

       # brier_score = brier_score_loss(y_test, predicted_probabilities)
       # print('Brier Score:', brier_score)

        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predictions': y_pred
        })
        all_predictions.append(predictions_df)

        all_actuals.extend(y_test.tolist())
        all_preds.extend(y_pred.tolist())
        print('###########################')
        print('classes---> ',clf.classes_)


    # After processing all splits, compute overall metrics
    '''
    joblib.dump(clf, 'models/EURUSD/random_forest_model_up_SPY.pkl')
    joblib.dump(pca, 'models/EURUSD/pca_transformation_up_SPY.pkl')
    joblib.dump(scaler, 'models/EURUSD/scaler_SPY.pkl')
    '''
    file_input = "/mnt/volume_nyc1_02"
    '''
    joblib.dump(clf, f'{file_input}/models/EURUSD/random_forest_classifier_up_EURUSD.pkl')
    joblib.dump(pca, f'{file_input}/models/EURUSD/pca_transformation_up_EURUSD.pkl')
    joblib.dump(scaler, f'{file_input}/models/EURUSD/scaler_EURUSD.pkl')
    '''
    print(predictions_df)
    print("\nOverall Classification Report:")
    #print(classification_report(all_actuals, all_preds, zero_division=1))
    print('Overall Confusion Matrix:', confusion_matrix(all_actuals, all_preds))

    
    # Combining all predictions and saving
    final_predictions_df = pd.concat(all_predictions)
    final_predictions_df.to_csv('predictions.csv', index=False)
  
    
    # You might want to return something from this function, like t
  

def random_forest_classifier(df):
    
    # Data Preprocessing
    start_date = pd.to_datetime('2003-02-02')
    end_date = pd.to_datetime('2016-01-02')
    threshold = 0.7 
    
    df = df.drop(columns=['touch_lower', 'touch_upper'])
    df = df.dropna(how='all')
    df = df[60:]
    
    # Splitting data
    train_datasets, test_datasets, weights = crossvalidation.run_split_process(df)
    # train_datasets = bootstrap.sequential_bootstrap_with_rebalance(train_datasets)    
    
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI', 'SpreadOC', 'SpreadLH', 'SMI']
    target_col = "label"
    
    all_predictions = []
    all_actuals = []
    all_preds = []
    n_components = 6
    scaler = StandardScaler()
    
    # Define a parameter grid for GridSearchCV
    '''
    param_grid = {
        'C': [ 50], 
        'gamma': [ 'auto'], 
        'kernel': ['linear']  
    }
    '''
    param_grid = {
        'n_estimators': [200, 500, 1000],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
   
    # Training and Predicting for each split
    for train_data, test_data, weight_data in zip(train_datasets, test_datasets, weights):
        #train = train_datasets
        #test = test_datasets
        #weight = weights[-1] 
        
        
        
        X_train = train_data[feature_cols]
        y_train = train_data[target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Standardize the data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Initialize GridSearchCV
        #clf = SVC(probability=True, C=50)
        clf =RandomForestClassifier( random_state=42, n_estimators=1000)

       # grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        #grid_search.fit(X_train, y_train, sample_weight=weight_data)

        #best_params = grid_search.best_params_
       # print(f"Best parameters found: {best_params}")

        #best_rf = grid_search.best_estimator_
        #grid_search = GridSearchCV(clf, param_grid,refit=True, verbose=3, n_jobs=-1)
        clf.fit(X_train, y_train, sample_weight=weight_data)

        # Use the best estimator to predict
        #best_svm = grid_search.best_estimator_
        #print('best svm:',best_svm)
        probas = clf.predict_proba(X_test)

        #y_pred = (probas[:, 1] >= threshold).astype(int)

        
        max_proba_indices = np.argmax(probas, axis=1)
        predicted_classes = clf.classes_[max_proba_indices]
        y_pred = predicted_classes

        


        # Print and store results
        print('######################')
        print('probas:', probas)
        print(classification_report(y_test, y_pred, zero_division=1))
        print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
        #predicted_probabilities = probas[:, 2]
        #print('predicted_probs:', predicted_probabilities)
        #print('y_test:', y_test)
        print(f'Y_true:{y_test} Y_pred:{y_pred}' )

        comparison_df = pd.DataFrame({'Y_true': y_test, 'Y_pred': y_pred})

        print(comparison_df)

       # brier_score = brier_score_loss(y_test, predicted_probabilities)
       # print('Brier Score:', brier_score)

        predictions_df = pd.DataFrame({
            'Actual': y_test,
            'Predictions': y_pred
        })
        all_predictions.append(predictions_df)

        all_actuals.extend(y_test.tolist())
        all_preds.extend(y_pred.tolist())
        print('###########################')
        print('classes---> ',clf.classes_)


    # After processing all splits, compute overall metrics
    
    joblib.dump(clf, 'models/EURUSD/random_forest_model_up_EURUSD_60.pkl')
    joblib.dump(pca, 'models/EURUSD/pca_transformation_up_EURUSD_60.pkl')
    joblib.dump(scaler, 'models/EURUSD/scaler_EURUSD.pkl')
    
    file_input = "/mnt/volume_nyc1_02"
   
    print(predictions_df)
    print("\nOverall Classification Report:")
    #print(classification_report(all_actuals, all_preds, zero_division=1))
    print('Overall Confusion Matrix:', confusion_matrix(all_actuals, all_preds))

    
    # Combining all predictions and saving
    final_predictions_df = pd.concat(all_predictions)
    final_predictions_df.to_csv('predictions.csv', index=False)
  


