from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import crossvalidation
import boostrap
from sklearn.metrics import brier_score_loss

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib





def make_predictions_up(symbol, new_data: pd.DataFrame) -> pd.Series:
    """
    Makes predictions on new data using a pre-trained Random Forest model and PCA transformation.
    
    Args:
    - new_data (pd.DataFrame): The new data for which predictions are to be made.

    Returns:
    - pd.Series: Predictions for the new data.
    """

    threshold = 0.7 

    # Load the trained model, PCA, and scaler
    #clf = joblib.load(f'models/{symbol}/random_forest_model_up_{symbol}.pkl')
    #pca = joblib.load(f'models/{symbol}/pca_transformation_up_{symbol}.pkl')
    #scaler = joblib.load(f'models/{symbol}/scaler_{symbol}.pkl')
    file_input = "/mnt/volume_nyc1_02"

    clf =joblib.load( f'{file_input}/models/SPY/support_vector_classifier_up_SPY.pkl')
    pca = joblib.load( f'{file_input}/models/SPY/pca_transformation_up_SPY.pkl')
    scaler = joblib.load( f'{file_input}/models/SPY/scaler_SPY.pkl')
    
    # Prepare the new data
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                    'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI','SpreadOC','SpreadLH']
    X_new = new_data[feature_cols]
    
    # Standardize the new data using the saved scaler
    X_new = scaler.transform(X_new)

    # Apply PCA transformation using the saved PCA
    X_new = pca.transform(X_new)
    
    # Make predictions using the loaded model
    predictions = clf.predict(X_new)
    probas = clf.predict_proba(X_new)

    predictions = (probas[:, 0] >= threshold).astype(int)
    
    return predictions, probas


def make_predictions_dwn(symbol,new_data: pd.DataFrame) -> pd.Series:
    """
    Makes predictions on new data using a pre-trained Random Forest model and PCA transformation.
    
    Args:
    - new_data (pd.DataFrame): The new data for which predictions are to be made.

    Returns:
    - pd.Series: Predictions for the new data.

    """
    file_input = "/mnt/volume_nyc1_02"
    clf =joblib.load( f'{file_input}/models/SPY/support_vector_classifier_dwn_SPY.pkl')
    pca = joblib.load( f'{file_input}/models/SPY/pca_transformation_dwn_SPY.pkl')
    scaler = joblib.load( f'{file_input}/models/SPY/scaler_SPY.pkl')

    # Load the trained model, PCA, and scaler
    #clf = joblib.load(f'models/{symbol}/random_forest_model_dwn_{symbol}.pkl')
    #pca = joblib.load(f'models/{symbol}/pca_transformation_dwn_{symbol}.pkl')
    #scaler = joblib.load(f'models/{symbol}/scaler_{symbol}.pkl')
    
    # Prepare the new data
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                    'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI','SpreadOC','SpreadLH']
    X_new = new_data[feature_cols]
    
    # Standardize the new data using the saved scaler
    X_new = scaler.transform(X_new)

    # Apply PCA transformation using the saved PCA
    X_new = pca.transform(X_new)
    
    # Make predictions using the loaded model
    predictions = clf.predict(X_new)
    probas = clf.predict_proba(X_new)
    
    return predictions, probas






'''

def random_forest_classifier(df):
    # Data Preprocessing
    start_date = pd.to_datetime('2003-02-02')
    end_date = pd.to_datetime('2016-01-02')

    threshold = 0.70 
    
    df = df.drop(columns=['touch_lower', 'touch_upper'])
    df.dropna(inplace=True)

    # Splitting data
    train_datasets, test_datasets, weights = crossvalidation.run_split_process(df)


    

    #train_datasets = boostrap.sequential_bootstrap_with_rebalance(train_datasets)    
    
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                    'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI','SpreadOC','SpreadLH']
    target_col = "label"
    
    all_predictions = []
    all_actuals = []
    all_preds = []
    n_components = 6
    scaler = StandardScaler()

    # Training and Predicting for each split
    for train, test, weights in zip(train_datasets, test_datasets, weights):

        
        X_train = train[feature_cols]
        y_train = train[target_col]

        X_test = test[feature_cols]
        y_test = test[target_col]

           # Standardize the data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)  # IMPORTANT: use transform, not fit_transform

        # Apply PCA
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)  # Again, just transform

        
        
        
        
        
        
        # Initialize and train the RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=1000, bootstrap=False, random_state=55, 
                                     class_weight='balanced_subsample', criterion='log_loss')
        clf.fit(X_train, y_train, sample_weight=weights)

       

        #print('ypreds:',y_pred)
        #print('ytest:',y_test)
        probas = clf.predict_proba(X_test)

        y_pred = (probas[:, 1] >= threshold).astype(int)
        
        print('######################')
        print('probas:',probas)
        print(classification_report(y_test, y_pred, zero_division=1))
        print('Confusion Matrix:', confusion_matrix(y_test, y_pred))

        predicted_probabilities = probas[:, 1]  # Assuming positive class is the second one
        
        print('predicted_probs:',predicted_probabilities)
        print('y_test:',y_test)
        #brier_score = brier_score_loss(y_test, predicted_probabilities)
        #print('Brier Score:', brier_score)


        # Storing feature importances
        importances = clf.feature_importances_
        for feature, importance in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")

        # Storing predictions for each split
        predictions_df = pd.DataFrame({
           # 'Date': X_test.index,
            'Actual': y_test,
            'Predictions': y_pred
           # 'Brier Score': brier_score
        })
        all_predictions.append(predictions_df)

        # Store actuals and predictions for computing overall metrics
        all_actuals.extend(y_test.tolist())
        all_preds.extend(y_pred.tolist())
        print('###########################')


    # After processing all splits, compute overall metrics
    joblib.dump(clf, 'random_forest_model_up.pkl')
    joblib.dump(pca, 'pca_transformation_up.pkl')

    print(predictions_df)
    print("\nOverall Classification Report:")
    print(classification_report(all_actuals, all_preds, zero_division=1))
    print('Overall Confusion Matrix:', confusion_matrix(all_actuals, all_preds))

    
    # Combining all predictions and saving
    final_predictions_df = pd.concat(all_predictions)
    final_predictions_df.to_csv('predictions.csv', index=False)
  
'''


   
'''
def random_forest_regressor(df):

    start_date = pd.to_datetime('2003-02-02')
    end_date = pd.to_datetime('2016-01-02')
    
    df = df.drop(columns=['touch_lower', 'touch_upper'])
    df.dropna(inplace=True)
    
    train = df[df.index <= end_date]
    test = df[df.index > end_date]
    
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                    'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI']
    
    X_train = train[feature_cols]
    y_train = train['label']
    
    X_test = test[feature_cols]
    y_test = test['label']
    
    # Using RandomForestRegressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    
    # Predicting regression scores instead of classification
    y_pred = reg.predict(X_test)
    
    # Here, you'd evaluate using regression metrics instead of classification metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Feature importances
    importances = reg.feature_importances_
    
    for feature, importance in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")


def adaboost_classifier(df):
    
    start_date = pd.to_datetime('2003-02-02')
    end_date = pd.to_datetime('2016-01-02')
    
    df = df.drop(columns=['touch_lower', 'touch_upper'])
    df.dropna(inplace=True)
    
    train = df[df.index <= end_date]
    test = df[df.index > end_date]
    
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                    'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI']
    target_col = "label"
    
    X_train = train[feature_cols]
    y_train = train['label']
    
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    # Initialize AdaBoost with a base estimator. Here, we'll use a DecisionTree.
    # You can adjust the n_estimators (number of trees) and learning_rate as needed.
      # Initialize AdaBoost with a base estimator. Here, we'll use a DecisionTree.
    base_est = DecisionTreeClassifier(max_depth=1)
    ada_clf = AdaBoostClassifier(base_estimator=base_est, random_state=42)
    
    # Define parameter grid for AdaBoost
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(ada_clf, param_grid, cv=5, scoring='precision')  # if you want to focus on F1-score
    grid_search.fit(X_train, y_train)
    
    # Get the best model from grid search
    best_ada_clf = grid_search.best_estimator_

    y_pred = best_ada_clf.predict(X_test)

    print("Best AdaBoost parameters:", grid_search.best_params_)
    print(classification_report(y_test, y_pred, zero_division=1))
    print('Confusion matrix:', confusion_matrix(y_test, y_pred))
    
    predictions_df = pd.DataFrame({
        'Date': X_test.index,
        'Actual': y_test,
        'Predictions': y_pred
    })
    
    # Save to CSV
    predictions_df.to_csv('adaboost_predictions.csv', index=False)
    
  
def random_forest_ts(df):
    tscv = TimeSeriesSplit(n_splits=5)
    
    df = df.drop(columns=['touch_lower', 'touch_upper'])
    df.dropna(inplace=True)
    
    feature_cols = ['Daily_Returns', 'Middle_Band', 'Upper_Band', 'Lower_Band',
                    'Log_Returns', 'MACD', 'Signal_Line_MACD', 'RSI']
    
    #feature_cols = ['Daily_Returns', 'MACD']
    target_col = "label"
    
    X = df[feature_cols]
    y = df[target_col]
    clf = RandomForestClassifier()

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        bootstrapped_sample = sequential_bootstrap_with_rebalance(pd.concat([X_train, y_train], axis=1), sample_size=len(X_train), window_size=20)
        X_train_bootstrapped = bootstrapped_sample.drop(columns='label')
        y_train_bootstrapped = bootstrapped_sample['label']

        clf = RandomForestClassifier(n_estimators=100, random_state=70)



    
    # Define parameter grid for AdaBoost
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.5, 1]
        }
       # base_est = DecisionTreeClassifier(max_depth=1)
       # ada_clf = AdaBoostClassifier(base_estimator=base_est, random_state=42)
    # Grid search with cross-validation
       # grid_search = GridSearchCV(ada_clf, param_grid, cv=5, scoring='f1')  # if you want to focus on F1-score
        #grid_search.fit(X_train, y_train)

        clf.fit(X_train_bootstrapped, y_train_bootstrapped)
        y_pred = clf.predict(X_test)
        
        print(classification_report(y_test, y_pred, zero_division=1))
        print('confusion_matrix:', confusion_matrix(y_test, y_pred))
        
        #importances = grid_search.feature_importances_
        #for feature, importance in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        #    print(f"{feature}: {importance:.4f}")
        
        predictions_df = pd.DataFrame({
            'Date': X_test.index,
            'Actual': y_test,
            'Predictions': y_pred
        })
        print(f"OOB Accuracy: {clf.oob_score:.4f}")
        # Save to CSV
        predictions_df.to_csv('predictions.csv', index=False)
    
'''     
    