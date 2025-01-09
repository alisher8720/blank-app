import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Model Selection and Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_auc_score, mean_squared_error, r2_score

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier

# Imbalanced Data Handling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek

# Other
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from scipy.stats import randint
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction import FeatureHasher
from scipy.sparse import csr_matrix


#EDA

df = pd.read_csv(r"C:\Users\hp\Downloads\Crime_Data_from_2020_to_Present.csv") #import dataset Crime_Data_from_2020_to_Present.csv
df.info()
df.isnull().sum() #assesstment of empty cells
#we do not know/or we do not whant  some columns be presented in the dataset although some of them are to certine extend clear what they about
df = df.drop(columns = ['DR_NO', 'Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'AREA', 'Rpt Dist No', 'Part 1-2', 'Crm Cd', 'Mocodes', 'Premis Cd', 'Weapon Used Cd', 'Status'])
# Correctly convert DATE OCC and TIME OCC
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p') # Corrected format
df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
df['TIME OCC'] = pd.to_datetime(df['TIME OCC'], format='%H%M').dt.time

# Combine DATE OCC and TIME OCC
df['OCCURRED_DATETIME'] = df.apply(lambda row: pd.Timestamp.combine(row['DATE OCC'], row['TIME OCC']), axis=1)


#PROPROCESSING 
# Extract time-based features (same as before)
df['Hour'] = df['OCCURRED_DATETIME'].dt.hour
df['DayOfWeek'] = df['OCCURRED_DATETIME'].dt.day_name()
df['IsWeekend'] = df['DayOfWeek'].isin(['Saturday', 'Sunday']).astype(int)

# Time of Day (same as before)
def time_of_day(hour):
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 22:
        return 'Evening'
    else:
        return 'Night'

df['TimeOfDay'] = df['Hour'].apply(time_of_day)

# IsNight (same as before)
df['IsNight'] = ((df['Hour'] >= 22) | (df['Hour'] < 6)).astype(int)

# Print the first few rows with the new features
print(df[['DATE OCC', 'TIME OCC', 'OCCURRED_DATETIME', 'Hour', 'DayOfWeek', 'IsWeekend', 'TimeOfDay', 'IsNight']].head().to_markdown(index=False))
print(df.info())
# Convert to datetime objects, handling errors
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
df['Date Rptd'] = pd.to_datetime(df['Date Rptd'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

# Calculate the time difference in hours
df['TimeToReport_Hours'] = (df['Date Rptd'] - df['DATE OCC']).dt.total_seconds() / 3600

# Extract the month of occurrence (you still might need this)
df['Occ_Month'] = df['DATE OCC'].dt.month

# Drop the original date columns (optional, but usually good practice)
df = df.drop(['DATE OCC', 'Date Rptd'], axis=1)
# Drop rows where 'Vict Sex' or 'Vict Descent' is empty or '-'
df = df[df['Vict Sex'].astype(str).str.strip().isin(['M','F'])]
df = df[df['Vict Descent'].astype(str).str.strip().isin(['H','W','B','O','A'])]

# Drop rows where 'Vict Age' is less than 0
df = df[df['Vict Age'] >= 0]

# Replace empty or '-' in 'Weapon Desc' and 'Cross Street' with 'undefined'
df['Weapon Desc'] = df['Weapon Desc'].fillna('undefined').astype(str).str.strip().replace('-', 'undefined')
df['Premis Desc'] = df['Premis Desc'].fillna('undefined').astype(str).str.strip().replace('-', 'undefined')

#if cross street = 1, if no = 0
df['Has_Cross_Street_np'] = np.where((df['Cross Street'].notna()) & (df['Cross Street'] != ''), 1, 0)
df = df.drop(columns = ['Cross Street'])
#check wether last operation worked well or not
df.info()
# refine lon and lat. make clusters and add them in one column with KMEANS



# 1. Clustering with KMeans
n_clusters = 20  # Number of clusters you want
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Location_Cluster'] = kmeans.fit_predict(df[['LAT', 'LON']])

# 2. Representing clusters in the DataFrame

# a) One column with cluster labels (recommended)
print("One column with cluster labels:")
print(df.head())
#lets get rid of lon and lat 
df = df.drop(columns = ['LAT', 'LON', 'LOCATION', 'OCCURRED_DATETIME'])
df.isnull().sum() #assesstment of empty cells again
df.info()
df=df.drop(['TIME OCC'], axis=1)
df.info()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cyclical encoding for 'Hour'
sample_df['Hour_Sin'] = np.sin(2 * np.pi * sample_df['Hour'] / 24)
sample_df['Hour_Cos'] = np.cos(2 * np.pi * sample_df['Hour'] / 24)

# Updated preprocessing step: Encoding categorical features
# Select categorical columns for encoding
categorical_columns = ['AREA NAME', 'Crm Cd Desc', 'Vict Sex', 'Vict Descent', 'Premis Desc', 
                       'Weapon Desc', 'Status Desc', 'DayOfWeek', 'TimeOfDay']

# Apply one-hot encoding and clean feature names
X_encoded = pd.get_dummies(sample_df.drop(columns=['Hour']), columns=categorical_columns, drop_first=True)
X_encoded['Hour_Sin'] = sample_df['Hour_Sin']  # Add cyclical time features
X_encoded['Hour_Cos'] = sample_df['Hour_Cos']  # Add cyclical time features

# Replace problematic characters in column names
X_encoded.columns = X_encoded.columns.str.replace(r'[\[\], ]+', '_', regex=True)
X_encoded.columns = X_encoded.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove other special characters
X_encoded.columns = X_encoded.columns.str.strip('_')  # Remove leading/trailing underscores

# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = X_encoded.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# Splitting data
y_hour = sample_df['Hour']
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_hour, test_size=0.2, random_state=42)

# Model Definitions
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2 Score': r2}
    print(f"{name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    
    # Confusion matrix and heatmap visualization
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Classification Report
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, y_pred))

# Comparing Results
results_df = pd.DataFrame(results).T
print("Model Performance Comparison:")
print(results_df)

# Visualization of Performance
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
results_df['MSE'].plot(kind='bar', ax=axes[0], color='skyblue', title='Model Comparison: MSE')
axes[0].set_ylabel('Mean Squared Error')
results_df['R2 Score'].plot(kind='bar', ax=axes[1], color='salmon', title='Model Comparison: R2 Score')
axes[1].set_ylabel('R2 Score')
plt.tight_layout()
plt.show()

# Additional Visualization: Feature importance for tree-based models
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
        top_features = feature_importances.nlargest(10)
        top_features.plot(kind='barh', figsize=(8, 5), color='teal')
        plt.title(f'Top 10 Feature Importances - {name}')
        plt.xlabel('Importance Score')
        plt.show()


st.title(" My project_Crime_data_analysis_US ")

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import pandas as pd

# Assuming sample_df is provided

# Cyclical encoding for 'Hour'
sample_df['Hour_Sin'] = np.sin(2 * np.pi * sample_df['Hour'] / 24)
sample_df['Hour_Cos'] = np.cos(2 * np.pi * sample_df['Hour'] / 24)

# Updated preprocessing step: Encoding categorical features
# Select categorical columns for encoding
categorical_columns = ['AREA NAME', 'Crm Cd Desc', 'Vict Sex', 'Vict Descent', 'Premis Desc', 
                       'Weapon Desc', 'Status Desc', 'DayOfWeek', 'TimeOfDay']

# Apply one-hot encoding and clean feature names
X_encoded = pd.get_dummies(sample_df.drop(columns=['Hour']), columns=categorical_columns, drop_first=True)
X_encoded['Hour_Sin'] = sample_df['Hour_Sin']  # Add cyclical time features
X_encoded['Hour_Cos'] = sample_df['Hour_Cos']  # Add cyclical time features

# Replace problematic characters in column names
X_encoded.columns = X_encoded.columns.str.replace(r'[\[\], ]+', '_', regex=True)
X_encoded.columns = X_encoded.columns.str.replace(r'[^a-zA-Z0-9_]', '', regex=True)  # Remove other special characters
X_encoded.columns = X_encoded.columns.str.strip('_')  # Remove leading/trailing underscores

st.title('Crime Data Model Comparison')

# Correlation matrix
st.subheader('Correlation Matrix')
plt.figure(figsize=(12, 10))
corr_matrix = X_encoded.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
st.pyplot(plt)

# Splitting data
y_hour = sample_df['Hour']
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_hour, test_size=0.2, random_state=42)

# Hyperparameter controls
n_estimators = st.sidebar.slider('n_estimators (Random Forest)', min_value=10, max_value=200, value=100, step=10)
max_depth = st.sidebar.slider('max_depth (Random Forest)', min_value=5, max_value=50, value=None, step=5)
learning_rate = st.sidebar.slider('learning_rate (XGBoost, LightGBM)', min_value=0.01, max_value=0.3, value=0.1, step=0.01)

# Model Definitions
models = {
    'RandomForest': RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', learning_rate=learning_rate),
    'LightGBM': LGBMClassifier(random_state=42, learning_rate=learning_rate),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2 Score': r2}
    st.write(f"{name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    
    # Confusion matrix and heatmap visualization
    st.subheader(f'{name} - Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

    # Classification Report
    st.write(f"Classification Report for {name}:")
    st.text(classification_report(y_test, y_pred))

# Comparing Results
results_df = pd.DataFrame(results).T
st.subheader('Model Performance Comparison')
st.dataframe(results_df)

# Visualization of Performance
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
results_df['MSE'].plot(kind='bar', ax=axes[0], color='skyblue', title='Model Comparison: MSE')
axes[0].set_ylabel('Mean Squared Error')
results_df['R2 Score'].plot(kind='bar', ax=axes[1], color='salmon', title='Model Comparison: R2 Score')
axes[1].set_ylabel('R2 Score')
st.pyplot(fig)

# Additional Visualization: Feature importance for tree-based models
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        st.subheader(f'Top 10 Feature Importances - {name}')
        feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
        top_features = feature_importances.nlargest(10)
        top_features.plot(kind='barh', figsize=(8, 5), color='teal')
        plt.title(f'Top 10 Feature Importances - {name}')
        plt.xlabel('Importance Score')
        st.pyplot(plt)
