# --- Data Handling & Manipulation ---
import pandas as pd
import numpy as np

# --- Data Preprocessing ---
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# --- Machine Learning Models ---
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBRegressor, XGBClassifier  # Assuming xgboost is installed
from lightgbm import LGBMRegressor, LGBMClassifier

# --- Model Selection & Evaluation ---
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, accuracy_score

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns

# --- Utility ---
from sklearn.pipeline import Pipeline
from scipy.stats import randint

#EDA and PREPROCESSING


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



# Clustering with KMeans
n_clusters = 20  # Number of clusters you want
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Location_Cluster'] = kmeans.fit_predict(df[['LAT', 'LON']])

#  Representing clusters in the DataFrame

# a) One column with cluster labels 
print("One column with cluster labels:")
print(df.head())
#lets get rid of lon and lat 
df = df.drop(columns = ['LAT', 'LON', 'LOCATION', 'OCCURRED_DATETIME'])
df.isnull().sum() #assesstment of empty cells again
df.info()
df=df.drop(['TIME OCC'], axis=1)
df.info()



sample_df = df.sample(5000, random_state=42)
# Data preprocessing
# Step 1: Cyclical Encoding for 'Hour'
sample_df['Hour_Sin'] = np.sin(2 * np.pi * sample_df['Hour'] / 24)
sample_df['Hour_Cos'] = np.cos(2 * np.pi * sample_df['Hour'] / 24)

# Step 2: Encoding categorical features
categorical_columns = ['AREA NAME', 'Crm Cd Desc', 'Vict Sex', 'Vict Descent', 'Premis Desc', 
                       'Weapon Desc', 'Status Desc', 'DayOfWeek', 'TimeOfDay']
X_encoded = pd.get_dummies(sample_df.drop(columns=['Hour']), columns=categorical_columns, drop_first=True)
X_encoded['Hour_Sin'] = sample_df['Hour_Sin']
X_encoded['Hour_Cos'] = sample_df['Hour_Cos']
X_encoded.columns = X_encoded.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True).str.strip('_')

# Target variable
y_hour = sample_df['Hour']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_hour, test_size=0.2, random_state=42)

# Model Training and Parameterization
n_estimators = 100
max_depth = None

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Visualizations and Metrics
# Correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = X_encoded.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RandomForest - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification Report
print("Classification Report")
print(classification_report(y_test, y_pred))

# Feature Importances
if hasattr(model, 'feature_importances_'):
    feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
    top_features = feature_importances.nlargest(10)
    top_features.plot(kind='barh', figsize=(8, 5), color='teal')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
    plt.show()
