# Импорты необходимых библиотек
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import pandas as pd

# Загрузка данных (укажите путь к вашему файлу)
data_path = 'crime_sample_data.csv'  # Укажите ваш путь к файлу с данными
sample_df = pd.read_csv(data_path).sample(5000, random_state=42)

# ------ Обработка данных ------
# Преобразование времени (часов) в циклические признаки для сохранения периодичности
sample_df['Hour_Sin'] = np.sin(2 * np.pi * sample_df['Hour'] / 24)
sample_df['Hour_Cos'] = np.cos(2 * np.pi * sample_df['Hour'] / 24)

# Категориальные признаки для кодирования
categorical_columns = ['AREA NAME', 'Crm Cd Desc', 'Vict Sex', 'Vict Descent', 'Premis Desc', 
                       'Weapon Desc', 'Status Desc', 'DayOfWeek', 'TimeOfDay']

# Кодирование категориальных признаков методом One-Hot Encoding
X_encoded = pd.get_dummies(sample_df.drop(columns=['Hour']), columns=categorical_columns, drop_first=True)
X_encoded['Hour_Sin'] = sample_df['Hour_Sin']
X_encoded['Hour_Cos'] = sample_df['Hour_Cos']

# Очистка имен столбцов от недопустимых символов
X_encoded.columns = X_encoded.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
X_encoded.columns = X_encoded.columns.str.strip('_')

# ------ Визуализация корреляционной матрицы ------
plt.figure(figsize=(12, 10))
corr_matrix = X_encoded.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix')
plt.show()

# ------ Разделение данных на обучающую и тестовую выборки ------
y_hour = sample_df['Hour']
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_hour, test_size=0.2, random_state=42)

# ------ Настройка гиперпараметров модели ------
n_estimators = 100  # Количество деревьев в лесу
max_depth = None    # Максимальная глубина дерева

# ------ Обучение модели ------
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------ Оценка производительности ------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"RandomForest - MSE: {mse:.4f}, R2 Score: {r2:.4f}")

# ------ Матрица ошибок ------
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ------ Отчет классификации ------
print('Classification Report:')
print(classification_report(y_test, y_pred))

# ------ Важность признаков ------
if hasattr(model, 'feature_importances_'):
    feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
    top_features = feature_importances.nlargest(10)
    top_features.plot(kind='barh', figsize=(8, 5), color='teal')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance Score')
    plt.show()
