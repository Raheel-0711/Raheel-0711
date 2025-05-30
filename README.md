import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


df = pd.read_csv("train.csv")

# ========== 1. DATA CLEANING ==========


df['Age'].fillna(df['Age'].median(), inplace=True)


df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


df.drop('Cabin', axis=1, inplace=True)

# ========== 2. NOISY DATA HANDLING ==========


def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df = remove_outliers_iqr(df, 'Fare')
df = remove_outliers_iqr(df, 'Age')

# ========== 3. DATA INTEGRATION ==========

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# ========== 4. CATEGORICAL ENCODING ==========

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])         # male=1, female=0
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  # C=0, Q=1, S=2 (auto order)

# ========== 5. SCALING NUMERIC FEATURES ==========

scaler = StandardScaler()
df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(df[['Age', 'Fare', 'FamilySize']])

# ========== OUTPUT ==========

df.to_csv("titanic_preprocessed.csv", index=False)
print("✅ Preprocessing complete. File saved as 'titanic_preprocessed.csv'.")
