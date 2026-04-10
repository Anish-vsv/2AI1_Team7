# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ==============================
# 2. Load Dataset
# ==============================
train_df = pd.read_csv("Titanic_train.csv")
test_df = pd.read_csv("Titanic_test.csv")

# ==============================
# 3. Handle Missing Values
# ==============================

# Age → fill with median
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())

# Embarked → fill with mode
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

# Fare → (test dataset may have missing)
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())

# Drop Cabin (too many missing values)
train_df = train_df.drop('Cabin', axis=1)
test_df = test_df.drop('Cabin', axis=1)

# ==============================
# 4. Convert Categorical → Numerical
# ==============================

# Sex → convert to numeric
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Embarked → One-Hot Encoding
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# ==============================
# 5. Drop Unnecessary Columns
# ==============================
train_df = train_df.drop(['Name', 'Ticket', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name', 'Ticket', 'PassengerId'], axis=1)

# ==============================
# 6. Align Train & Test Columns (VERY IMPORTANT)
# ==============================
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

# ==============================
# 7. Feature Scaling
# ==============================
scaler = StandardScaler()

# Scale Age & Fare
train_df[['Age', 'Fare']] = scaler.fit_transform(train_df[['Age', 'Fare']])
test_df[['Age', 'Fare']] = scaler.transform(test_df[['Age', 'Fare']])

# ==============================
# 8. Final Check
# ==============================
print("Train Data Info:\n")
print(train_df.info())

print("\nMissing Values:\n")
print(train_df.isnull().sum())

# ==============================
# 9. Save Cleaned Data
# ==============================
train_df.to_csv("cleaned_train.csv", index=False)
test_df.to_csv("cleaned_test.csv", index=False)

print("\n✅ Data Preprocessing Completed Successfully!")