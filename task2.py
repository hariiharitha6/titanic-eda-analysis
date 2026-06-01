import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Titanic-Dataset.csv")

 
# Basic Information
 

print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== SUMMARY STATISTICS =====")
print(df.describe())

 
# Histogram - Age
 

plt.figure(figsize=(8,5))
df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("Age_Histogram.png")
plt.show()

 
# Boxplot - Age
 

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Age"])
plt.title("Age Boxplot")
plt.savefig("Age_Boxplot.png")
plt.show()

 
# Boxplot - Fare
 

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Boxplot")
plt.savefig("Fare_Boxplot.png")
plt.show()

 
# Correlation Matrix
 

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(8,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.savefig("Correlation_Matrix.png")
plt.show()

# Pairplot


sns.pairplot(
    df[['Age', 'Fare', 'Pclass', 'Survived']]
)

plt.savefig("Pairplot.png")
plt.show()

# Pattern Analysis


print("\n===== SURVIVAL BY SEX =====")
print(df.groupby("Sex")["Survived"].mean())

print("\n===== SURVIVAL BY CLASS =====")
print(df.groupby("Pclass")["Survived"].mean())

print("\n===== SURVIVAL BY EMBARKED =====")
print(df.groupby("Embarked")["Survived"].mean())

print("\nEDA COMPLETED SUCCESSFULLY")