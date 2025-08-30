import pandas as pd
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("student_success_dataset(Sheet1).csv")
print("Missing Values in each column")
print(df.isnull().sum()) # har ek colum me kitne missing values hai
le=LabelEncoder()
df['Internet']=le.fit_transform(df['Internet']) # yes/no ko 1/0 me convert kar diya
df['Passed']=le.fit_transform(df['Passed'])
print("After Encoding ")
print(df.head())
print("Data-types after cleaning")
print(df.dtypes)

