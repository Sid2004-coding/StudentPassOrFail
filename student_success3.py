import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("student_success_dataset(Sheet1).csv")
le=LabelEncoder()
df['Internet']=le.fit_transform(df['Internet']) # yes/no ko 1/0 me convert kar diya
df['Passed']=le.fit_transform(df['Passed'])

features=['StudyHours','Attendance','PastScore','SleepHours']
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scaler.fit_transform(df[features])#normalization of features

X=df_scaled[features] # features
y=df_scaled['Passed'] # target variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Classification Report")
print(classification_report(y_test,y_pred))
print("Confusion Matrix")
cm=confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['Not Passed','Passed'],yticklabels=['Not Passed','Passed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("------------Predict Your Result------------") # student se input lene ke liye ki voh pass hoga ya nahi
try:
    study_hours=float(input("Enter Study Hours: "))
    attendance=float(input("Enter Attendance: "))
    past_score=float(input("Enter Past Score: "))
    sleep_hours=float(input("Enter Sleep Hours: "))
    user_input_df=pd.DataFrame([{
        'StudyHours':study_hours,
        'Attendance':attendance,
        'PastScore':past_score,
        'SleepHours':sleep_hours,
    }])
    user_input_scaled=scaler.transform(user_input_df) # agar kahi student 80 dal toh voh bhi scale ho jaye scale karne kei liye
    prediction=model.predict(user_input_scaled)[0] # 0 or 1
    result='Passed' if prediction==1 else 'Not Passed'
    print(f"Prediction based on input: {result}")
except Exception as e:
    print("An error occurred")