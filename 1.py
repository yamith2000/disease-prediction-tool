import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def predict_disease():
    symptoms = np.array([var.get() for var in symptom_vars])
    symptoms = symptoms.reshape(1, -1)
    prediction = model.predict(symptoms)[0]
    result_label.config(text=f"Based on your symptoms, you may have {prediction}")
    

df = pd.read_csv('diseases.csv')
features = df.columns.drop('Disease')
X = df[features]
y = df['Disease']

model = DecisionTreeClassifier(random_state=0)
model.fit(X, y)

root = tk.Tk()
root.title("Disease Prediction Tool")


symptom_vars = []
for i, feature in enumerate(features):
    var = tk.IntVar()
    tk.Checkbutton(root, text=feature, variable=var, font=(12)).grid(row=i+1, column=0)
    symptom_vars.append(var)

predict_button = tk.Button(root, text="Predict", command=predict_disease, font=(12), bg="yellow")
predict_button.grid(row=i//2, column=1)

result_label = tk.Label(root, text="Please select your symptoms to get a prediction", font=(20))
result_label.grid(row=0, column=0)

root.mainloop()
