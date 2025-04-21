import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('E:\\pridiction\\diabetes\\diabetes.csv')  # Make sure the path is correct

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Now, ask the user for all 8 important input features with options
print("\nPlease answer the following important questions:")

# User inputs (with options)
pregnancies = float(input("How many times have you been pregnant? (Enter a number): "))
glucose = int(input("Aapka blood glucose level kya hai? (Rakt sharkara ka level)\nOption 1: 70-99 mg/dL (Normal)\nOption 2: 100-125 mg/dL (Pre-diabetes)\nOption 3: 126 mg/dL ya usse zyada (Diabetes)\nEnter option number: "))
glucose_values = {1: 85, 2: 110, 3: 130}  # Placeholder values based on options
glucose = glucose_values.get(glucose, 85)  # Default to 85 if invalid input

blood_pressure = int(input("Aapka blood pressure kitna hai?\nOption 1: Normal (120/80 mmHg tak)\nOption 2: High (130/90 mmHg ya zyada)\nOption 3: Low (90/60 mmHg tak)\nEnter option number: "))
blood_pressure_values = {1: 80, 2: 100, 3: 70}  # Placeholder values based on options
blood_pressure = blood_pressure_values.get(blood_pressure, 80)

skin_thickness = int(input("Aapki skin thickness (triceps ke aas-paas) kitni hai?\nOption 1: 10-20 mm\nOption 2: 20-30 mm\nOption 3: 30 mm se zyada\nEnter option number: "))
skin_thickness_values = {1: 15, 2: 25, 3: 35}  # Placeholder values based on options
skin_thickness = skin_thickness_values.get(skin_thickness, 15)

insulin = int(input("Aapka insulin level kya hai?\nOption 1: 5-20 µU/mL (Normal)\nOption 2: 20-50 µU/mL (Borderline)\nOption 3: 50 µU/mL ya usse zyada (High)\nEnter option number: "))
insulin_values = {1: 10, 2: 30, 3: 60}  # Placeholder values based on options
insulin = insulin_values.get(insulin, 10)

bmi = float(input("Aapka BMI (Body Mass Index) kitna hai?\nOption 1: 18.5-24.9 (Normal weight)\nOption 2: 25-29.9 (Overweight)\nOption 3: 30 ya zyada (Obese)\nEnter option number: "))
bmi_values = {1: 22, 2: 27, 3: 32}  # Placeholder values based on options
bmi = bmi_values.get(bmi, 22)

dpf = int(input("Kya aapke family mein kisi ko diabetes hai? (Diabetes Pedigree Function)\nOption 1: Haan\nOption 2: Nahi\nOption 3: Pata nahi\nEnter option number: "))
dpf_values = {1: 0.5, 2: 0.1, 3: 0.2}  # Placeholder values based on options
dpf = dpf_values.get(dpf, 0.1)

age = int(input("Aapki age kitni hai?\nOption 1: 18-30\nOption 2: 31-50\nOption 3: 51 ya zyada\nEnter option number: "))
age_values = {1: 25, 2: 40, 3: 55}  # Placeholder values based on options
age = age_values.get(age, 25)

# Create a list from the user's input (with all 8 features)
user_input = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]

# Predict the probability (output will be in percentage)
prediction_prob = model.predict_proba(user_input)[0]
diabetic_prob = prediction_prob[1] * 100  # Probability of being diabetic
non_diabetic_prob = prediction_prob[0] * 100  # Probability of being non-diabetic

# Display the result as a percentage
print(f"\nProbability of being Diabetic: {diabetic_prob:.2f}%")
print(f"Probability of not being Diabetic: {non_diabetic_prob:.2f}%")

# Display result in graph
plt.bar('Diabetic', diabetic_prob, color='red', label='Diabetic')
plt.bar('Not Diabetic', non_diabetic_prob, color='green', label='Not Diabetic')
plt.title("Diabetes Prediction")
plt.xlabel("Outcome")
plt.ylabel("Probability (%)")
plt.legend()
plt.show()

# Conclusion based on the prediction
if diabetic_prob > 50:
    prediction_text = 'Diabetic'
else:
    prediction_text = 'Not Diabetic'

print(f"\nThe prediction is: {prediction_text}")
