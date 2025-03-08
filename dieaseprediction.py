import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


df = pd.read_csv('/training_data.csv')  


X = df['symptoms']  # Symptoms in text format
y = df['prognosis']  # Disease label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

new_symptoms_text = input("Enter your symptoms: ")  # User enters symptoms as text
new_symptoms_tfidf = vectorizer.transform([new_symptoms_text])  # Convert to numerical format
predicted_disease = model.predict(new_symptoms_tfidf)
print(f'Predicted Disease: {predicted_disease[0]}')
