import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("⏳ Loading dataset... (Please wait)")

# Load CSV files
fake_news = pd.read_csv('Fake.csv')
true_news = pd.read_csv('True.csv')

# Add labels (0 = Fake, 1 = Real)
fake_news['label'] = 0
true_news['label'] = 1

# Combine both datasets into one DataFrame
df = pd.concat([fake_news, true_news], ignore_index=True)

# Remove any rows with empty text
df = df.dropna(subset=['text'])

print("🚀 Starting AI model training... (This might take 1-2 minutes)")

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text data to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(tfidf_train, y_train)

# Calculate and print model accuracy
accuracy = model.score(tfidf_test, y_test)
print(f"✅ Model training successful! Accuracy: {accuracy * 100:.2f}%")

print("💾 Saving the trained model...")
# Save the trained model and vectorizer for future use in the API
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("🎉 Done! 'model.pkl' and 'vectorizer.pkl' are ready.")