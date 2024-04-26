import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import neattext.functions as nfx
from joblib import dump

# Load dataset
df = pd.read_csv("./data/emotion_dataset_raw.csv")

# Data cleaning using Neattext
df['Clean_Text'] = df['Text'].apply(lambda x: nfx.remove_userhandles(x))
df['Clean_Text'] = df['Clean_Text'].apply(lambda x: nfx.remove_stopwords(x))

# Display data distribution
print(df.head())
print(df['Emotion'].value_counts())
sns.countplot(x='Emotion', data=df)

# Prepare features and labels
X = df['Clean_Text']
y = df['Emotion']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating pipelines for Logistic Regression and Naive Bayes
pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', LogisticRegression())
])

pipeline_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('classifier', MultinomialNB())
])

param_grid_lr = {
    'classifier__C': [0.1, 1, 10]
}
grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=3, verbose=2, n_jobs=-1)
grid_lr.fit(X_train, y_train)

param_grid_nb = {
    'classifier__alpha': [0.01, 0.1, 1]
}
grid_nb = GridSearchCV(pipeline_nb, param_grid_nb, cv=3, verbose=2, n_jobs=-1)
grid_nb.fit(X_train, y_train)

print("Best parameters for Logistic Regression:", grid_lr.best_params_)
print("Best cross-validation score for Logistic Regression: {:.2f}".format(grid_lr.best_score_))
print("Accuracy on test set for Logistic Regression:", accuracy_score(y_test, grid_lr.predict(X_test)))
print(classification_report(y_test, grid_lr.predict(X_test)))

print("Best parameters for Naive Bayes:", grid_nb.best_params_)
print("Best cross-validation score for Naive Bayes: {:.2f}".format(grid_nb.best_score_))
print("Accuracy on test set for Naive Bayes:", accuracy_score(y_test, grid_nb.predict(X_test)))
print(classification_report(y_test, grid_nb.predict(X_test)))


sample_text = "Recession is hittinh hard"
print("Predicted emotion with Logistic Regression:", grid_lr.predict([sample_text]))
print("Predicted emotion with Naive Bayes:", grid_nb.predict([sample_text]))

dump(grid_lr.best_estimator_, 'logistic_regression_model.pkl')
dump(grid_nb.best_estimator_, 'naive_bayes_model.pkl')