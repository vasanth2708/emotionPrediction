from joblib import load


def make_prediction(text):
    logistic_regression_model = load('logistic_regression_model.pkl')
    naive_bayes_model = load('naive_bayes_model.pkl')
    logistic_prediction = logistic_regression_model.predict([text])
    naive_bayes_prediction = naive_bayes_model.predict([text])
    return {
        "Logistic Regression": logistic_prediction,
        "Naive Bayes": naive_bayes_prediction
    }
