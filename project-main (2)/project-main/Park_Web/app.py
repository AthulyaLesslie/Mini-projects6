from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dframe = pd.read_csv("C:\\Users\\athul\\Downloads\\project-main (2)\\project-main\\Park_Web\\park.data")
    features = dframe.loc[:, dframe.columns != 'status'].iloc[:, 1:].values
    labels = dframe.loc[:, 'status'].values

    # Scale the features to a non-negative range
    scaler = MinMaxScaler((0, 1))
    features = scaler.fit_transform(features)

    selector = SelectKBest(score_func=chi2, k=5)
    selected_features = selector.fit_transform(features, labels)
    selected_feature_indices = selector.get_support(indices=True)

    selected_columns = ['status'] + list(dframe.columns[1:][selected_feature_indices])
    selected_df = dframe[selected_columns]
    X = selected_df.drop('status', axis=1).values
    y = selected_df['status'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=7, subsample=0.8, colsample_bytree=0.7, gamma=0.01)
    model.fit(X_train, y_train)

    input_data = np.array([float(request.form['fo']), float(request.form['flo']), float(request.form['shimmer']), float(request.form['dfa']), float(request.form['d2'])]).reshape(1, -1)
    print("input:", input_data)
    scaler = MinMaxScaler((-1, 1))

    # Use the scaler fitted on the training data to transform the input features
    scaled_input_features = scaler.fit_transform(input_data)

    prediction = model.predict(input_data)
    print("prediction:", prediction)
    if prediction == 0:
        result = 'No Parkinson\'s Disease'
    else:
        result = 'Parkinson\'s Disease'

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
