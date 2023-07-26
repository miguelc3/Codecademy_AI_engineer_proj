import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
import numpy as np


df = pd.read_csv(r'C:\Users\Miguel\pyproj\Codecademy_AI_engineer_proj\creditcard.csv')


data = df.drop(columns=['Class'])
target = df['Class']

# Data is going to be split into training, test and validation sets
"""
20% test
20% validation
60% training
"""
X_trainval, X_test, y_trainval, y_test = train_test_split(data, target
                                                          , test_size=0.2
                                                          , stratify=df['Class']
                                                          , random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval
                                                  , test_size=0.25
                                                  , stratify=y_trainval
                                                  , random_state=42)

# Initialize the StandardScaler object and fit it to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Scale the training, validation, and test sets using the scaler
X_train_std = scaler.transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

"""
Undersampling will be utilized to address the issue of imbalanced classes.
Undersampling is a technique to balance uneven datasets by keeping all of the data in the minority class and decreasing 
the size of the majority class. 
"""

# Instantiate RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Undersample the training set
X_train_under, y_train_under = rus.fit_resample(X_train_std, y_train)

# Undersample the validation set
X_val_under, y_val_under = rus.fit_resample(X_val_std, y_val)
#
# # ============== Logistic Regression ==============
# """
# Perform grid search to find the best hyperparameters for the model
# Verify best penalty and C values
# """
penalty = ['l2']
C = np.logspace(0, 4, 10, 100, 1000)
param_grid = dict(C=C, penalty=penalty)

logistic = LogisticRegression(solver='lbfgs', max_iter=100000)
logistic_grid = GridSearchCV(logistic, param_grid, cv=5, scoring='roc_auc', verbose=10, n_jobs=-1)
logistic_grid.fit(X_train_under, y_train_under)

# Get the best hyperparameters from the grid search
best_logistic = logistic_grid.best_estimator_

# Find stats of performance on the training and validation set
best_logistic.fit(X_train_under, y_train_under)
print('Validation Accuracy: {:.3f}'.format(best_logistic.score(X_val_under, y_val_under)))

# ============== Random Forest ==============
"""
Perform grid search to find the best hyperparameters for the model
Verify best n_estimators and max_depth values
"""
n_estimators = [100, 300, 500, 600, 700]
max_depth = [3, 5, 10, 15, 20]
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)

rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', verbose=10, n_jobs=-1)
rf_grid.fit(X_train_under, y_train_under)

# Get the best hyperparameters from the grid search
best_rf = rf_grid.best_estimator_

# Find stats of performance on the training and validation set
best_rf.fit(X_train_under, y_train_under)
print('Validation Accuracy: {:.3f}'.format(best_rf.score(X_val_under, y_val_under)))

# ============== Support Vector Machine ==============
"""
Perform grid search to find the best hyperparameters for the model
Verify best C and gamma values
"""
C = [0.1, 1, 10]
gamma = [0.001, 0.01, 0.1, 1]

param_grid = dict(C=C, gamma=gamma)
svm_grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5, scoring='roc_auc', verbose=10, n_jobs=-1)
svm_grid.fit(X_train_under, y_train_under)

# Get the best hyperparameters from the grid search
best_svm = svm_grid.best_estimator_

# Find stats of performance on the training and validation set
best_svm.fit(X_train_under, y_train_under)
print('Validation Accuracy: {:.3f}'.format(best_svm.score(X_val_under, y_val_under)))

# ============== Dummy Classifier ==============
"""
Dummy to serve as a baseline for comparing with actual classifiers, especially with imbalanced classes.
"""
dummy = DummyClassifier()
dummy.fit(X_train_under, y_train_under)

# Find stats of performance on the training and validation set
print('Validation Accuracy: {:.3f}'.format(dummy.score(X_val_under, y_val_under)))

# ============== Evaluate the models ==============
# Determine threshold for each model -> highest f1_score


def find_t(model, nr_steps):
    highest_f1 = 0
    best_threshold = 0
    best_acc = 0
    best_rec = 0
    best_pre = 0
    # Iterate over a range of thresholds
    for t in np.linspace(0, 1, nr_steps):
        # Predict the target variable using the given t
        y_predict = (model.predict_proba(X_val_under)[:, 1] >= t)
        # Calculate various evaluation metrics
        f1 = f1_score(y_val_under, y_predict)
        acc = accuracy_score(y_val_under, y_predict)
        rec = recall_score(y_val_under, y_predict)
        pre = precision_score(y_val_under, y_predict)
        # Update the best t and metrics if F1 score improves
        if f1 > highest_f1:
            best_threshold, highest_f1, best_acc, best_rec, best_pre = \
                t, f1, acc, rec, pre
    # Return the best t and evaluation metrics
    return best_threshold, highest_f1, best_acc, best_rec, best_pre


# Define a list of models and their names
models = [logistic_grid, rf_grid, svm_grid]
model_names = ["Logistic Regression", "Random Forest", 'SVM']
# Create an empty list to store the results
chart = list()

# Iterate over the models and find the best threshold for each one
for item, name in zip(models, model_names):
    best_thresh, high_f1, high_acc, high_rec, high_pre = find_t(item, 20)
    # Append the results to the chart list
    chart.append([name, best_thresh, high_f1, high_acc, high_rec, high_pre])

# Create a pandas dataframe from the chart list and display it
chart = pd.DataFrame(chart, columns=['Model', 'Best Threshold', 'F1 Score', 'Accuracy', 'Recall', 'Precision'])

# Find model with highest F1 score
best_model = chart[chart['F1 Score'] == chart['F1 Score'].max()]['Model'].values[0]
print('Best model is {}'.format(best_model))

# Find idx of best model in models names list
idx = model_names.index(best_model)
best_estimator = models[idx]

# Create a pipeline with the best model
pipeline = Pipeline([('scaler', StandardScaler()), ('model', best_estimator)])

# Fit the pipeline to the training data
pipeline.fit(X_train_under, y_train_under)

# Predict the target variable
y_pred = pipeline.predict(X_val_under)

# Print the evaluation metrics
print('Accuracy: {:.3f}'.format(accuracy_score(y_val_under, y_pred)))
print('Recall: {:.3f}'.format(recall_score(y_val_under, y_pred)))
print('Precision: {:.3f}'.format(precision_score(y_val_under, y_pred)))
print('F1: {:.3f}'.format(f1_score(y_val_under, y_pred)))


