## Machine Learning / AI engineer Codecademy project

In this repository is the project code for an assignment from the Codecademy course.

The project will consist of using a kaggle dataset (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) with credit card transaction data and creating a machine learning pipeline to detect fraudulent transactions.

Basically, I started by testing some algorithms (logistic regression, random forests and svm), used gridsearch to find the best hyperparameters and calculated statistics to check which one performed better. For this, I assumed that the best one was the one with the highest f1_score.

After that, I created a pipeline that pre-processed the data using StandardScaller and applied the best model to the data.

The next steps would probably be to test a neural network on the data to see how it performs.
