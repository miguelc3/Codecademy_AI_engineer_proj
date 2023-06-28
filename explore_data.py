import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\Miguel\pyproj\Codecademy_AI_engineer_proj\creditcard.csv')

"""
The dataset includes 28 features, which are numerical values obtained by PCA transformation to maintain the 
confidentiality of sensitive information

Also it has a time column, which is the number of seconds elapsed between each transaction and the first transaction
It has a column named Amount which is the transaction amount
And a column named Class which is the target variable, it takes the value 1 in case of fraud and 0 otherwise
"""

# Print nr of fraud and non-fraud transactions
is_fraud = data.Class.value_counts()
print('Number of fraud transactions: ', is_fraud[1])
print('Number of non-fraud transactions: ', is_fraud[0])

# See time description
print(data.Time.describe())
time_vals = data.Time.value_counts().sort_index()
plt.plot(time_vals)
plt.title('Time distribution')

# See amount description
print(data.Amount.describe())
amount_vals = data.Amount.value_counts().sort_index()
plt.plot(amount_vals)
plt.title('Amount distribution')

print('Not fraudulent transactions:')
n_fraud = data[data.Class == 0]
print(n_fraud.Time.describe())

print('Fraudulent transactions:')
fraud = data[data.Class == 1]
print(fraud.Time.describe())

"""
Statistical values are very similar except for the min value
"""

# Plot correlation heatmap
corr = data.corr()
plt.figure(figsize=(14, 14))
heat = plt.imshow(corr, cmap='Blues', interpolation='none')
plt.colorbar(heat)
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)
plt.show()

"""
No features are highly correlated with the target variable -> good
"""


