"""
Course Project

avega24@georgefox.edu
"""

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics

# Importing the data
medical_data_df = pd.read_csv('2016-2023-imi-results-long-view.csv',
                              na_values={'# of Cases':'.', '# of Deaths':'.'})

# Filtering out NAN values
medical_data_df = medical_data_df.dropna()


"""
# Section to get fig2.pdf
# Only getting the top 10Counties
topCounties = medical_data_df[medical_data_df['COUNTY'].
                isin(['Los Angeles', 'Orange', 'San Bernardino', 'Riverside', 'San Diego',
                      'Alameda', 'San Francisco', 'Kern', 'Santa Clara', 'Sacramento'])]

print(topCounties['COUNTY'].value_counts())

# Made arrays to hold names of each county and number of their cases
county_names = []
county_count = []

# Adds values to each array
for name, count in topCounties['COUNTY'].value_counts().items():
    county_names.append(name)
    county_count.append(count)


# Plots the count per county
fig, ax = plt.subplots()
ax.bar(county_names, county_count)
ax.set_title("County and # of their unique case")
ax.set_xlabel("County")
ax.set_ylabel("Count")
ax.tick_params('x', rotation=45)
plt.show()
"""


"""
# Section to get fig3.pdf
# # Selecting the lowest 5 conditions
# lowest_data = medical_data_df[medical_data_df['Procedure/Condition'].
#             isin(['Espophageal Resection', 'AAA Repair Open Unruptured', 'Pancreatic Cancer',
#                   'Acute Stroke Subarachnoid', 'PCI'])]
# 
# print(lowest_data['Procedure/Condition'].value_counts())
# 
# # Categorizing data for the three different medical conditions
# espophageal_data = lowest_data[lowest_data['Procedure/Condition'] == 'Espophageal Resection']
# aaa_data = lowest_data[lowest_data['Procedure/Condition'] == 'AAA Repair Open Unruptured']
# cancer_data = lowest_data[lowest_data['Procedure/Condition'] == 'Pancreatic Cancer']
# acute_data = lowest_data[lowest_data['Procedure/Condition'] == 'Acute Stroke Subarachnoid']
# pci_data = lowest_data[lowest_data['Procedure/Condition'] == 'PCI']
# 
# # Setting up plots for the three conditions
# plt.scatter(espophageal_data['# of Cases'], espophageal_data['# of Deaths'],
#                                             c='red', label='Espophageal Resection',alpha=0.3)
# plt.scatter(aaa_data['# of Cases'], aaa_data['# of Deaths'],
#                                             c='green', label='AAA Repair Open Unruptured', alpha=0.3)
# plt.scatter(cancer_data['# of Cases'], cancer_data['# of Deaths'],
#                                             c='blue', label='Pancreatic Cancer', alpha=0.3)
# plt.scatter(acute_data['# of Cases'], acute_data['# of Deaths'],
#                                             c='brown', label='Acute Stroke Subarachnoid', alpha=0.3)
# plt.scatter(pci_data['# of Cases'], pci_data['# of Deaths'],
#                                             c='black', label='PCI', alpha=0.3)
# 
# plt.title('# of Cases vs Deaths')
# plt.xlabel('# of Cases')
# plt.ylabel('# of Deaths')
# plt.legend()
# plt.show()
"""

# Selecting only three medical conditions
case_data = medical_data_df[medical_data_df['Procedure/Condition'].
            isin(['Pneumonia', 'Heart Failure', 'GI Hemorrhage'])]
print(case_data['Procedure/Condition'].value_counts())

# Categorizing data for the three different medical conditions
pneumonia_data = case_data[case_data['Procedure/Condition'] == 'Pneumonia']
heart_failure_data = case_data[case_data['Procedure/Condition'] == 'Heart Failure']
gi_hemorrhage_data = case_data[case_data['Procedure/Condition'] == 'GI Hemorrhage']

# Setting up plots for the three conditions
plt.scatter(pneumonia_data['# of Cases'], pneumonia_data['# of Deaths'],
                                            c='grey', label='Pneumonia',alpha=0.3)
plt.scatter(heart_failure_data['# of Cases'], heart_failure_data['# of Deaths'],
                                            c='red', label='Heart Failure', alpha=0.3)
plt.scatter(gi_hemorrhage_data['# of Cases'], gi_hemorrhage_data['# of Deaths'],
                                            c='green', label='GI Hemorrhage', alpha=0.3)

plt.title('# of Cases vs Deaths')
plt.xlabel('# of Cases')
plt.ylabel('# of Deaths')
plt.legend()
plt.show()


# Setting up X and y
X = case_data[['# of Cases', '# of Deaths']]
y = case_data['Procedure/Condition']

# Scale the features in X
scaler = sklearn.preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# K-fold cross-validation
# Creating a k-fold generator object that splits the data into 5 different splits
kfold_skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True)

# Lists to hold y_test and y_pred
y_test_list = []
y_pred_list = []

for i, (train_index, test_index) in enumerate(kfold_skf.split(X,y)):
    print(f'\n\nfold {i}')

    # Using train indices for this fold to slice out just the roles for the training and testing set
    # Uses .iloc to indicate that we are using row indices, not column names
    X_train = X[train_index]
    y_train = y.iloc[train_index]
    X_test = X[test_index]
    y_test = y.iloc[test_index]

    # Making a model using the current training data
    model = sklearn.svm.SVC(class_weight='balanced')
    model.fit(X_train,y_train)

    # Predicts the test set
    y_pred = model.predict(X_test)

    # Calculating both Raw and Balanced accuracies
    accuracy = sklearn.metrics.accuracy_score(y_test,y_pred)
    print(f'Accuracy: {accuracy:.3f}')
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test,y_pred)
    print(f'Balanced accuracy: {balanced_accuracy:.3f}')

    # Extending the lists by appending each item from the current field
    y_test_list.extend(y_test)
    y_pred_list.extend(y_pred)


# Computing the overall metrics
print(f'\n\nAll folds')
print(f'len(test) = {len(y_test_list)}')
list_accuracy = sklearn.metrics.accuracy_score(y_test_list, y_pred_list)
print(f'List accuracy: {list_accuracy:.3f}')
list_balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test_list, y_pred_list)
print(f'List balanced accuracy: {list_balanced_accuracy:.3f}')


# Confusion Matrix
sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_test_list,y_pred_list, cmap='Blues', colorbar=None)
plt.title('All Folds')
plt.show()