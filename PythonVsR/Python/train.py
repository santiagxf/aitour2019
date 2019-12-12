from azureml.core import Dataset, Run

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn_pandas import DataFrameMapper
from sklearn.externals import joblib
import numpy as np
import os

# Get current run context for AzureML
run = Run.get_context()
dataset = run.input_datasets['hr_employee_attrition']

# Convert to pandas dataframe
attritionData = dataset.to_pandas_dataframe()

# Separete numeral from categorical features
columns_target = 'Attrition'
columns_categorical = attritionData.select_dtypes(include=['object']).columns.values
columns_numerical = attritionData.columns.difference(columns_categorical)
columns_numerical = [x for x in columns_numerical if x != columns_target]

# Create a transformation for numerical features, an inputer (for columns with missing values) and an scaler

numeric_transformations = [([f], Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])) for f in columns_numerical]

# Create a transformation for categorical feature, a one hot encoder
categorical_transformations = [([f], OneHotEncoder(handle_unknown='ignore', sparse=False)) for f in columns_categorical]
columns_categorical = np.delete(columns_categorical, [0])

# Put both transformations together
transformations = numeric_transformations + categorical_transformations

# Create a Pipeline with the preprocessing step and train a classifier

clf = Pipeline(steps=[('preprocessor', DataFrameMapper(transformations)),
                      ('classifier', RandomForestClassifier(n_estimators=100))])


# Remove the target from the training set
attritionX = attritionData.drop(columns_target, axis=1)
attritiony = attritionData[columns_target]

# Separate in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(attritionX,
                                                    attritiony,
                                                    test_size = 0.2,
                                                    random_state=123,
                                                    stratify=attritiony)

# Start training
model = clf.fit(X_train, y_train)

# +
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):

    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


# -

cm = confusion_matrix(y_test, clf.predict(X_test))
print(cm)
fig = print_confusion_matrix(cm, class_names=["not leaving","leaving"])
run.log_image("Confusion", plot=fig)

# Log metrics
run.log("Accuracy", clf.score(X_test, y_test))
run.log("Method", "RandomForestClassifier")

model_file_name = 'log_reg.pkl'
model_file_path = os.path.join('./outputs/', model_file_name)
# save model in the outputs folder so it automatically get uploaded
with open(model_file_name, 'wb') as file:
    joblib.dump(value=clf, filename=model_file_path)

# +
from azureml.core import Model
from azureml.core import Dataset

azureml_model = run.register_model(model_name='hr-employee-attrition',
                                   model_path=model_file_path,
                                   description='Logistic regression Scikit-Learn model for Attrition prediction.',
                                   tags={'area': 'attrition', 'type': 'classification'},
                                   datasets=[(Dataset.Scenario.TRAINING, dataset)])

# +
from interpret.ext.blackbox import TabularExplainer
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient

# Explain predictions on your local machine
tabular_explainer = TabularExplainer(model, X_train, features=attritionData.feature_names)

# Explain overall model predictions (global explanation)
# Passing in test dataset for evaluation examples - note it must be a representative sample of the original data
# x_train can be passed as well, but with more examples explanations it will
# take longer although they may be more accurate
global_explanation = tabular_explainer.explain_global(X_test)

# Uploading model explanation data for storage or visualization in webUX
# The explanation can then be downloaded on any compute
comment = 'Global explanation on regression model trained on boston dataset'
client.upload_model_explanation(global_explanation, comment=comment, model_id=azureml_model.id)
