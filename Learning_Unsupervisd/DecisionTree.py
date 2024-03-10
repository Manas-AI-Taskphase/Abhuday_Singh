import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE

def processdata(file):
    df = pd.read_csv(file)
    targets = df['Threat']
    attributes = df.drop('Threat', axis=1)
    smote = SMOTE()
    attributes, targets = smote.fit_resample(attributes, targets)

    train_attributes, test_attributes, train_targets, test_targets \
        = train_test_split(attributes, targets)
    return train_attributes, test_attributes, train_targets, test_targets

def trainmodel(train_attributes, train_targets):
    clf = DecisionTreeClassifier()
    clf.fit(train_attributes, train_targets)
    return clf

def testmodel(test_attributes, test_targets, clf):
    predictions = clf.predict(test_attributes)
    report = classification_report(test_targets, predictions)
    # Calculate F1 score
    f1 = f1_score(test_targets, predictions)
    return report, f1

if __name__ == "__main__":
    #train_x, test_x, train_y, test_y = processdata('/Users/abhudaysingh/Downloads/test (1).csv')
    train_x, test_x, train_y, test_y = processdata('/Users/abhudaysingh/Downloads/threats.csv')
    classification_model = trainmodel(train_x, train_y)
    rep, f1 = testmodel(test_x, test_y, classification_model)
    print("Classification Report:\n", rep)
    print("\nF1 Score:", f1)
    
