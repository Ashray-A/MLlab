import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                   'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
})

encoders = {}
for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

def entropy(y):
    vals, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))

def info_gain(data, split_attribute, target):
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / len(data)) *
        entropy(data[data[split_attribute] == vals[i]][target])
        for i in range(len(vals))
    )
    return entropy(data[target]) - weighted_entropy

def id3(data, target, features):
    if len(np.unique(data[target])) == 1:
        return int(np.unique(data[target])[0])
    if len(features) == 0:
        return int(data[target].mode()[0])

    gains = [info_gain(data, f, target) for f in features]
    best = features[np.argmax(gains)]
    tree = {best: {}}

    for val in np.unique(data[best]):
        sub = data[data[best] == val]
        if sub.empty:
            tree[best][str(val)] = int(data[target].mode()[0])
        else:
            new_features = [f for f in features if f != best]
            tree[best][str(val)] = id3(sub, target, new_features)

    return tree

features = list(data.columns[:-1])
target = 'PlayTennis'
tree = id3(data, target, features)
print("Tree:", tree)

def predict(tree, sample):
    for attr, branches in tree.items():
        val = str(sample[attr])
        if val in branches:
            result = branches[val]
            if isinstance(result, dict):
                return predict(result, sample)
            else:
                return encoders[target].inverse_transform([result])[0]
    return "Unknown"

sample = {}
for feature in features:
    user_val = input(f"Enter value for {feature}: ")
    sample[feature] = encoders[feature].transform([user_val])[0]

pred = predict(tree, sample)
print("\nPredicted class:", pred)
