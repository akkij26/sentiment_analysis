import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data
def load_npz(file_path):
    with np.load(file_path) as data:
        return {key: data[key] for key in data}

train_data = load_npz('train.npz')
test_data = load_npz('test.npz')
train_emb1, train_emb2, train_labels = train_data['emb1'], train_data['emb2'], train_data['preference']
test_emb1, test_emb2 = test_data['emb1'], test_data['emb2']

print("training set shape")
# we have 18750 points 

# splitting into training and validation set 
data_train_emb1, data_test_emb1, labels_train, labels_test = train_test_split(train_emb1, train_labels, test_size=0.20, random_state=42)
data_train_emb2, data_test_emb2, labels_train, labels_test = train_test_split(train_emb2, train_labels, test_size=0.20, random_state=42)

# print("1",data_train_emb1.shape)
# print("2",data_train_emb2.shape)
# print("3",labels_train.shape)
# print("1",data_test_emb1.shape)
# print("2",data_test_emb2.shape)
# print("3",labels_test.shape)

# Concatenate the embeddings
train_features = np.concatenate((data_train_emb1, data_train_emb2), axis=1)
test_features = np.concatenate((data_test_emb1, data_test_emb2), axis=1)

rf_model = RandomForestClassifier()
#rf_model.fit(train_features, labels_train)

# Make predictions
#train_predictions = rf_model.predict(test_features)

# Calculate accuracy (training data)
#accuracy = accuracy_score(labels_test, train_predictions)

#print(f"Model Accuracy: {accuracy}")

# Bagging Classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Number of base estimators: 100
num_trees = 100

# Bagging classifier
model = BaggingClassifier(base_estimator=rf_model, n_estimators=num_trees, random_state=42)

# Fit model to training data
model.fit(train_features, labels_train)

# Predict and evaluate
test_predictions = model.predict(test_features)
accuracy = accuracy_score(labels_test, test_predictions)

print(f"Model Accuracy: {accuracy}")
