import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_npz(file_path):
    with np.load(file_path) as data:
        return {key: data[key] for key in data}

train_data = load_npz('train.npz')
test_data = load_npz('test.npz')
train_emb1, train_emb2, train_labels = train_data['emb1'], train_data['emb2'], train_data['preference']
test_emb1, test_emb2 = test_data['emb1'], test_data['emb2']


# Concatenate the embeddings
train_features = np.concatenate((train_emb1, train_emb2), axis=1)
print(train_features)
test_features = np.concatenate((test_emb1, test_emb2), axis=1)

rf_model = RandomForestClassifier()
rf_model.fit(train_features, train_labels)

# Make predictions
train_predictions = rf_model.predict(train_features)

# Calculate accuracy (training data)
accuracy = accuracy_score(train_labels, train_predictions)
print(f"Model Accuracy: {accuracy}")


