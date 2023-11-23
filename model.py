import numpy as np

def load_npz(file_path):
    with np.load(file_path) as data:
        return {key: data[key] for key in data}

train_data = load_npz('train.npz')
test_data = load_npz('test.npz')
train_emb1, train_emb2, train_labels = train_data['emb1'], train_data['emb2'], train_data['preference']
test_emb1, test_emb2 = test_data['emb1'], test_data['emb2']