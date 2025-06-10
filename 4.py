import pandas as pd

def find_s_algorithm(file_path):
    data = pd.read_csv(file_path)
    attributes = data.columns[:-1]
    label = data.columns[-1]
    hypothesis = ['?' for _ in attributes]
    for _, row in data.iterrows():
        if row[label] == 'Yes':
            for i, val in enumerate(row[attributes]):
                hypothesis[i] = val if hypothesis[i] == '?' or hypothesis[i] == val else '?'
    return hypothesis

file_path = 'training_data.csv'
hypothesis = find_s_algorithm(file_path)
print("Final Hypothesis:", hypothesis)
