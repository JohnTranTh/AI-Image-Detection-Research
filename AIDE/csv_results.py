import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score

# Load the CSV file
df = pd.read_csv('prediction_output.csv', header=None)
df.columns = ['image_path', 'true_label', 'predicted_label', 'confidence_score']

# Convert columns to appropriate types
df['true_label'] = df['true_label'].astype(int)
df['predicted_label'] = df['predicted_label'].astype(int)
df['confidence_score'] = df['confidence_score'].astype(float)

# Extract necessary arrays
true_labels = df['true_label'].values
predicted_labels = df['predicted_label'].values
confidence_scores = df['confidence_score'].values

# Accuracy: how many predicted labels match true labels
accuracy = accuracy_score(true_labels, predicted_labels)

# Average Precision: especially useful for binary classification with confidence scores
average_precision = average_precision_score(true_labels, confidence_scores)

print(f"Accuracy: {accuracy:.4f}")
print(f"Average Precision: {average_precision:.4f}")
