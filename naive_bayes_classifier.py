import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_file_path = input ("Path Of SMS Spam Collection Dataset: ")

def naive_bayes(x_train, y_train, x_test):
    class_counts = defaultdict(int)
    word_counts = defaultdict(lambda: defaultdict(int))

    for text, label in zip(x_train, y_train):
        class_counts[label] += 1
        for word in text.split():
            word_counts[label][word] += 1

    total_count = sum(class_counts.values())
    class_probs = {label: count / total_count for label, count in class_counts.items()}

    word_probs = {}
    for label, words in word_counts.items():
        total_words = sum(words.values())
        word_probs[label] = {word: (count + 1) / (total_words + len(words)) for word, count in words.items()}

    predictions = []
    for text in x_test:
        class_scores = {label: np.log(class_probs[label]) for label in class_probs}

        for word in text.split():
            for label in class_probs:
                if word in word_probs[label]:
                    class_scores[label] += np.log(word_probs[label][word])
                else:
                    class_scores[label] += np.log(1 / (sum(word_probs[label].values()) + len(word_probs[label])))

        predictions.append(max(class_scores, key=class_scores.get))
    return predictions

data = pd.read_csv(data_file_path, header=None, names=['label', 'message'], on_bad_lines='skip', na_filter=True)
data.dropna(subset=['label', 'message'], inplace=True)
x = data['message'].fillna("")
y = data['label']

'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

predictions = naive_bayes(x_train, y_train, x_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy : {accuracy * 100:.2f}%")
'''

train_sizes = [800, 1600, 2400, 3200, 4000]
results = []

for train_size in train_sizes:
    accuracies = []
    for _ in range(5):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None)

        predictions = naive_bayes(x_train, y_train, x_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    results.append(mean_accuracy)

results_df = pd.DataFrame({
    'Train Size' : train_sizes,
    'Average Accuracy' : results
})

print(results_df)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, results, marker='o', linestyle='-', color='b', label='Accuracy')
plt.title('Average Accuracy vs Training Size')
plt.xlabel('Training Size')
plt.ylabel('Average Accuracy')
plt.grid(True)
plt.xticks(train_sizes)
plt.legend()
plt.show()