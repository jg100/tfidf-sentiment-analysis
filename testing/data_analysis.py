import os

categories = [0] * 10
folder_path = "../aclImdb/train/pos"
neg_folder_path = "../aclImdb/train/neg"
for file in os.listdir(folder_path):
    rating = int(file[file.index('_') + 1: file.index('.')])
    categories[rating - 1] = categories[rating - 1] + 1

for file in os.listdir(neg_folder_path):
    rating = int(file[file.index('_') + 1: file.index('.')])
    categories[rating - 1] = categories[rating - 1] + 1

print(sum(categories))
