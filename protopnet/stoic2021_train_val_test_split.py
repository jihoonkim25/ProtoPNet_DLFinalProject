import os
import shutil
import csv
import numpy as np
import pandas as pd


trimmed_set = []
trimmed_path = os.path.abspath("/stoic-2021") + "/metadata/trimmed.csv"

with open(trimmed_path, newline='') as f:
    reader = csv.reader(f)
    trimmed_set = list(reader)
print("Trimmed set: ", len(trimmed_set))

# split train/val/test from trimmed dataset (note that trimmed dataset is already randomized)
train_set = []
val_set = []
test_set = []
for i in range(325):
    train_set.append(trimmed_set[i])
for j in range(i+1,350):
    val_set.append(trimmed_set[j])
for k in range(j+1,400):
    test_set.append(trimmed_set[k])
print("Train set: ", len(train_set))
print("Validation set: ", len(val_set))
print("Test set: ", len(test_set))

# save train/val/test scan_id + label into csv
with open("./metadata/train.csv", "w", newline="") as f:
    write = csv.writer(f)
    write.writerows(train_set)
with open("./metadata/val.csv", "w", newline="") as f:
    write = csv.writer(f)
    write.writerows(val_set)
with open("./metadata/test.csv", "w", newline="") as f:
    write = csv.writer(f)
    write.writerows(test_set)

# test whether save worked as planned
with open("./metadata/train.csv", newline='') as f:
    reader = csv.reader(f)
    train = list(reader)
print("Train set: ", len(train))

# move trimmed data into new directory
# c = 0
# for t in trimmed_set:
#     old_path = os.path.abspath("/stoic-2021") + \
#         "/data/mha/" + str(t[0]) + ".mha"
#     new_path = os.path.abspath("/stoic-2021") + "/trim/" + str(t[0]) + ".mha"
#     shutil.copy(old_path, new_path)
#     print(str(c) + " of " + str(len(trimmed_set)) + " copied")
#     c += 1
