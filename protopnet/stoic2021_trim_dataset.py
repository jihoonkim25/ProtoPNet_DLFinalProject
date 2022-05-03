import os
import shutil
import csv

trimmed_set = []
trimmed_path = os.path.abspath("/stoic-2021") + "/metadata/trimmed.csv"

with open(trimmed_path, newline='') as f:
    reader = csv.reader(f)
    trimmed_set = list(reader)
#print(trimmed_set)
c = 0
for t in trimmed_set:
    old_path = os.path.abspath("/stoic-2021") + "/data/mha/" + str(t[0]) + ".mha"
    new_path = os.path.abspath("/stoic-2021") + "/trim/" + str(t[0]) + ".mha"
    shutil.copy(old_path, new_path)
    print(str(c) + " of " + str(len(trimmed_set)) + " copied")
    c += 1