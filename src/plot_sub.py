from pathlib import Path
import csv
import matplotlib.pyplot as plt
import sys

CSV_PATH = Path("outputs/sub_dist.csv")
OUTPUT = Path("outputs/sub_dist.png")

classes = []
images_all = []
images_sel = []

if not CSV_PATH.exists():
    print("Fajl ne postoji!")
    sys.exit()

with CSV_PATH.open("r") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        classes.append(row[0])
        images_all.append(int(row[1]))
        images_sel.append(int(row[2]))

images_left = []

for i in range(len(images_all)):
    images_left.append(images_all[i] - images_sel[i])

plt.figure()
plt.bar(classes, images_sel, label = "Nas uzorak")
plt.bar(classes, images_left, bottom = images_sel, label = "Ukupno slika")
plt.ylabel("Broj slika")
plt.title("Raspodela klasa (subsample)")
plt.legend()
plt.savefig(OUTPUT)
plt.close()

print("NAPRAVLJEN -> ", OUTPUT)