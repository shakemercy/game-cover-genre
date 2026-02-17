from pathlib import Path
import csv
import matplotlib.pyplot as plt
import sys

CSV_PATH = Path("outputs/raw_dist.csv")
OUTPUT = Path("outputs/raw_dist.png")

classes = []
images = []

if not CSV_PATH.exists():
    print("Fajl ne postoji!")
    sys.exit()

with CSV_PATH.open("r") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        classes.append(row[0])
        images.append(int(row[1]))

plt.figure()
plt.bar(classes, images)
plt.ylabel("Broj slika")
plt.title("Raspodela klasa (nebalansirano)")
plt.savefig(OUTPUT)
plt.close()

print("NAPRAVLJEN -> ", OUTPUT)
