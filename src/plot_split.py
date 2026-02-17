from pathlib import Path
import sys
import csv
import matplotlib.pyplot as plt

CSV_PATH = Path("outputs/split_dist.csv")
OUTPUT = Path("outputs/split_dist.png")

classes = []
trains = []
vals = []
tests = []
totals = []

if not CSV_PATH.exists():
    print("Fajl ne postoji!")
    sys.exit()

with CSV_PATH.open("r") as f:
    r = csv.reader(f)
    header = next(r, None)
    for row in r:
        classes.append(row[0])
        trains.append(int(row[1]))
        vals.append(int(row[2]))
        tests.append(int(row[3]))
        totals.append(int(row[4]))

temp = [x + y for x, y in zip(trains, vals)]
temp2 = [x + y for x,y in zip(temp, tests)]

plt.figure(figsize = (10, 6))
plt.bar(classes, trains, label = "Trening skup")
plt.bar(classes, vals, bottom = trains, label = "Validation skup")
plt.bar(classes, tests, bottom = temp, label = "Test skup")
plt.bar(classes, totals, bottom = temp2, label = "Ukupno")
plt.ylabel("Broj slika")
plt.title("Raspodela train, val, test, total")
plt.legend()
plt.savefig(OUTPUT)
plt.close()

print("NAPRAVLJEN -> ", OUTPUT)