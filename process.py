
import matplotlib.pyplot as plt
import numpy as np

file = open('results3.txt', 'r')
lines = file.readlines()

dic = {}

tests = ["trafficV", "game-of-life", "structureV", "generationV", "BFS", "CC", "PR", "BFSV", "CCV", "PRV"]
data = {
    'CUDA': [],
    'SharedOA': [],
    'COAL': [],
    'TypePointer': [],
    'DynamicSharedOA': []
}

for line in lines:
    if line.strip() == "":
        continue
    spl = line.split(" ")

    if spl[0].endswith("MEM"):
        data['SharedOA'].append(float(spl[1]))
    elif spl[0].endswith("COAL"):
        data['COAL'].append(float(spl[1]))
    elif spl[0].endswith("TP"):
        data['TypePointer'].append(float(spl[1]))
    elif spl[0].endswith("MEM_2"):
        data['DynamicSharedOA'].append(float(spl[1]))
    else:
        data['CUDA'].append(float(spl[1]))

# normalize data
for i in range(len(tests)):
    sharedoa_factor = 1 / data['SharedOA'][i]
    for key in data:
        data[key][i] *= sharedoa_factor

# convert to tuple
for key in data:
    data[key] = tuple(e for e in data[key])

print(data)

x = np.arange(len(tests))
width = 0.1
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in data.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, fmt='%.2f', padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Performance (Normalized to SharedOA)')
ax.set_title('Tests')
ax.set_xticks(x + width, tests)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 10)

plt.show()

