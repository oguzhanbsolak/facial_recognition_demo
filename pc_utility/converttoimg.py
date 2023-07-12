
import matplotlib.pyplot as plt
import numpy as np

with open("rgb4.log") as f:
	content = f.readlines()


img = np.zeros((160, 120, 3), np.uint8)
for y in range(160):
	for x in range(120):
		img[y][x][0] = int(content[y*120+x].split(",")[0])
		img[y][x][1] = int(content[y*120+x].split(",")[1])
		img[y][x][2] = int(content[y*120+x].split(",")[2])

print(img.min(), img.max())
plt.imshow(img)
plt.show()
