import numpy as np
import matplotlib.pyplot as plt

image = np.load("dataset/images/frame_04153.npy")

plt.figure(figsize=(10, 7))
plt.imshow(image, cmap="gray")
plt.title("frame_04153 - Worst CNN Prediction")
plt.axis("on")
plt.show()