import numpy as np
from PIL import Image

import numpy as np
class ImagePreprocessor:
    def load(self, path):
        from PIL import Image
        image = Image.open(path).convert("L")
        self.image_array = np.array(image)
        return self.image_array

    def cluster(self, k):
        from sklearn.cluster import KMeans
        reshaped_image = self.image_array.reshape(-1, 1)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(reshaped_image)
        self.clustered_image = kmeans.cluster_centers_[kmeans.labels_].reshape(self.image_array.shape)
        return self.clustered_image

    def convolve(self):
        from scipy.signal import convolve2d
        a = -0.0625
        b = 0.5
        kernel = np.array([
            [a, a, a],
            [a, b, a],
            [a, a, a]
        ])
        self.convolved_image = convolve2d(self.clustered_image, kernel, mode="same")
        return self.convolved_image

    def threshold(self, threshold=0.5):
        self.binary_image = np.where(self.convolved_image > threshold, 0, 255).astype(np.uint8)
        return self.binary_image

    def plot_all(self, path):
        self.load(path)
        self.cluster(k=4)
        self.convolve()
        self.threshold()
        self.show_all()

    def show_all(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(25, 5))
        axes[0].imshow(self.image_array, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        axes[1].imshow(self.clustered_image, cmap="gray")
        axes[1].set_title("Clustered Image")
        axes[1].axis("off")
        axes[2].imshow(self.convolved_image, cmap="gray")
        axes[2].set_title("Convolved Image")
        axes[2].axis("off")
        axes[3].imshow(self.binary_image, cmap="gray")
        axes[3].set_title("Thresholded Image")
        axes[3].axis("off")
        plt.tight_layout()
        plt.show()

class Converter:
    def fit(self,picture):
        black_indices = np.argwhere(picture == 0)
        sampled_points = black_indices
        mu =black_indices.mean()
        distances = np.linalg.norm((sampled_points - mu), axis=1)
        sampled_points = sampled_points[np.argsort(distances)]
        sampled_points = sampled_points[2000:]
        random_indices = np.random.choice(np.arange(sampled_points.shape[0]), size=1000, replace=False)
        sampled_points = sampled_points[random_indices]
        return sampled_points
        T = np.linspace(0,500)
        T = T/500




# Create an instance of the ImagePreprocessor class
processor = ImagePreprocessor()

# Path to the image file
image_path = "./data/testing purposes/Face.png"

# Load the image and preprocess it
image_array = processor.load(image_path)
clustered_image = processor.cluster(k=4)
convolved_image = processor.convolve()
binary_image = processor.threshold()
binary_image = binary_image[1:binary_image.shape[0]-1,1:binary_image.shape[1]-1]
# Create an instance of the Converter class
converter = Converter()

# Get the sampled points using Converter
sampled_points = converter.fit(binary_image)

# Animate the appearance of the sampled points
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ax.imshow(binary_image, cmap="gray")
points, = ax.plot([], [], 'ro', markersize=3)

def update(frame):
    if frame < len(sampled_points):
        points.set_data(sampled_points[:frame+1, 1], sampled_points[:frame+1, 0])
    return points,

ani = FuncAnimation(fig, update, frames=len(sampled_points)+1, interval=50, blit=True)

plt.show()
