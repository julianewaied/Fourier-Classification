import numpy as np
from PIL import Image

image_path = "./data/testing purposes/Face.png"
class ImagePreprocessor:
    def fit(self,path):
        self.load(path)
        self.cluster(k=4)
        self.convolve()
        self.threshold()
    def get(self,type):
        if type=="image":
            return self.image_array
        elif type == "conv":
            return self.convolved_image
        elif type == "clust":
            return self.clustered_image
        elif type == "bin":
            return self.binary_image
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
        self.binary_image = self.binary_image[1:self.binary_image.shape[0]-1,1:self.binary_image.shape[1]-1]
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
        self.black_indices = np.argwhere(picture == 0)
    def get(self,type):
        return self.black_indices

def getBorder(black_points):
    y_coordinates = black_points[:, 1]

    unique_y = np.unique(y_coordinates)

    min_x_points = []

    for y in unique_y:
        filtered_points = black_points[black_points[:,1] == y]
        min_x_index = np.argmin(filtered_points[:, 0])
        min_x_point = filtered_points[min_x_index]
        min_x_points.append(min_x_point)
    min_x_points = np.array(min_x_points)
    return min_x_points


processor = ImagePreprocessor()
processor.fit(image_path)
binary_image = processor.get("bin")
converter = Converter()
converter.fit(binary_image)
sampled_points = converter.get("black")


sampled_points = getBorder(sampled_points)








import matplotlib.pyplot as plt
plt.imshow(binary_image, cmap="gray")
plt.plot(sampled_points[:, 1], sampled_points[:, 0], '-r')
plt.show()