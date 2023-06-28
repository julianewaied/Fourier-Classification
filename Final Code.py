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
        np.random.shuffle(black_indices)
        sampled_points = black_indices
        sampled_points = sampled_points[:500]
        mu =black_indices.mean()
        distances = np.sum(np.abs(sampled_points - mu), axis=1)
        sampled_points = sampled_points[np.argsort(distances)]
        sampled_points = sampled_points[:500]
        return sampled_points
        T = np.linspace(0,500)
        T = T/500
processor = ImagePreprocessor()
processor.plot_all("C:/Users/WIN10PRO/Desktop/My Stuff/University/Others/ML and FT/Fourier-Classification/data/testing purposes/me.png")

