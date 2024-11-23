import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray, label2rgb
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from skfuzzy import cmeans
from PIL import Image

# Load your image
image_path = 'fat.jpg'  # Replace this with your image path
uploaded_image = Image.open(image_path)
image = np.array(uploaded_image)

# Function to apply Otsu's Thresholding
def otsu_segmentation(image):
    gray_image = rgb2gray(image)
    thresh = threshold_otsu(gray_image)
    segmented = (gray_image > thresh).astype(np.uint8)
    return label2rgb(segmented, image=image, bg_label=0)

# Function to apply Watershed Segmentation
def watershed_segmentation(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    distance = distance_transform_edt(binary)
    ret, markers = cv2.connectedComponents((distance > 0.2 * distance.max()).astype(np.uint8))
    markers = markers + 1
    markers[binary == 255] = 0
    segmented = watershed(-distance, markers, mask=binary)
    return label2rgb(segmented, image=image)

# Function to apply K-Means Clustering
def kmeans_segmentation(image, n_clusters=3):
    flat_image = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(flat_image)
    labels = kmeans.labels_.reshape(image.shape[:2])
    return label2rgb(labels, image=image)

# Function to apply Fuzzy C-Means
def fuzzy_cmeans_segmentation(image, n_clusters=3):
    flat_image = image.reshape((-1, 3)).T  # Fuzzy C-Means expects features as rows
    cntr, u, _, _, _, _, _ = cmeans(flat_image, n_clusters, 3, error=0.005, maxiter=500)
    labels = np.argmax(u, axis=0).reshape(image.shape[:2])
    return label2rgb(labels, image=image)

# Apply the segmentation techniques
otsu_segmented = otsu_segmentation(image)
watershed_segmented = watershed_segmentation(image)
kmeans_segmented = kmeans_segmentation(image, n_clusters=3)
fuzzy_segmented = fuzzy_cmeans_segmentation(image, n_clusters=3)

# Plot the results
plt.figure(figsize=(20, 6))

# Original image
plt.subplot(1, 5, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

# Otsu segmentation
plt.subplot(1, 5, 2)
plt.title("Otsu's Segmentation")
plt.imshow(otsu_segmented)
plt.axis("off")

# Watershed segmentation
plt.subplot(1, 5, 3)
plt.title("Watershed Segmentation")
plt.imshow(watershed_segmented)
plt.axis("off")

# K-Means segmentation
plt.subplot(1, 5, 4)
plt.title("K-Means Segmentation")
plt.imshow(kmeans_segmented)
plt.axis("off")

# Fuzzy C-Means segmentation
plt.subplot(1, 5, 5)
plt.title("Fuzzy C-Means Segmentation")
plt.imshow(fuzzy_segmented)
plt.axis("off")

plt.tight_layout()
plt.show()
