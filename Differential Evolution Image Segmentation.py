%reset -f

# Differential Evolution Image Segmentation

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define Differential Evolution parameters
def de_segment(image_array, n_segments, n_population=50, n_iterations=100, mutation_factor=0.8, crossover_rate=0.7):
    height, width, channels = image_array.shape
    flat_image = image_array.reshape((-1, channels))
    n_pixels = flat_image.shape[0]

    # Initialize population
    population = np.random.rand(n_population, n_segments, channels)

    # Define fitness function: Clustering sum of squared errors
    def fitness(individual):
        distances = np.linalg.norm(flat_image[:, None, :] - individual, axis=2)
        closest_centroid = np.argmin(distances, axis=1)
        sse = 0
        for i in range(n_segments):
            cluster_points = flat_image[closest_centroid == i]
            if len(cluster_points) > 0:
                sse += np.sum(np.linalg.norm(cluster_points - individual[i], axis=1) ** 2)
        return sse

    # Store cost values
    cost_values = []

    # Optimization loop
    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")

        next_population = np.copy(population)
        for i in range(n_population):
            # Mutation: Generate donor vector
            indices = [idx for idx in range(n_population) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            donor = a + mutation_factor * (b - c)

            # Crossover: Generate trial vector
            trial = np.copy(population[i])
            for j in range(n_segments):
                if np.random.rand() < crossover_rate:
                    trial[j] = donor[j]

            # Selection: Evaluate fitness
            if fitness(trial) < fitness(population[i]):
                next_population[i] = trial

        # Update population
        population = next_population

        # Evaluate fitness for the updated population
        fitness_values = np.array([fitness(individual) for individual in population])
        best_fitness = np.min(fitness_values)
        best_individual = population[np.argmin(fitness_values)]

        print(f"  Best fitness so far: {best_fitness:.4f}")

        # Store the best fitness value for plotting
        cost_values.append(best_fitness)

    return best_individual, cost_values

# Function to apply clustering
def apply_segmentation(image_array, centroids):
    flat_image = image_array.reshape((-1, image_array.shape[2]))
    distances = np.linalg.norm(flat_image[:, None, :] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    segmented_image = centroids[labels].reshape(image_array.shape)
    return segmented_image.astype(np.uint8)

# Convert to black and white
def to_black_and_white(image):
    gray_image = np.mean(image, axis=2)  # Convert to grayscale
    bw_image = (gray_image > gray_image.mean()).astype(np.uint8) * 255  # Threshold for black and white
    return bw_image

# Load the image
image_path = 'fat2.jpg'  # Replace with your image path
image = Image.open(image_path)
image = np.array(image)

# Define the number of segments
n_segments = 4

# Perform segmentation
segmented_centroids, cost_values = de_segment(image, n_segments)
segmented_image = apply_segmentation(image, segmented_centroids)
bw_segmented_image = to_black_and_white(segmented_image)

# Plot the original, segmented images, black-and-white segmented, and the cost values
plt.figure(figsize=(15, 12))

# Original image
plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

# Segmented image
plt.subplot(2, 3, 2)
plt.title('DE Segmented Image')
plt.imshow(segmented_image)
plt.axis('off')

# Black-and-white segmented image
plt.subplot(2, 3, 3)
plt.title('Black and White Segmented')
plt.imshow(bw_segmented_image, cmap='gray')
plt.axis('off')

# Cost values plot
plt.subplot(2, 1, 2)
plt.title('Cost Function Over Iterations')
plt.semilogy(range(1, len(cost_values) + 1), cost_values, marker='o', label='Cost Function (Log Scale)')
plt.xlabel('Iteration')
plt.ylabel('Cost (Log Scale)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
