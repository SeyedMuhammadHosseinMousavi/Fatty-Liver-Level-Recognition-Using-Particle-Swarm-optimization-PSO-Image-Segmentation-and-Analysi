
%reset -f

# PSO Image Segmentation

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Define PSO parameters
def pso_segment(image_array, n_segments, n_particles=50, n_iterations=20):
    height, width, channels = image_array.shape
    flat_image = image_array.reshape((-1, channels))
    n_pixels = flat_image.shape[0]

    # Initialize particle positions and velocities
    particles = np.random.rand(n_particles, n_segments, channels)
    velocities = np.random.rand(n_particles, n_segments, channels) * 0.1
    personal_best = np.copy(particles)
    global_best = np.copy(particles[np.random.randint(n_particles)])
    global_best_fitness = float('inf')

    # Define fitness function: Clustering sum of squared errors
    def fitness(particle):
        distances = np.linalg.norm(flat_image[:, None, :] - particle, axis=2)
        closest_centroid = np.argmin(distances, axis=1)
        sse = 0
        for i in range(n_segments):
            cluster_points = flat_image[closest_centroid == i]
            if len(cluster_points) > 0:
                sse += np.sum(np.linalg.norm(cluster_points - particle[i], axis=1) ** 2)
        return sse

    # Store cost values
    cost_values = []

    # Optimization loop
    inertia_weight = 0.7
    cognitive_weight = 1.5
    social_weight = 1.5

    for iteration in range(n_iterations):
        print(f"Iteration {iteration + 1}/{n_iterations}")
        for i in range(n_particles):
            # Update velocity
            inertia = inertia_weight * velocities[i]
            cognitive = cognitive_weight * np.random.rand() * (personal_best[i] - particles[i])
            social = social_weight * np.random.rand() * (global_best - particles[i])
            velocities[i] = inertia + cognitive + social

            # Update position
            particles[i] += velocities[i]

            # Evaluate fitness
            current_fitness = fitness(particles[i])
            personal_best_fitness = fitness(personal_best[i])

            if current_fitness < personal_best_fitness:
                personal_best[i] = particles[i]
            if current_fitness < global_best_fitness:
                global_best = particles[i]
                global_best_fitness = current_fitness

        # Save global best fitness for this iteration
        cost_values.append(global_best_fitness)
        print(f"  Best fitness so far: {global_best_fitness:.4f}")

    return global_best, cost_values

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
image_path = 'fat.jpg'  # Replace with your image path
image = Image.open(image_path)
image = np.array(image)

# Define the number of segments
n_segments = 4

# Perform segmentation
segmented_centroids, cost_values = pso_segment(image, n_segments)
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
plt.title('PSO Segmented Image')
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
plt.plot(range(1, len(cost_values) + 1), cost_values, marker='o', label='Cost Function')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
