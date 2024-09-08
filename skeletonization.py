import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.spatial.distance import cdist
#load test_sceletonization.png
image = cv2.imread('test_sceletonization.png')

start_time = time.time()

skeleton = cv2.ximgproc.thinning(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

num_labels, labels = cv2.connectedComponents(skeleton)
num_points_per_segment = 1

#get num of points in each segment
points = np.unique(labels, return_counts=True)[1]
num_points_per_segment = np.maximum(np.ceil(points / 300), 1).astype(np.int32)

all_points = []  
for label in range(1, num_labels):  # Start at 1 to skip the background label (0)
    segment_mask = (labels == label).astype(np.uint8)
    segment_points = np.argwhere(segment_mask > 0)
    segment_points = segment_points.reshape(-1, 2)

    if len(segment_points) >= num_points_per_segment[label]:
        # Perform k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, centers = cv2.kmeans(segment_points.astype(np.float32), num_points_per_segment[label], None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # Snap centers to nearest skeleton points (corrected)
        distances = cdist(centers, segment_points)
        closest_point_indices = np.argmin(distances, axis=1)  # Find closest for EACH center
        snapped_centers = segment_points[closest_point_indices]
        all_points.extend(snapped_centers)

    elif len(segment_points) > 0:
        # If too few points for k-means, still include one
        center_index = np.random.choice(len(segment_points)) 
        all_points.append(segment_points[center_index])

centers = np.array(all_points, dtype=np.int32)

#display skeleton overlayed on original image ovverlay the centers
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.imshow(skeleton, cmap='gray', alpha=0.5)
plt.scatter(centers[:, 1], centers[:, 0], c='r', s=100)
plt.axis('off')
plt.show()
