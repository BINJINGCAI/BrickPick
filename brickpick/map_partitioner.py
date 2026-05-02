import numpy as np
import os
import yaml
import cv2
import heapq
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.future import graph
from skimage import color
import math

class MapPartitioner:
    def __init__(self, yaml_path, inflation_radius=3, min_area=500, alpha=1.0, beta=0.002, merge_cost_threshold=1.5):
        self.yaml_path = yaml_path
        self.inflation_radius = inflation_radius
        self.min_area = min_area
        self.alpha = alpha
        self.beta = beta
        self.merge_cost_threshold = merge_cost_threshold

        self._load_map()

    # load the map
    def _load_map(self):
        """load the yaml and pgm file"""
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        pgm_path = os.path.join(os.path.dirname(self.yaml_path), config['image'])
        self.resolution = config['resolution']
        self.origin = config['origin']

        """Read image"""
        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load map:{pgm_path}")
        
        """Binarization"""
        self.occ_map = np.zeros(img, dtype=np.uint8)
        self.occ_map[img > config['occupied_thresh'] * 255] = 1 #occupied
        self.occ_map[img < config['occupied_thresh'] * 255] = 0 #free

        printf(f"Map loaded. Size:{self.occ_map.shape}, Resolution:{self.resolution}")
    
    def _astar(self, start, goal, binary_mask):
        """A* search(8 connected), find out the shortest path from start to goal"""
        if not binary_mask[start[0], start[1]] or not binary_mask[goal[0], goal[1]]:
            return None
        
        def heuristic(a, b):
            """Calculate the heuristic distance between two points"""
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        open_set = []
        heapq.heappush(open_set, (0.0, start)) # (f_score, current)
        came_from = {}
        g_score = {start: 0.0}
        closed_set = set()

        """A stupid way (* ￣︿￣) to set 8 connected neighbors """
        neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
               # Rebuilt the path
               path = [current]
               while current in came_from:
                   current = came_from[current]
                   path.append(current)
               return path

            if current in closed_set:
                continue
            closed_set.add(current)

            for dy, dx in neighbors:
                ny, nx = current[0] + dy, current[1] + dx
                neighbor = (ny, nx)

                if 0 <=ny < binary_mask.shape[0] and 0 <= nx < binary_mask.shape[1]:
                    if binary_mask[ny, nx] == 0 or neighbor in closed_set:
                        continue

                    # diagonal cost
                    move_cost = 1.414 if (dy != 0 and dx !=0) else 1.0
                    tentative_g_score = g_score[current] + move_cost

                    # update the a shorter path, update the f score.
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score, neighbor))
                    
        return None

    def step1_preprocess(self, home_pixel):
        """ Preprocess the map, abstract the freedom space """
        print("Step 1: Preprocess map...")
        self.home_pixel = tuple(home_pixel)

        """Morphological dilation (with safety margin reserved)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.inflation_radius*2+1, self.inflation_radius*2+1))
        free_mask = (self.occ_map == 0).astype(np.uint8)
        inflated_mask = cv2.dilate(free_mask, kernel, iterations=1)

        """ Connected domain analysis: Extract the Free regions connected to Home """
        labeled_array, _ = ndimage.label(inflated_mask)

        if labeled_array[self.home_pixel] == 0:
            raise ValueError(f"Home pixel {self.home_pixel} is not in the free/inflated space!")

        self.free_mask = (labeled_array == labeled_array[self.home_pixel]).astype(np.uint8) # The Free-space mask M
        print(f" -> Extracted reachable free space. Pixels:{np.sum(self.free_mask)}")

    def step2_find_seeds(self, min_peak_distance=20):
        """Find the seed points(predict the brick area)"""
        print("Step 2: Find seeds...")

        # distance transform
        dist_map = ndimage.distance_transform_edt(self.free_mask)
        self.dist_map = dist_map # The distance map M

        """1. Open area center
            Use peak local max to draw the peak of the distance transformation
        """
        from skimage.feature import peak_local_max
        peaks = peak_local_max(dist_map, min_distance=min_peak_distance, labels=self.free_mask)

        """
        Near the obstacle 
        Extract the boundary zone through corrosion
        """
        kernel_small = np.ones((5,5), np.uint8)
        eroded_mask = cv2.erode(self.free_mask, kernel_small, iterations=1)
        boundary_band = self.free_mask - eroded_mask 

        boundary_coords = np.argwhere(boundary_band >0 )
        # 简单的下采样以减少种子数量
        if len(boundary_coords) > 0:
            step = max(1, len(boundary_coords) // 50) 
            boundary_seeds = boundary_coords[::step]
        else:
            boundary_seeds = np.empty((0, 2), dtype=int)
            
        # 合并种子点
        if len(peaks) > 0 and len(boundary_seeds) > 0:
            all_seeds = np.vstack((peaks, boundary_seeds))
        else:
            all_seeds = peaks if len(peaks) > 0 else boundary_seeds
            
        self.seeds = all_seeds
        print(f"  -> Found {len(self.seeds)} seed points.")


