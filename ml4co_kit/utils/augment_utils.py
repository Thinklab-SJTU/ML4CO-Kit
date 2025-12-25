r"""
Augmentation Utilities
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from typing import Sequence


###############################################
#             Points Augmentation             #
###############################################      


def _rotation_points(points: np.ndarray) -> np.ndarray:
    # Random rotation angle
    angle = np.random.uniform(0, 2 * np.pi)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    
    # Rotation matrix
    rotation_matrix = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])
    
    # Choose the centroid as the center point
    center = points.mean(axis=0)
    
    # Apply rotation
    points = (points - center) @ rotation_matrix.T + center
    return points


def _translation_points(
    points: np.ndarray, translation_scale: float = 0.01
) -> np.ndarray:
    # Random translation
    translation = np.random.uniform(
        low=-translation_scale, high=translation_scale, size=2
    )
    points = points + translation
    return points


def _normalize_points(points: np.ndarray) -> np.ndarray:
    # Get the minimum and maximum coordinates
    min_coords = points.min()
    max_coords = points.max()
    range_coords = max_coords - min_coords
    
    # Normalize the points to the range [0, 1]
    points = (points - min_coords) / range_coords
    return points


def _flip_points(points: np.ndarray) -> np.ndarray:
    # 0: horizontal flip, 1: vertical flip
    flip_axis = np.random.choice([0, 1])  

    # Flip the points
    points[:, flip_axis] = 1 - points[:, flip_axis]
    return points


def points_augment(
    points: np.ndarray,
    rotation_probs: float = 0.3,
    translation_probs: float = 0.3,
    normalize: bool = True,
    flip_probs: float = 0.3,
) -> np.ndarray:
    """Augment the points."""
    # Rotation
    if np.random.random() < rotation_probs:
        points = _rotation_points(points)
    
    # Translation
    if np.random.random() < translation_probs:
        points = _translation_points(points)
    
    # Normalize
    if normalize:
        points = _normalize_points(points)
    
    # Flip
    if np.random.random() < flip_probs:
        points = _flip_points(points)
    
    # Return the augmented points
    return points


###############################################
#              Graph Augmentation             #
###############################################


def _generate_isomorphic_mapping(nodes_num: int) -> np.ndarray:
    # Generate a random permutation
    permutation = np.random.permutation(nodes_num)
    
    # Generate the isomorphic mapping
    mapping = np.zeros(shape=(nodes_num, nodes_num), dtype=np.int32)
    mapping[np.arange(0, nodes_num, dtype=np.int32), permutation] = 1
    return mapping, permutation


def graph_augment(
    graph: np.ndarray, 
    sol: np.ndarray = None, 
    ref_sol: np.ndarray = None,
    nodes_weight: np.ndarray = None,
) -> Sequence[np.ndarray]:
    """Augment the graph and sol."""
    # Generate the isomorphic mapping
    mapping, permutation = _generate_isomorphic_mapping(graph.shape[0])
    
    # Augment the graph
    graph = mapping.T @ graph @ mapping
    
    # Augment the sol
    if sol is not None:
        sol = sol[permutation]
    
    # Augment the ref_sol
    if ref_sol is not None:
        ref_sol = ref_sol[permutation]
    
    # Augment the nodes_weight
    if nodes_weight is not None:
        nodes_weight = nodes_weight[permutation]
    
    # Return the augmented graph, sol, and ref_sol
    return graph, sol, ref_sol