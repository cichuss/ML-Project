import random
import warnings

import matplotlib
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning, message="Precision is ill-defined")
matplotlib.use('TkAgg')


def create_random_nd_hierarchy():
    random.seed(1234)

    classes = ['rock', 'jazz', 'blues', 'pop', 'funk', 'country', 'kids', 'opera', 'electronic',
               'heavy-metal', 'classical']

    def build_hierarchy(classes_subset):
        if len(classes_subset) == 1:
            return None  # leaf — no further splitting

        # Split in half
        sorted_subset = sorted(classes_subset)
        random.shuffle(sorted_subset)
        mid = len(sorted_subset) // 2
        left = sorted(sorted_subset[:mid])
        right = sorted(sorted_subset[mid:])

        node = {'left': left, 'right': right}
        hierarchy[tuple(sorted(classes_subset))] = node

        # Recursively split left and right
        build_hierarchy(left)
        build_hierarchy(right)

        return node

    hierarchy = {}
    # Create the main split stored under the 'root' key
    sorted_classes = sorted(classes)
    random.shuffle(sorted_classes)
    mid = len(sorted_classes) // 2
    left_root = sorted(sorted_classes[:mid])
    right_root = sorted(sorted_classes[mid:])
    root_node = {'left': left_root, 'right': right_root}
    hierarchy['root'] = root_node

    # Recursively split the left and right side
    build_hierarchy(left_root)
    build_hierarchy(right_root)

    return hierarchy
