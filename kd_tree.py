from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import heapq



@dataclass
class KDNode:
    "Κόμβος ενός k-d tree."

    point: Sequence[float]          #to simeio ston k-diastato xoro
    index: int                      #index tou simeiou sto arxiko array
    axis: int                       #poio dimension xrismipoiitai gia to split
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTree:
    "Απλή υλοποίηση k-d tree με median split."

    def __init__(self, points: Sequence[Sequence[float]]):
        "Αρχικοποιεί KD-Tree από λίστα k-διάστατων σημείων."

        self.n_points = len(points)
        self.k = len(points[0]) if self.n_points > 0 else 0

        self._points = points

        if self.n_points == 0:
            self.root = None
        else:
            indices = list(range(self.n_points))
            self.root = self._build(indices, depth=0)

    def _build(self, indices: List[int], depth: int) -> Optional[KDNode]:
        "Αναδρομική κατασκευή KD-Tree με median split"

        if not indices:
            return None

        axis = depth % self.k

        #taxinomisi ton indexes me vasi tin timi tou axona
        indices.sort(key=lambda i: self._points[i][axis])

        #epilogi median gia isorropimeno dentro
        median_pos = len(indices) // 2
        median_index = indices[median_pos]
        median_point = self._points[median_index]

        #node creation
        node = KDNode(
            point=median_point,
            index=median_index,
            axis=axis,
        )

        #ftiaxnoume anadromika ta ipodentra
        left_indices = indices[:median_pos]
        right_indices = indices[median_pos + 1 :]

        node.left = self._build(left_indices, depth + 1)
        node.right = self._build(right_indices, depth + 1)

        return node

    def is_empty(self) -> bool:
        "Επιστρέφει True αν το δέντρο είναι άδειο."
        return self.root is None

    def __len__(self) -> int:
        "Επιστρέφει πόσα σημεία περιέχει το δέντρο."
        return self.n_points

    def range_query(
        self,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
    ) -> List[int]:
        "Επιστρέφει indices σημείων εντός του hyper-rectangle [lower_bounds, upper_bounds]."

        if self.root is None or self.k == 0:
            return []

        if len(lower_bounds) != self.k or len(upper_bounds) != self.k:
            raise ValueError("Bounds must have length equal to k dimensions")

        results: List[int] = []
        self._range_query(self.root, lower_bounds, upper_bounds, results)
        return results

    def _range_query(
        self,
        node: Optional[KDNode],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        results: List[int],
    ) -> None:
        "Αναδρομική συνάρτηση για range query."
        if node is None:
            return

        point = node.point
        axis = node.axis

        #checkaroume ean to simeio einai mesa sto range se OLA TA DIMENSIONS
        inside = True
        for dim in range(self.k):
            if point[dim] < lower_bounds[dim] or point[dim] > upper_bounds[dim]:
                inside = False
                break

        if inside:
            results.append(node.index)

        #aristero ipodentro
        if node.left is not None and lower_bounds[axis] <= point[axis]:
            self._range_query(node.left, lower_bounds, upper_bounds, results)

        #dexio ipodentro
        if node.right is not None and upper_bounds[axis] >= point[axis]:
            self._range_query(node.right, lower_bounds, upper_bounds, results)
        
    
    # kNN (k Nearest Neighbors)
    def knn_query(
        self,
        query_point: Sequence[float],
        k: int = 5,
        return_distances: bool = False,
    ) -> List[int] | List[Tuple[int, float]]:
        """Επιστρέφει τους k κοντινότερους γείτονες του query_point (Euclidean).
Αν return_distances=True επιστρέφει (index, distance), αλλιώς μόνο indices."""

        if self.root is None or self.k == 0 or k <= 0:
            return []

        if len(query_point) != self.k:
            raise ValueError("query_point must have length equal to k dimensions")

        k = min(k, self.n_points)

        # max-heap me ( -dist_sq, index ) gia na kratame toys k kalyterous
        best: List[Tuple[float, int]] = []
        self._knn_search(self.root, query_point, k, best)

        # taksinomisi apo pio kontino se mio makrino
        best_sorted = sorted(((-d2, idx) for d2, idx in best), key=lambda x: x[0])

        if return_distances:
            return [(idx, dist ** 0.5) for dist, idx in best_sorted]
        return [idx for dist, idx in best_sorted]

    def _knn_search(
        self,
        node: Optional[KDNode],
        query_point: Sequence[float],
        k: int,
        best: List[Tuple[float, int]],
    ) -> None:
        "Αναδρομική αναζήτηση kNN με pruning."
        if node is None:
            return

        # apostasi tou node.point apo query
        d2 = 0.0
        p = node.point
        for dim in range(self.k):
            diff = p[dim] - query_point[dim]
            d2 += diff * diff

        # enimerosi heap 
        if len(best) < k:
            heapq.heappush(best, (-d2, node.index))
        else:
            # an brikame kalytero apo ton xeirotero, antikatastasi
            worst_neg_d2, _ = best[0]
            if -d2 > worst_neg_d2:
                heapq.heapreplace(best, (-d2, node.index))

        axis = node.axis
        diff_axis = query_point[axis] - p[axis]

        # prwta pame sto pio kontino ypodentro
        near = node.left if diff_axis < 0 else node.right
        far = node.right if diff_axis < 0 else node.left
        self._knn_search(near, query_point, k, best)

        # pruning
        if len(best) < k:
            self._knn_search(far, query_point, k, best)
        else:
            worst_d2 = -best[0][0]
            if diff_axis * diff_axis <= worst_d2:
                self._knn_search(far, query_point, k, best)
        

