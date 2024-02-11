import scipy.sparse
import scipy.spatial    


class TSPEvaluator(object):
    def __init__(self, points):
        self.dist_mat = scipy.spatial.distance_matrix(points, points)

    def evaluate(self, route):
        total_cost = 0
        for i in range(len(route) - 1):
            total_cost += self.dist_mat[route[i], route[i + 1]]
        return total_cost