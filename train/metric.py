import numpy as np

def precision(y_pred, y_true):
    true_positive = np.logical_and(y_pred, y_true).sum(axis=0)
    false_positive = np.logical_and(y_pred, ~y_true).sum(axis=0)

    return true_positive / (true_positive + false_positive)

def recall(y_pred, y_true):
    true_positive = np.logical_and(y_pred, y_true).sum(axis=0)
    false_negative = np.logical_and(~y_pred, y_true).sum(axis=0)

    return true_positive / (true_positive + false_negative)


class F1_score:
    def __init__(self):
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def compute(self):
        y_pred = np.vstack(self.y_pred)
        y_true = np.vstack(self.y_true)
        
        f1_score = np.zeros(y_pred.shape[1])
        precision_score = precision(y_pred != 0, y_true != 0)
        recall_score = recall(y_pred != 0, y_true != 0)
        mask_precision_score = np.logical_and(precision_score != 0, ~np.isnan(precision_score))
        mask_recall_score = np.logical_and(recall_score != 0, ~np.isnan(recall_score))
        mask = np.logical_and(mask_precision_score, mask_recall_score)
        
        f1_score[mask] = 2 * (precision_score[mask] * recall_score[mask]) / (precision_score[mask] + recall_score[mask])

        return f1_score

class R2_score:
    def __init__(self):
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def compute(self):
        y_pred = np.vstack(self.y_pred)
        y_true = np.vstack(self.y_true)
        
        mask = np.logical_and(y_pred != 0, y_true != 0)
        rss = (((y_pred - y_true)**2) * mask).sum(axis=0)
        k = (mask * 16).sum(axis=0)
        r2_score = np.ones(rss.shape[0])
        mask2 = (k != 0)
        r2_score[mask2] = 1 - rss[mask2] / k[mask2]

        return r2_score

class ScalarMetric:
    def __init__(self):
        self.scalar = 0
        self.num = 0

    def update(self, scalar):
        self.scalar += scalar
        self.num += 1
        return self

    def compute(self):
        return self.scalar / self.num

    def reset(self):
        self.scalar = 0
        self.num = 0

class AccuracyMetric:
    def __init__(self):
        self.correct = 0
        self.num = 0

    def update(self, y_pred, y_true):
        self.correct += np.sum(y_pred == y_true)
        self.num += y_pred.size

    def compute(self):
        return self.correct / self.num

    def reset(self):
        self.correct = 0
        self.num = 0

if __name__ == "__main__":
    y_true = np.array([[1, 0, 3, 2, 4, 5],
                   [0, 1, 2, 3, 4, 5],
                   [5, 4, 3, 2, 1, 0]])

    y_pred = np.array([[1, 0, 2, 1, 3, 4],
                   [0, 1, 3, 2, 4, 5],
                   [4, 3, 2, 1, 0, 5]])

    scalar_metric = ScalarMetric()
    accuracy_metric = AccuracyMetric()
    f1_metric = F1_score()
    r2_metric = R2_score()

    scalar_metric.update(42)
    accuracy_metric.update(y_pred, y_true)
    f1_metric.update(y_pred, y_true)
    r2_metric.update(y_pred, y_true)

    scalar_result = scalar_metric.compute()
    accuracy_result = accuracy_metric.compute()
    f1_result = f1_metric.compute()
    r2_result = r2_metric.compute()

    print("Scalar Metric:", scalar_result)
    print("Accuracy Metric:", accuracy_result)
    print("F1 Score Metric:", sum(f1_result)/len(f1_result))
    print("R2 Score Metric:", r2_result)



