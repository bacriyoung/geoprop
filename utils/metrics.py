import numpy as np
from sklearn.metrics import confusion_matrix

def fast_hist(pred, label, n):
    """
    Fast confusion matrix calculation using bincount.
    """
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)

class IoUCalculator:
    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, gt):
        """
        Update internal state with new batch/room predictions.
        """
        # Ensure input consistency
        pred = pred.flatten()
        gt = gt.flatten()
        self.confusion_matrix += fast_hist(pred, gt, self.num_classes)

    def compute(self):
        """
        Returns:
            oa (float): Overall Accuracy (Purity of pseudo-labels)
            miou (float): Mean Intersection over Union (Balanced quality)
            ious (np.array): Per-class IoU
        """
        diag = np.diag(self.confusion_matrix)
        row_sum = self.confusion_matrix.sum(axis=1) # GT count
        col_sum = self.confusion_matrix.sum(axis=0) # Pred count
        
        # 1. Per-class IoU
        union = row_sum + col_sum - diag
        ious = diag / (union + 1e-6)
        
        # 2. Mean IoU (The King Metric)
        miou = np.nanmean(ious)
        
        # 3. Overall Accuracy (The Purity Check)
        total_correct = diag.sum()
        total_valid = self.confusion_matrix.sum()
        oa = total_correct / (total_valid + 1e-6)
        
        return oa, miou, ious

def calc_local_metrics(pred, gt, num_classes=13):
    """
    Helper for single-room metrics. Returns both OA and mIoU.
    """
    hist = fast_hist(pred.flatten(), gt.flatten(), num_classes)
    diag = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - diag
    ious = diag / (union + 1e-6)
    
    miou = np.nanmean(ious)
    oa = diag.sum() / (hist.sum() + 1e-6)
    
    return oa, miou