import numpy as np

def fast_hist(pred, label, n):
    """
    Compute confusion matrix using bincount for speed.
    """
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k], minlength=n**2).reshape(n, n)

class IoUCalculator:
    """
    Accumulate confusion matrix and compute IoU/mIoU.
    Supports dynamic number of classes.
    """
    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, gt):
        """
        Update internal confusion matrix with new batch/room data.
        """
        self.confusion_matrix += fast_hist(pred.flatten(), gt.flatten(), self.num_classes)

    def compute(self):
        """
        Returns:
            ious: (num_classes,) array of IoU per class
            miou: scalar mIoU
        """
        diag = np.diag(self.confusion_matrix)
        # axis 1 is predicted, axis 0 is truth (standard convention)
        # row_sum = ground truth count, col_sum = prediction count
        row_sum = self.confusion_matrix.sum(axis=1)
        col_sum = self.confusion_matrix.sum(axis=0)
        
        union = row_sum + col_sum - diag
        ious = diag / (union + 1e-6)
        miou = np.nanmean(ious)
        return ious, miou

def calc_local_miou(pred, gt, num_classes=13):
    """
    Quick helper for single-room mIoU calculation.
    """
    hist = fast_hist(pred.flatten(), gt.flatten(), num_classes)
    diag = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - diag
    ious = diag / (union + 1e-6)
    return np.nanmean(ious)