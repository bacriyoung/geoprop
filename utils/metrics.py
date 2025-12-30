import numpy as np
from sklearn.metrics import confusion_matrix

class IoUCalculator:
    def __init__(self, num_classes=13):
        self.num_classes = num_classes
        self.cm = np.zeros((num_classes, num_classes))

    def update(self, pred, gt):
        mask = (gt >= 0) & (gt < self.num_classes) & (pred >= 0) & (pred < self.num_classes)
        if mask.sum() > 0:
            self.cm += confusion_matrix(gt[mask], pred[mask], labels=range(self.num_classes))

    def compute(self):
        tp = np.diag(self.cm)
        union = self.cm.sum(0) + self.cm.sum(1) - tp
        iou = tp / (union + 1e-6)
        return iou, np.nanmean(iou), tp.sum() / (self.cm.sum() + 1e-6)

def calc_local_miou(pred, gt, num_classes=13):
    mask = (pred != -100) & (gt != -100)
    if mask.sum() == 0: return 0.0
    
    p, g = pred[mask], gt[mask]
    cm = confusion_matrix(g, p, labels=range(num_classes))
    tp = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - tp
    iou = tp / (union + 1e-6)
    return np.nanmean(iou)