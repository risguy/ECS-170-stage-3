'''
Concrete Evaluate class reporting accuracy together with macro precision / recall / F1
'''

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class Evaluate_Metrics(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        true_y, pred_y = self.data['true_y'], self.data['pred_y']
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_y, pred_y, average='macro', zero_division=0)
        return {'accuracy': accuracy_score(true_y, pred_y),
                'precision': precision, 'recall': recall, 'f1': f1}
