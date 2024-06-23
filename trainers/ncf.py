from .base import AbstractTrainer
import torch
import torch.nn as nn
import numpy as np


class NCFTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.BCEWithLogitsLoss()

    @classmethod
    def code(cls):
        return 'ncf'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        users, seqs, labels = batch
        logits = self.model(users, seqs)  # B x T x V

        logits = logits.view(-1, logits.size(-1))  # (B*T) x V
        # labels = labels.view(-1)  # B*T
        loss = self.ce(logits, labels)
        return loss

    def calculate_metrics(self, batch):
        return self.metrics(batch)

    def hit(self, gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

    def ndcg(self, gt_item, pred_items):
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            return np.reciprocal(np.log2(index + 2))
        return 0

    def metrics(self, batch):
        HR, NDCG = {}, {}
        metrics = {}
        top_k = self.metric_ks
        for k in top_k:
            HR[str(k)] = []
            NDCG[str(k)] = []
        user, item, label = batch
        predictions = self.model(user, item)
        _, indices = torch.topk(predictions, max(top_k))
        recommends = torch.take(
            item, indices).cpu().numpy().tolist()

        for item_one, recommend in zip(item.cpu().numpy().tolist(), recommends):
            gt_item = item_one[0]
            for k in top_k:
                HR[str(k)].append(self.hit(gt_item, recommends))
                NDCG[str(k)].append(self.ndcg(gt_item, recommends))

        for k in top_k:
            metrics['Recall@%d' % k], metrics['NDCG@%d' % k] = np.mean(HR[str(k)]), np.mean(NDCG[str(k)])
        return metrics
