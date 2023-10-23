from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from vncorenlp import VnCoreNLP
import numpy as np
import torch
from model import Combined_model
from datasets import load_dataset
from preprocess import Preprocess
from tqdm import tqdm
from metric import R2_score, F1_score, ScalarMetric, AccuracyMetric
from loss import loss_classifier, loss_regressor, loss_softmax, sigmoid_focal_loss, bce_loss_weights, CB_loss
from utils import pred_to_label

"""
Input: 
    - Test data in CSV format.
    - Pre-trained model name and model weights.

Process:
    1. Initialize with input data and device setup.
    2. Load test data and preprocess it.
    3. Load the pre-trained model.
    4. Evaluate the model on the test data.
    5. Calculate evaluation metrics, including loss, accuracy, F1 score, R2 score, and a final score.

Output: 
    - Print evaluation metrics.
"""

class SmartReviewEvaluator:
    def __init__(self, test_dir, model_name, model_weights):
        self.test_dir = test_dir
        self.model_name = model_name
        self.model_weights = model_weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.rdrsegmenter = VnCoreNLP("train/VnCoreNLP/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None

    def load_test_data(self):
        data_files = {'test': self.test_dir}
        dataset = load_dataset('csv', data_files=data_files)
        return dataset

    def preprocess_data(self, dataset):
        preprocess = Preprocess(self.tokenizer, self.rdrsegmenter)
        tokenized_datasets = preprocess.run(dataset)
        return tokenized_datasets

    def load_model(self):
        model = Combined_model(self.model_name)
        model.load_state_dict(torch.load(self.model_weights, map_location=self.device))
        model.eval()
        self.model = model

    def evaluate(self):
        test_dataset = self.load_test_data()
        tokenized_test_data = self.preprocess_data(test_dataset)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        test_dataloader = DataLoader(tokenized_test_data["test"], batch_size=4, collate_fn=data_collator, shuffle=False)

        val_loss = ScalarMetric()
        val_loss_classifier = ScalarMetric()
        val_loss_regressor = ScalarMetric()
        val_acc = AccuracyMetric()
        val_f1_score = F1_score()
        val_r2_score = R2_score()
        result = None

        for batch in tqdm(test_dataloader):
            inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            with torch.no_grad():
                outputs_classifier, outputs_regressor = self.model(**inputs)
                loss1 = loss_classifier(outputs_classifier, batch['labels_classifier'].float())
                loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].float(), device=self.device)
                loss = loss1 + loss2
                outputs_classifier = outputs_classifier.cpu().numpy()
                outputs_regressor = outputs_regressor.cpu().numpy()
                outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
                y_true = batch['labels_regressor'].numpy()
                outputs = pred_to_label(outputs_classifier, outputs_regressor)
                result = np.concatenate([result, np.round(outputs)], axis=0) if result is not None else np.round(outputs)
                val_loss_classifier.update(loss1.item())
                val_loss_regressor.update(loss2.item())
                val_loss.update(loss.item())
                val_acc.update(np.round(outputs), y_true)
                val_f1_score.update(np.round(outputs), y_true)
                val_r2_score.update(np.round(outputs), y_true)

        f1_score = val_f1_score.compute()
        r2_score = val_r2_score.compute()
        final_score = (f1_score * r2_score).sum() * 1 / 6

        print("Test Loss:", val_loss.compute(), "Loss Classifier:", val_loss_classifier.compute(),
              "Loss Regressor:", val_loss_regressor.compute())
        print("Acc", val_acc.compute())
        print("F1_score", f1_score)
        print("R2_score", r2_score)
        print("Final_score", final_score)

if __name__ == "__main__":
    TEST_DIR = "dataset/chall_02_private_test.csv"
    MODEL_NAME = "vinai/phobert-base"
    MODEL_WEIGHTS = "./train/weights/model_v3.pt"

    smart_review_evaluator = SmartReviewEvaluator(TEST_DIR, MODEL_NAME, MODEL_WEIGHTS)
    smart_review_evaluator.load_model()
    smart_review_evaluator.evaluate()
