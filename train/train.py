import json
from transformers import AutoTokenizer, DataCollatorWithPadding, AdamW, get_scheduler
from torch.utils.data import DataLoader
from vncorenlp import VnCoreNLP
from tqdm.auto import tqdm
import numpy as np
import torch
import random
from datasets import load_dataset
from preprocess import Preprocess
from model import Combined_model
from metric import R2_score, F1_score, ScalarMetric, AccuracyMetric
from loss import loss_classifier, loss_regressor, loss_softmax, sigmoid_focal_loss, bce_loss_weights, CB_loss
from utils import pred_to_label

"""
Input: 
    - Training and testing data in CSV format, specified by file paths (TRAIN_DIR and TEST_DIR).
    - Pre-trained language model (PhoBERT) for tokenization and feature extraction.
    - Vietnamese text segmentation tool (VnCoreNLP) for tokenization.

Output:
    - Trained model with the best weights saved to a file (model_v6.pt).
    - Model performance metrics, including test loss, accuracy, F1 score, R2 score, and a final score.
    - Output scores are used to determine the best model, which is saved with the highest final score.
"""
def load_config(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config

class Smart_Analysis_Training:
    def __init__(self, config):
        self.train_dir = config['TRAIN_DIR']
        self.test_dir = config['TEST_DIR']
        self.save_weight_dir = config['SAVE_WEIGHT_DIR']
        self.model_name = config['MODEL_NAME']
        self.seed = config['SEED']
        self.device = config['DEVICE']
        self.num_epochs = config['NUM_EPOCHS']
        self.best_score = -1

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def load_data(self):
        data_files = {'train': self.train_dir, 'test': self.test_dir}
        dataset = load_dataset('csv', data_files=data_files)
        return dataset

    def preprocess_data(self, dataset, tokenizer, rdrsegmenter):
        preprocess = Preprocess(tokenizer, rdrsegmenter)
        tokenized_datasets = preprocess.run(dataset)
        return tokenized_datasets

    def create_data_loaders(self, tokenized_datasets, batch_size=32):
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        train_dataloader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, collate_fn=data_collator, shuffle=True)
        test_dataloader = DataLoader(tokenized_datasets["test"], batch_size=batch_size, collate_fn=data_collator)
        return train_dataloader, test_dataloader

    def create_model(self):
        model = Combined_model(self.model_name)
        model.to(self.device)
        return model

    def train_model(self, model, train_dataloader, optimizer, num_epochs, lr_scheduler):
        pb_train = tqdm(range(num_epochs * len(train_dataloader)))
        for epoch in range(num_epochs):
            train_loss = 0
            model.train()
            for batch in train_dataloader:
                inputs = {'input_ids': batch['input_ids'].to(self.device), 'attention_mask': batch['attention_mask'].to(self.device)}
                outputs_classifier, outputs_regressor = model(**inputs)
                loss1 = sigmoid_focal_loss(outputs_classifier, batch['labels_classifier'].to(self.device).float(), alpha=-1, gamma=1, reduction='mean')
                loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].to(self.device).float(), self.device)
                loss = 10 * loss1 + loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                pb_train.update(1)
                pb_train.set_postfix(loss_classifier=loss1.item(), loss_regressor=loss2.item(), loss=loss.item())
                train_loss += loss.item() / len(train_dataloader)
            print("Train Loss:", train_loss)

    def evaluate_model(self, model, test_dataloader):
        pb_test = tqdm(range(self.num_epochs * len(test_dataloader)))
        val_loss = ScalarMetric()
        val_loss_classifier = ScalarMetric()
        val_loss_regressor = ScalarMetric()
        val_acc = AccuracyMetric()
        val_f1_score = F1_score()
        val_r2_score = R2_score()
        num = 0
        correct = 0
        result = None
        model.eval()
        for batch in test_dataloader:
            inputs = {'input_ids': batch['input_ids'].to(self.device), 'attention_mask': batch['attention_mask'].to(self.device)}
            with torch.no_grad():
                outputs_classifier, outputs_regressor = model(**inputs)
                loss1 = loss_classifier(outputs_classifier, batch['labels_classifier'].to(self.device).float())
                loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].to(self.device).float(), self.device)
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
                pb_test.update(1)

        f1_score = val_f1_score.compute()
        r2_score = val_r2_score.compute()
        final_score = (f1_score * r2_score).sum() * 1 / 6

        if final_score > self.best_score:
            self.best_score = final_score
            torch.save(model.state_dict(), self.save_weight_dir)

        print("Test Loss:", val_loss.compute(), "Loss Classifier:", val_loss_classifier.compute(), "Loss Regressor:", val_loss_regressor.compute())
        print("Acc", val_acc.compute())
        print("F1_score", f1_score)
        print("R2_score", r2_score)
        print("Final_score", final_score)
        print("Best_score", self.best_score)

if __name__ == "__main__":
    config = load_config('train_config.json')
    tokenizer = AutoTokenizer.from_pretrained(config['MODEL_NAME'])
    rdrsegmenter = VnCoreNLP(config['VNCORENLP'], annotators="wseg", max_heap_size='-Xmx500m')
    text_classifier = Smart_Analysis_Training(config)
    text_classifier.set_seed()
    dataset = text_classifier.load_data()
    tokenized_datasets = text_classifier.preprocess_data(dataset, tokenizer, rdrsegmenter)
    train_dataloader, test_dataloader = text_classifier.create_data_loaders(tokenized_datasets)
    model = text_classifier.create_model()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = text_classifier.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    text_classifier.train_model(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=10,
        lr_scheduler=lr_scheduler
    )

