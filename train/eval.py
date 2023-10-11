from transformers import AutoTokenizer, DataCollatorWithPadding, AdamW, get_scheduler, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from vncorenlp import VnCoreNLP
from tqdm.auto import tqdm
import numpy as np
import torch
from model import Combined_model
from datasets import load_dataset
from preprocess import Preprocess
from model import Combined_model
from metric import *
from loss import *
from utils import *

# Define paths and configurations
TEST_DIR = "/Users/nampham/OneDrive - Đại học FPT- FPT University/Intern/Smart Review Classification/Smart-Review_Analysis/dataset/chall_02_private_test.csv"
MODEL_NAME = "vinai/phobert-base"
MODEL_WEIGHTS = "/Users/nampham/OneDrive - Đại học FPT- FPT University/Intern/Smart Review Classification/Smart-Review_Analysis/train/weights/model_v6.pt"

# Segmenter input
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load test dataset
data_files = {'test': TEST_DIR}
dataset = load_dataset('csv', data_files=data_files)

# Preprocess
preprocess = Preprocess(tokenizer, rdrsegmenter)
tokenized_datasets = preprocess.run(dataset)

# Data loader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

test_dataloader = DataLoader(tokenized_datasets["test"],
                             batch_size=4,
                             collate_fn=data_collator,
                             shuffle=False)

# Load pretrained model
model = Combined_model(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location='cpu'))  
model.eval()  

# Initialize evaluation metrics
val_loss = ScalarMetric()
val_loss_classifier = ScalarMetric()
val_loss_regressor = ScalarMetric()
val_acc = AccuracyMetric()
val_f1_score = F1_score()
val_r2_score = R2_score()
result = None

# Evaluate on the test dataset
for batch in tqdm(test_dataloader):
    inputs = {'input_ids': batch['input_ids'],
              'attention_mask': batch['attention_mask']}
    with torch.no_grad():
        outputs_classifier, outputs_regressor = model(**inputs)
        loss1 = loss_classifier(outputs_classifier, batch['labels_classifier'].float())
        loss2 = loss_softmax(outputs_regressor, batch['labels_regressor'].float(), device='cpu')
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

print("Test Loss:", val_loss.compute(), "Loss Classifier:", val_loss_classifier.compute(), "Loss Regressor:", val_loss_regressor.compute())
print("Acc", val_acc.compute())
print("F1_score", f1_score)
print("R2_score", r2_score)
print("Final_score", final_score)
