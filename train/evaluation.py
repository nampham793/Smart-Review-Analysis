import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
from model import Combined_model
from utils import pred_to_label
from metric import F1_score, R2_score, AccuracyMetric
from tqdm import tqdm


class Evaluation:
    def __init__(self, data_path, model_path):
        self.data = data_path
        self.model_path = model_path

    def get_data(self):
        model = Combined_model("vinai/phobert-base")
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        data = pd.read_csv(self.data)

        reviews = data['Review']
        categories = data.columns[1:]

        rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

        predicted_labels = []
  
        for review in tqdm(reviews):
            tokenized_input = tokenizer(review, return_tensors="pt", padding=True, truncation=True)
            tokenized_input.pop('token_type_ids', None)

            with torch.no_grad():
                outputs_classifier, outputs_regressor = model(**tokenized_input)
                outputs_classifier = outputs_classifier.cpu().numpy()
                outputs_regressor = outputs_regressor.cpu().numpy()
                outputs_regressor = outputs_regressor.argmax(axis=-1) + 1

                outputs = pred_to_label(outputs_classifier, outputs_regressor)
                
                predicted_labels.append(outputs)


        predicted_labels_flat = np.concatenate(predicted_labels, axis=0)
        predicted_df = pd.DataFrame(predicted_labels_flat, columns=categories)
        result_df = pd.concat([data['Review'], predicted_df], axis=1)

        return result_df

    def evaluation(self, predicted_data, true_data):
        f1_scores = {}
        r2_scores = {}
        accuracy_scores = {}

        for category in true_data.columns[1:]:
            f1 = F1_score()
            r2 = R2_score()
            accuracy = AccuracyMetric()

            f1.update(predicted_data[category], true_data[category])
            r2.update(predicted_data[category], true_data[category])
            accuracy.update(predicted_data[category], true_data[category])

            f1_scores[category] = f1.compute()
            r2_scores[category] = r2.compute()
            accuracy_scores[category] = accuracy.compute()

        return f1_scores, r2_scores, accuracy_scores

def run():
    DATA_DIR = "/Users/nampham/OneDrive - Đại học FPT- FPT University/Intern/Smart Review Classification/Smart-Review_Analysis/dataset/chall_02_private_test.csv"
    MODEL_DIR = "/Users/nampham/OneDrive - Đại học FPT- FPT University/Intern/Smart Review Classification/Smart-Review_Analysis/train/weights/model_v3.pt"

    eval = Evaluation(DATA_DIR, MODEL_DIR)
    result_df = eval.get_data()
    true_df = pd.read_csv(DATA_DIR)

    true_data = true_df.drop('Review', axis=1)
    predicted_data = result_df.drop('Review', axis=1)

    f1_scores, r2_scores, accuracy_scores = eval.evaluation(predicted_data, true_data)

    f1_score = sum(f1_scores.values())/len(f1_scores.values())
    r2_score = sum(r2_scores.values())/len(r2_scores.values())
    accuracy_score = sum(accuracy_scores.values())/len(accuracy_scores)

    print("F1 Scores:", sum(f1_score)/len(f1_score))
    print("R2 Scores:", sum(r2_score)/len(r2_score))
    print("Accuracy Score:", accuracy_score)
    print("Accuracy Scores for each aspect:", accuracy_scores)

if __name__ == "__main__":
    run()
