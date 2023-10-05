from transformers import AutoModel, AutoConfig, AutoTokenizer
from vncorenlp import VnCoreNLP
import torch.nn as nn
import numpy as np
import torch
from preprocess import Preprocess
from utils import pred_to_label


class Regressor_Model(nn.Module):
    def __init__(self, checkpoint, num_outputs):
        super(CustomModelRegressor, self).__init__()
        self.num_outputs = num_outputs
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(
            checkpoint, output_attentions=True, output_hidden_states=True))
        for parameter in self.model.parameters():
            parameter.requires_grad = False  # Corrected the spelling of "requires_grad"
        self.dropout = nn.Dropout(0.1)
        self.output1 = nn.Linear(768 * 4, 6)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        outputs = torch.cat((outputs[2][-1][:, 0, ...], outputs[2][-2][:, 0, ...],
                            outputs[2][-3][:, 0, ...], outputs[2][-4][:, 0, ...]), -1)
        outputs = self.dropout(outputs)
        outputs = self.output1(outputs)
        outputs = nn.Sigmoid()(outputs) * 5

        return outputs


class Classifier_Model(nn.Module):
    def __init__(self, checkpoint, num_outputs):
        super(CustomModelClassifier, self).__init__()
        self.num_outputs = num_outputs
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(
            checkpoint, output_attentions=True, output_hidden_states=True))
        for parameter in self.model.parameters():
            parameter.requires_grad = False  # Corrected the spelling of "requires_grad"
        self.dropout = nn.Dropout(0.1)
        self.output1 = nn.Linear(768 * 4, 30)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        outputs = torch.cat((outputs[2][-1][:, 0, ...], outputs[2][-2][:, 0, ...],
                            outputs[2][-3][:, 0, ...], outputs[2][-4][:, 0, ...]), -1)
        outputs = self.dropout(outputs)
        outputs = self.output1(outputs)

        return outputs


class Combined_model(nn.Module):
    def __init__(self, checkpoint):
        super(Combined_model, self).__init__()
        self.model = model = AutoModel.from_config(AutoConfig.from_pretrained(
            checkpoint, output_attentions=True, output_hidden_states=True))

        # Layers:
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768 * 4, 512)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 6)  # Classification layer
        self.fc3 = nn.Linear(512, 30)  # Regression layer

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.cat((outputs[2][-1][:,0, ...],outputs[2][-2][:,0, ...], outputs[2][-3][:,0, ...], outputs[2][-4][:,0, ...]),-1)
        
        outputs = self.dropout1(outputs)

        # Apply additional layers
        outputs = self.fc1(outputs)
        outputs = self.relu(outputs)
        outputs = self.dropout2(outputs)

        # Classifier output
        outputs_classifier = self.fc2(outputs)
        outputs_classifier = nn.Sigmoid()(outputs_classifier)

        # Regressor output
        outputs_regressor = self.fc3(outputs)
        outputs_regressor = outputs_regressor.reshape(-1, 6, 5)

        return outputs_classifier, outputs_regressor
