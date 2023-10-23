from transformers import AutoModel, AutoConfig, AutoTokenizer
from vncorenlp import VnCoreNLP
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch
from preprocess import Preprocess
from utils import pred_to_label
import json
'''
		Combined_model is a PyTorch neural network model designed for a specific task.
		It takes as input a sequence of tokens, typically represented as input_ids and
		attention_mask, and processes them through a pre-trained transformer model.
		The transformer extracts contextual information from the input sequence.

		Input:
			- input_ids: A sequence of token IDs representing the input text.
			- attention_mask: A mask indicating which tokens in the input are relevant (1) or not (0).

		Output:
			- outputs_classifier: The classification output, which provides the probabilities
			for each class in a multi-class classification problem. It uses the Sigmoid
			activation function for each class to output values between 0 and 1.
			Shape: (batch_size, num_classes)

			- outputs_regressor: The regression output, which provides a set of continuous values
			for each input. It is reshaped to have dimensions (batch_size, num_classes, num_regressor_outputs).
			Shape: (batch_size, num_classes, num_regressor_outputs)

		The model combines hidden states from multiple transformer layers for enhanced
		feature extraction, making it suitable for multi-task problems that involve both
		classification and regression tasks.
'''

# Define constants for magic numbers
EMBEDDING_SIZE = 768
NUM_CLASSES = 6
NUM_REGRESSOR_OUTPUTS = 5
DROP_OUT = 0.1


class Combined_model(nn.Module):
    def __init__(self, checkpoint):
        super(Combined_model, self).__init__()
        self.model = AutoModel.from_config(AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True))
        self.classifier = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE * 4, 256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(256, NUM_CLASSES)
        )
        self.regressor = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE * 4, 256),
            nn.ReLU(),
            nn.Dropout(DROP_OUT),
            nn.Linear(256, NUM_CLASSES*NUM_REGRESSOR_OUTPUTS)
        )
  
    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = torch.cat((outputs[2][-1][:, 0, ...], outputs[2][-2][:, 0, ...], outputs[2][-3][:, 0, ...], outputs[2][-4][:, 0, ...]), -1)
  
        outputs_classifier = self.classifier(outputs)
        outputs_regressor = self.regressor(outputs)
  
        outputs_classifier = nn.Sigmoid()(outputs_classifier)
        outputs_regressor = outputs_regressor.view(-1, 6, 5)
  
        return outputs_classifier, outputs_regressor
