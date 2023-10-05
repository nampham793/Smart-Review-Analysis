import torch
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
from model import Combined_model
from utils import pred_to_label  

# Load the trained model
model = Combined_model("vinai/phobert-base") 
model.load_state_dict(torch.load("/Users/nampham/OneDrive - Đại học FPT- FPT University/Intern/Smart Review Classification/Smart-Review_Analysis/train/weights/model.pt", map_location='cpu'))  # Load the trained weights

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
review_sentence = "trải nghiệm mua sắm tại chợ là tuyệt vời.hay nan lại vài giây để có một trái nghiệm thú vị"
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
tokenized_input = tokenizer(review_sentence, return_tensors="pt", padding=True, truncation=True)

# Remove 'token_type_ids' from tokenized_input
tokenized_input.pop('token_type_ids', None)

# Make predictions
with torch.no_grad():
    outputs_classifier, outputs_regressor = model(**tokenized_input)
    result = pred_to_label(outputs_classifier, outputs_regressor)
    print(result)
