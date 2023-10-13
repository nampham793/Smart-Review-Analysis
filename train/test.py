import torch
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP
from model import Combined_model
from utils import pred_to_label  

# Load the trained model
model = Combined_model("vinai/phobert-base") 
model.load_state_dict(torch.load("/Users/nampham/OneDrive - Đại học FPT- FPT University/Intern/Smart Review Classification/Smart-Review_Analysis/train/weights/model_v6.pt", map_location='cpu'))  

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
review_sentence = "Đồ ăn ngon thực đơn đa dạng. Quán sạch và gọn gàng thích hợp cho gia đình hội tụ dịp cuối tuần. Sẽ quay lại."
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
tokenized_input = tokenizer(review_sentence, return_tensors="pt", padding=True, truncation=True)

# Remove 'token_type_ids' from tokenized_input
tokenized_input.pop('token_type_ids', None)

# Make predictions
with torch.no_grad():
    outputs_classifier, outputs_regressor = model(**tokenized_input)
    outputs_classifier = outputs_classifier.cpu().numpy()
    outputs_regressor = outputs_regressor.cpu().numpy()
    outputs_regressor = outputs_regressor.argmax(axis=-1) + 1
   
	# Convert output to label
    outputs = pred_to_label(outputs_classifier, outputs_regressor)

    print("Giai tri: ", outputs[0][0])
    print("Luu tru: ", outputs[0][1])
    print("Nha hang: ", outputs[0][2])
    print("An uong: ", outputs[0][3])
    print("Di chuyen: ", outputs[0][4])
    print("Mua sam: ", outputs[0][5])

