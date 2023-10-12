import numpy as np
import re

def remove_emoji(raw_text):
    emoji = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F700-\U0001F77F"  # alchemical symbols
                        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                        u"\U0001F004-\U0001F0CF"  # Miscellaneous Symbols and Arrows
                        u"\U0001F170-\U0001F251"  # Enclosed Characters
                        u"\u2B05"                  
                        "]+", flags=re.UNICODE)
    return emoji.sub(r'', raw_text)

def remove_special_char(raw_text):
    special_char = re.compile("�+")  
    return special_char.sub(r'', raw_text)

def remove_punctuation(raw_text):
    punctuation = re.compile(r"[!#$%&()*+;<=>?@[\]^_`{|}~]")
    return punctuation.sub(r'', raw_text)

def remove_number(raw_text):
    number = re.compile(" \d+")
    return number.sub(r'', raw_text)

def remove_annotation(raw_text):
    nhan_vien = r'\bnv\b|\bnvien\b'
    khach_san = r'\bks\b|\bksan\b'
    return re.sub(nhan_vien, "nhân viên", re.sub(khach_san, "khách sạn", raw_text))

def run(raw_text):
    return remove_annotation(remove_number(remove_punctuation(remove_special_char(remove_emoji(raw_text)))))

def clean_text(text):
    return {"Review": run(text["Review"].lower())}

class Preprocess():
    def __init__(self, tokenizer, rdrsegmenter):
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
        self.feature = ['giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam']

    def segment(self, example):
        return {"Segment": " ".join([" ".join(sen) for sen in self.rdrsegmenter.tokenize(example["Review"])])}
 
    def tokenize(self, example):
        return self.tokenizer(example["Segment"], truncation=True)
    
    def label(self, example):
        return {'labels_regressor': np.array([example[i] for i in self.feature]),
            'labels_classifier': np.array([int(example[i] != 0) for i in self.feature])}
        
    def run(self, dataset):
        dataset = dataset.map(clean_text)
        dataset = dataset.map(self.segment)
        dataset = dataset.map(self.tokenize, batched=True)
        dataset = dataset.map(self.label)
        dataset = dataset.remove_columns(['Review', 'giai_tri', 'luu_tru', 'nha_hang', 'an_uong', 'di_chuyen', 'mua_sam', 'Segment'])
        dataset.set_format("torch")

        return dataset



    