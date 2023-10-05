import re
import os
import numpy as np
import pandas as pd

class PreProcess():
    def remove_emoji(self, raw_text):
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

    def remove_special_char(self, raw_text):
        special_char = re.compile("�+")  
        return special_char.sub(r'', raw_text)

    def remove_punctuation(self, raw_text):
        punctuation = re.compile(r"[!#$%&()*+;<=>?@[\]^_`{|}~]")
        return punctuation.sub(r'', raw_text)

    def remove_number(self, raw_text):
        number = re.compile(" \d+")
        return number.sub(r'', raw_text)

    def remove_annotation(self, raw_text):
        nhan_vien = r'\bnv\b|\bnvien\b'
        khach_san = r'\bks\b|\bksan\b'
        return re.sub(nhan_vien, "nhân viên", re.sub(khach_san, "khách sạn", raw_text))

    def prep(self, raw_text):
        return self.remove_annotation(self.remove_number(self.remove_punctuation(self.remove_special_char(self.remove_emoji(raw_text)))))

if __name__ == "__main__":
    raw_text = "Hello World !😚0123456789 @Adfsd2 ks ksan nv �"
    preprocess = PreProcess()
    print(raw_text)
    cleaned = preprocess.prep(raw_text=raw_text)
    print(cleaned)


        
