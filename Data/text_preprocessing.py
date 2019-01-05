import os
import re

def main():
    directory_in_str = './Project/raw/'
    directory_out_str = './Project/processed/'
    directory = os.fsencode(directory_in_str)
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        f = open(directory_in_str + filename, 'r', encoding='UTF-8')
        f_out = open(directory_out_str + 'p_' + filename, 'w', encoding='UTF-8')
        f_out.write(preprocess(f.read().strip()))
        f.close()
        f_out.close()

def preprocess(text):
    new_text = re.sub(r"[^–a-zA-ZäöåÄÖÅöåéé0-9\,\.\-\s]+", "", text) # remove weird characters
    new_text = re.sub(r"(?<=\n) +", "", new_text) # remove leading spaces
    new_text = re.sub(r"(?<=\n)[0-9]+", "", new_text) # remove page numbers
    new_text = re.sub("\n\s*\n", "\n", new_text) # remove multiple line breaks
    new_text = re.sub("[\.\?\!] *", "\n", new_text) # break lines at dots
    new_text = re.sub(r"\n(?=[a-zäöåöåéé])", " ", new_text) # remove incorrect line breaks
    new_text = re.sub("\n\s*\n", "\n", new_text) # remove multiple line breaks, new ones appear from line break
    new_text = re.sub(r"(?<=\n)[0-9] +", "", new_text) # remove leading numbers
    new_text = re.sub("\s\s", " ", new_text) # remove multiple whitespaces

    return new_text.strip()

