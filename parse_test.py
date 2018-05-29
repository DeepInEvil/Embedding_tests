import os
import re
import numpy as np
import xml.etree.ElementTree as ET


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == '__main__':
    main_dir = '/data/dchaudhu/ESWC_challenge/'
    test_reviews = open(main_dir + 'data' + '/test.tsv', 'w')
    tree = ET.parse(main_dir + '/task1_testset.xml')
    root = tree.getroot()
    for ele in root:
        test_reviews.write(ele.attrib['id'] + ', ' + clean_str(ele[3].text) + '\n')

    test_reviews.close()
