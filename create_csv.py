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
    #done_process = ['Toys_Games', 'Tools_Home_Improvement', 'Books', 'Amazon_Instant_Video', 'Movies_TV', 'Video_Games', 'Electronics', 'Health', 'Shoes', 'Baby', 'Automotive', 'Software', 'Sports_Outdoors']
    all_reviews = open('reviews.txt', 'w')
    domains = [d for d in os.listdir(main_dir) if os.path.isdir(d)]
    for domain in domains:
            print ("Processing data from domain:" + domain)
            dom_dir = main_dir+domain
            xmls = os.listdir(dom_dir)
            pos = open(dom_dir+'/'+'positive.tsv', 'w')
            neg = open(dom_dir+'/'+'negative.tsv', 'w')
            for xm in xmls:
                print ('Processing data for file:' + xm)
                tree = ET.parse(dom_dir + '/' + xm)
                root = tree.getroot()
                if 'neg' in xm:
                    for ele in root:
                        neg.write(ele.attrib['id'] + ', ' + clean_str(ele[3].text) + '\n')
                        all_reviews.write(clean_str(ele[3].text) + '\n')
                else:
                    for ele in root:
                        pos.write(ele.attrib['id'] + ', ' + clean_str(ele[3].text) + '\n')
                        all_reviews.write(clean_str(ele[3].text) + '\n')
            neg.close()
            pos.close()

    all_reviews.close()