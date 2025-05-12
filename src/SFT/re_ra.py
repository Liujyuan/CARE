import pandas as pd
import requests
from http import HTTPStatus
from openai import OpenAI
import json
import tqdm
from tqdm import tqdm
import pandas
from sklearn.metrics import classification_report
import re

def extract_last_sentence(text):
    sentences = re.findall(r'[^。！？]+[。！？]', text)

    if len(sentences) > 1:

        return ''.join(sentences[-1:])
    elif len(sentences) == 1:

        return sentences[0]
    else:

        return text
def map_string_to_number(input_string):

    mapping = {
        0: 'against',
        1: 'neutral',
        2: 'favor'
    }


    input_string_lower = input_string.lower()

    best_match_number = None
    best_match_index = -1

    for number, keyword in mapping.items():
        keyword_lower = keyword.lower()
        index = input_string_lower.rfind(keyword_lower)
        if index > best_match_index:
            best_match_index = index
            best_match_number = number

    return best_match_number if best_match_number is not None else 1

def get_ans(author_parent, author_child, submission, subreddit, pb, cb, label,ans,reason,re_ans,re_reason):

    mapping = {
        0: 'disagree',
        1: 'neutral',
        2: 'agree'
    }
    label_map = mapping[label]
    ans_map = mapping[ans]

    input = 'AUTHOR_PARENT:' + author_parent + ',AUTHOR_CHILD:' + author_child + ",SUBREDDIT:" + subreddit + ",POST:" + submission + ",PARENT_COMMENT:" + pb + ",CHILD_REPLY:" + cb

    ans_reason = ',previous misjudgment:' + ans_map + ',rationale of misjudgment:' + reason
    input = input + ans_reason
    instruction = ('You are a helpful assistant at (Dis)agreement Detection,\
                    (Dis)agreement Detection is categorizing the child_reply\'s stance toward parent_comment from agree, disagree and neutral.\
                   Providing you with the author_parent\'s parent_comment, and the author_child\'s child_reply under submission_text in subreddit, along with the misjudgment and the rationale of misjudgment.\
                   First, self-critique the previous misjudgment and rationale of misjudgment.\
                  Second, self-correct to the correct judgment.\
                      The last sentence must contain only one true stance value:')
    output= re_reason
    last = 'The last sentence must contain only one true stance value:'
    return {
        "instruction": instruction+last,
        "input": input,
        "output": output
    }

df = pd.read_csv('_re.csv')
print(df)
tqdm.pandas()

re = df.progress_apply(lambda row: get_ans(row['author_parent'],row['author_child'],row['submission_text'], row['subreddit'],row['body_parent'],row['body_child'],row['label'], \
                                           row['ans'],row['reason'], row['re_ans'],row['re_reason']),
                           axis=1).tolist()

import json
with open('_re.json', 'w') as f:
    json.dump(re, f, indent=4)