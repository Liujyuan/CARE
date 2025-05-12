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
        if index > best_match_index:  #
            best_match_index = index
            best_match_number = number

    return best_match_number if best_match_number is not None else 1
def get_ans(author_parent,author_child,submission, subreddit,pb,cb,label):
    mapping = {
        0: 'disagree',
        1: 'neutral',
        2: 'agree'
    }

    label_map = mapping[label]
    input = 'author_parent:' + author_parent + ',author_child:' + author_child  + ",subreddit:" + subreddit + ",post:" + submission+ ",parent_comment:" + pb + ",child_reply:" + cb
    instruction_old = 'You are a helpful assistant at (Dis)agreement Detection. Given the submission_text, the author_parent\'s parent_comment, and the author_child\'s child_reply under post in subreddit, categorize the child_reply\'s stance toward parent_comment from agree, disagree and neutral.'
    last = 'The last sentence must contain only one true stance value:'
    output = label_map
    return {
        "instruction": instruction_old+ last,
        "input": input,
        "output": output
    }

    label_map = mapping[label]
    input = 'AUTHOR_PARENT:' + author_parent + ',AUTHOR_CHILD:' + author_child + ",SUBREDDIT:" + subreddit + ",POST:" + submission + ",PARENT_COMMENT:" + pb + ",CHILD_REPLY:" + cb
    instruction_old = 'You are a helpful assistant at (Dis)agreement Detection. Given the submission_text, the AUTHOR_PARENT\'s PARENT_COMMENT, and the AUTHOR_CHILD\'s CHILD_REPLY under POST in SUBREDDIT, categorize the child_reply\'s stance toward parent_comment from agree, disagree and neutral.'
    last = 'The last sentence must contain only one true stance value:'
    output = label_map
    return {
        "instruction": instruction_old + last,
        "input": input,
        "output": output
    }



df = pd.read_csv('_ans.csv')
df = pd.read_csv('_rewrite.csv')
tqdm.pandas()

ans = df.progress_apply(lambda row: get_ans(row['author_parent'],row['author_child'],row['submission_text'], row['subreddit'],row['body_parent'],row['body_child'],row['label']),
                           axis=1).tolist()

import json
with open('_ans.json', 'w') as f:
    json.dump(ans, f, indent=4)


# with open('_rw.json', 'w') as f:
#     json.dump(topics, f, indent=4)