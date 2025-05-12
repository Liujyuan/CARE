import pandas as pd
import requests
from http import HTTPStatus
from openai import OpenAI
import json
import tqdm
from tqdm import tqdm


def get_ans(author_parent, author_child, submission, subreddit, pb, cb, label,reason):#, path, np, nc):  # ,hc,hp,path
    """
    """
    mapping = {
        0: 'disagree',
        1: 'neutral',
        2: 'agree'
    }
    label_map = mapping[label]

    input = 'AUTHOR_PARENT:' + author_parent + ',AUTHOR_CHILD:' + author_child + ",SUBREDDIT:" + subreddit + ",POST:" + submission + ",PARENT_COMMENT:" + pb + ",CHILD_REPLY:" + cb
    instruction_old = 'You are a helpful assistant at (Dis)agreement Detection. Given the submission_text, the AUTHOR_PARENT\'s PARENT_COMMENT, and the AUTHOR_CHILD\'s CHILD_REPLY under POST in SUBREDDIT, categorize the child_reply\'s stance toward parent_comment from agree, disagree and neutral.'
    last = 'The last sentence must contain only one true stance value:'
    COT_ra='Please think step by step, and give the rationale.'
    CARP_ra='First,List CLUES (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references, Internet expression, topic-specific jokes) of comment-reply that support the (Dis)agreement Detection.\
      Next, deduce the diagnostic REASONING process from premises (i.e. clues, input) that support the (Dis)agreement Detection.\
      Finally, based on the clues, the reasoning and the input, categorize the child_reply\'s stance toward parent_comment from agree, disagree and neutral. Please think step by step.'
    last = 'The last sentence must contain only one true stance value:'  # Please think step by step, and give the rationale.#+ label_map

    output= reason
    return {
        "instruction": instruction_old+COT_ra+last,
        "input": input,
        "output": output
    }


df = pd.read_csv('_ra.csv')

print(df)
tqdm.pandas()

ra = df.progress_apply(lambda row: get_ans(row['author_parent'],row['author_child'],row['submission_text'], row['subreddit'],row['body_parent'],row['body_child'],row['label'], \
                                           row['reason']),
                           axis=1).tolist()

import json

with open('_ra.json', 'w') as f:
    json.dump(ra, f, indent=4)
