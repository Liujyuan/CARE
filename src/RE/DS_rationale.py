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
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import numpy as np

def extract_last_sentence(text):
    sentences = re.findall(r'[^。！？]+[。！？]', text)
    if len(sentences) > 1:
        return ''.join(sentences[-1:])
    elif len(sentences) == 1:
        return sentences[0]
    else:
        return text

def map_string_to_number(input_string):
    # 
    mapping = {
        0: ['disagree', 'against'],
        1: ['neutral'],
        2: ['agree', 'favor']
    }

    input_string_lower = input_string.lower()

    best_match_number = None
    best_match_index = -1

    for number, keywords in mapping.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = list(re.finditer(pattern, input_string_lower))
            if matches:
                last_match = matches[-1]
                index = last_match.start()
                if index > best_match_index:
                    best_match_index = index
                    best_match_number = number


    return best_match_number if best_match_number is not None else 1


def get_ans(author_parent, author_child, submission, subreddit, pb, cb, label):
    """
    """
    mapping = {
        0: 'disagree',
        1: 'neutral',
        2: 'agree'
    }
    label_map = mapping[label]



    input = 'author_parent:' + author_parent + ',author_child:' + author_child + ",submission_text:" + submission + ",subreddit:" + subreddit + ",parent_comment:" + pb + ",child_reply:" + cb
    instruction_direct = 'You are a helpful assistant at (Dis)agreement Detection, Giving you author_parent, author_child\'s respective parent_comment and child_reply on submission_text under subreddit,\
            categorize the child_reply\'s stance toward parent_comment from agree, disagree and Neutral.'
    instruction_cash = 'You are a helpful assistant at (Dis)agreement Detection, given parent_comment and child_reply from the post in the subreddit.\
                  First,List Causal factors (i.e., keywords, phrases, contextual information, semantic meaning, semantic relationships, tones, references, Internet expression, topic-specific jokes) of comment-reply that support the (Dis)agreement Detection.\
        Next, deduce the diagnostic REASONING process from premises (i.e. Causal factors, input) that support the (Dis)agreement Detection.\
        Finally, based on the Causal factors, the reasoning and the input, categorize the child_reply\'s stance toward parent_comment from agree, disagree and neutral. Please think step by step'
    instruction_cot = 'You are a helpful assistant at (Dis)agreement Detection, given parent_comment and child_reply from the post in the subreddit.\
        categorize the child_reply\'s attitude toward  parent_comment from agree, disagree and neutral. Please think step by step, and give the rationale.'
    last = 'The last sentence must contain only one true stance value:'  # Please think step by step, and give the rationale.#+ label_map


    DA_ICL='''
   1. Neutral
Definition: A neutral commenter typically does not express clear support or opposition to a particular opinion, topic, or proposal. They may provide factual information, ask questions, or show understanding without taking sides.

Characteristics:

Often offers different perspectives without leaning too heavily toward any one side.

The goal is to remain objective and rational, focusing on analysis or discussion rather than strong personal opinions.

comment: "This topic is indeed complex, and both sides have valid points."

reply: In sensitive or highly controversial topics, commenters may choose to remain neutral to avoid conflict or because they do not wish to engage emotionally in the discussion.

2. Agree
Definition: An agreeing commenter clearly expresses their support for a particular opinion, topic, or proposal, aligning with the position presented in the discussion.

Characteristics:

These commenters often reinforce or expand upon the original viewpoint, providing additional support.

Their comments may include emotional enthusiasm or praise, showing their personal agreement with the point made.

comment: "I completely agree with this! This approach is really effective, especially in the current context."

reply: When a viewpoint aligns with a commenter’s values or interests, they are likely to take an agreeing position and express their support.

3. Disagree
Definition: A disagreeing commenter clearly expresses opposition to a particular opinion, topic, or proposal. They may provide counterarguments, challenge the existing viewpoint, or provoke further discussion.

Characteristics:

These commenters typically offer contradictory evidence or logic to support their opposing stance.

Their comments may be critical but can also be constructive, aiming to encourage deeper discussion.

comment: "I don’t quite agree with this viewpoint. Actually, I think it could lead to more problems."

reply: When a viewpoint conflicts with a commenter’s personal beliefs, experiences, or values, they tend to take a disagreeing stance and engage in rebuttal.
'''
    demo_agree = ''' The following criteria are used to determine agree:
        1.Strong Agree: Clearly and firmly agreeing with the comment or viewpoint.
    2.Fact-based Agree: Supporting the original comment by providing objective facts or data that reinforce the viewpoint.
    3.Cultural Agree : Agreeing with the comment by referencing cultural, societal, or regional trends that support the viewpoint.
    4.Emotional Agreem : Expressing agreement through shared feelings or emotional support, reinforcing the original sentiment.
        '''
    demo_disagree = ''' The following criteria are used to determine disagree:
        1.Ideological Disagree: Disagreeing based on differing fundamental beliefs or political ideologies, often highlighting contrasting principles or values.
    2.Pragmatic Disagree: Disagreeing based on practical considerations or real-world constraints, often highlighting feasibility or effectiveness.
    3.Factual Disagree : Disagreeing based on differing interpretations or presentations of facts, often involving corrections or counter-evidence.
    4.Emotional Disagree : Disagreeing with a strong emotional tone, often involving personal attacks or heightened language.'''
    demo_neutral = '''
       The following criteria are used to determine Neutral:
         1.Inquiry-based Neutral: Maintaining neutrality by asking questions or seeking more information.
    2.Observational Neutral : Offering neutral observations or descriptions without expressing agreement or disagreement.
    3.Informational Neutral : Providing factual or explanatory information without taking a stance.
    4.Speculative Neutral: Maintaining neutrality by offering speculative or hypothetical responses without taking a firm stance.
    5.Emotional Neutral: Expressing neutral sentiments or emotions without taking a stance on the issue.
        '''
    common_dis_agree='''The following criteria are used to determine agree or disagree:
    agree:If the reply expresses agreement with the original comment, it indicates support or confirmation of the point made in the comment. A reply reinforces or extends the argument of the original comment, usually using an affirmative tone.
    disagree:If the reply expresses a disagreement or rebuttal to the original comment, indicating an opposing position or viewpoint. The response will usually directly present an opinion or rebuttal that is opposed to the original comment.
'''

    CA_inter = common_dis_agree+demo_neutral

    CA_con='''The following criteria are used to determine neutral:
    Non-emotional or non-biased: The comment and reply do not express strong emotions such as anger, joy, fear, etc., nor do they show clear support or opposition. They remain objective and neutral without personal emotional bias.

Neutral language in content and response: Replies typically avoid extreme positions or emotional expressions, focusing more on facts, reasoning, or clarification, attempting to understand the issue or ask questions.

No attacking or blaming: Replies avoid attacking, blaming, or belittling the commenter or their viewpoint. Instead, they offer constructive questions, comments, or speculations.

Informative: The comment and reply usually share information or perspectives in a neutral and non-partisan way. For example, some replies provide facts or clear explanations without showing bias or personal opinions.
'''
    messages = [{'role': 'system', 'content': instruction_cash+last},
                {'role': 'user', 'content': input
                 }]
    try:
        response = client.chat.completions.create(
            model=model_res,
            messages=messages,
            stream=False,
            max_tokens=2048,
            temperature=0)
        if hasattr(response, 'choices') and response.choices:
            text = response.choices[0].message.content
            print('submission_text:', submission)
            print('parent:', pb)
            print('child:', cb)
            # print('user:', user)
            print('reason:', text)
            predict_label = extract_last_sentence(text)
            predict_label = map_string_to_number(predict_label)
            print('predict_label is ', predict_label)
            print('True label is ', label)
            return predict_label, text
        else:
            raise ValueError("No choices in the response")
    except Exception as e:
        error_msg = f"Error occurred: {str(e)}"
        print(error_msg)
        return (0, error_msg)

def log_progress_with_time(current, total, log_file='progress_with_time.log'):
    progress = current / total * 100
    if (progress % 1 < 1e-3) or current == total:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f"Time: {current_time}, Progress: {int(progress)}%\n")


def apply_with_progress_and_logging(df, func, log_file='progress_with_time.log'):
    total = len(df)
    with open(log_file, 'w') as f:
        f.write('')

    results = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        result = func(row)
        log_progress_with_time(idx + 1, total, log_file)
        results.append(result)
    return pd.Series(results, index=df.index)



def call_api_with_row(row):
    return get_ans(
        row['author_parent'],
        row['author_child'],
        row['submission_text'],
        row['subreddit'],
        row['body_parent'],
        row['body_child'],
        row['label'],
    )
import time
from random import uniform

def call_with_exponential_backoff(func, row, max_retries=5, base_delay=1):
    """Calls `func` with `row`, and applies exponential backoff on failures."""
    attempt = 0
    while attempt < max_retries:
        try:
            return func(row)
        except Exception as e:
            attempt += 1
            if attempt == max_retries:
                raise e
            delay = min(base_delay * 2 ** attempt + uniform(0, 0.2), 60)  #
            print(f"Retrying in {delay:.2f} seconds... (attempt {attempt} of {max_retries})")
            time.sleep(delay)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



def apply_with_concurrency(df, max_workers=10):
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(call_api_with_row, row): idx for idx, row in df.iterrows()}

        for future in tqdm(as_completed(future_to_index), total=len(future_to_index)):
            try:
                result = future.result()
                index = future_to_index[future]
                results[index] = result if result is not None else ('Error', 'Failed to get result')
            except Exception as e:
                print(f"An error occurred: {e}")
                index = future_to_index[future]
                results[index] = (0, str(e))

    return pd.Series(results)

def apply_with_progress_and_logging(df, func, log_file='progress_with_time.log', max_workers=50):
    total = len(df)
    with open(log_file, 'w') as f:
        f.write('')

    results = apply_with_concurrency(df, max_workers=max_workers)
    for idx in range(1, len(df) + 1):
        log_progress_with_time(idx, total, log_file)

    return results

base_url = ""
client = OpenAI(api_key="EMPTY", base_url=base_url)
model_res=''



mode='test'#test train

if mode=='train':
    df = pd.read_csv('train_data.csv')
elif mode=='test':
    df = pd.read_csv('valid_data.csv')

tqdm.pandas()

results = apply_with_progress_and_logging(df, lambda row: call_api_with_row(row), max_workers=1)
ans, reason = zip(*results)
series = pd.Series(ans)
count = series.value_counts()
print(count)
print(classification_report(df['label'], ans, digits=4))

df_selected = df.assign(ans=ans, reason=reason)

df_true = df_selected[df_selected['ans'] == df_selected['label']].copy()
df_false = df_selected[df_selected['ans'] != df_selected['label']].copy()
df_true.to_csv(mode+'_direct_true_ra.csv', index=False)
df_false.to_csv(mode+'_direct_false_ra.csv', index=False)

print(df_true)
classification_reports = {}

grouped = df.groupby('subreddit')
ans = np.array(ans)
for subreddit, group in grouped:
    ans_sub = ans[group.index]

    series = pd.Series(ans_sub)
    count = series.value_counts()
    print(f"Subreddit: {subreddit}")
    print(count)
    report = classification_report(group['label'], ans_sub, digits=4)
    print(report)
