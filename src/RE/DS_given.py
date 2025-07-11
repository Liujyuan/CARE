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

# demo=concatenated_data
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
    mapping = {
        0: 'disagree',
        1: 'neutral',
        2: 'agree'
    }
    label_map = mapping[label]
    input = 'author_parent:' + author_parent + ',author_child:' + author_child + ",submission_text:" + submission + ",subreddit:" + subreddit + ",parent_current_comment:" + pb + ",child_current_reply:" + cb
    # user = ',parent_history_interactions:' + hp + ',child_history_interactions:' + hc
    input=input
    instruction = 'You are a helpful assistant at (Dis)agreement Detection, Giving you author_parent, author_child\'s respective parent_current_comment and child_current_reply on submission_text under subreddit,\
              categorize the child_reply\'s stance toward parent_comment from agree, disagree and Neutral. \
               Please think step by step, and give the rationale.'
    last = 'The last sentence must contain only one true stance value:' +label_map
    messages = [{'role': 'system', 'content': instruction+last},
                {'role': 'user', 'content': input
                 }]

    try:
        response = client.chat.completions.create(
            model=model_res,
            messages=messages,
            stream=False,
            max_tokens=2048,
            temperature=0.9)

        if hasattr(response, 'choices') and response.choices:
            text = response.choices[0].message.content
            print('submission_text:', submission)
            print('parent:', pb)
            print('child:', cb)
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



from concurrent.futures import ThreadPoolExecutor, as_completed


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


def apply_with_progress_and_logging(df, func, log_file='progress_with_time.log', max_workers=200):
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

mode='train'#test train
if mode=='train':
    df = pd.read_csv('train_data.csv')
elif mode=='test':
    df = pd.read_csv('valid_data.csv')

df = df[:1000]
tqdm.pandas()

results = apply_with_progress_and_logging(df, lambda row: call_api_with_row(row), max_workers=100)

ans, reason = zip(*results)

print(classification_report(df['label'], ans, digits=4))

df_selected = df.assign(ans=ans, reason=reason)
df_true = df_selected[df_selected['ans'] == df_selected['label']].copy()
df_false = df_selected[df_selected['ans'] != df_selected['label']].copy()
df_true.to_csv(mode+'_direct_true_given.csv', index=False)
df_false.to_csv(mode+'_direct_false_given.csv', index=False)

