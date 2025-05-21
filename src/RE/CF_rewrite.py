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
import random
random.seed(0)
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

    mapping_agree = {
        0: ' Agree',
        1: ' Agree',
        2: ' Agree',
        3: ' Agree'
    }
    mapping_disagree = {
        0: ' Disagree',
        1: ' Disagree',
        2: ' Disagree',
        3: ' Disagree'
    }
    mapping_neutral = {
        0: 'Inquiry-based Neutral',
        1: 'Observational Neutral',
        2: 'Informational Neutral',
        3: 'Speculative Neutral',
        4: 'Emotional Neutral'
    }


    seed_n = random.randint(0, len(mapping_neutral)-1)
    seed_a= random.randint(0, len(mapping_agree)-1)
    seed_d = random.randint(0, len(mapping_disagree)-1)

    mapping = {
        0: 'disagree',
        1: 'neutral',
        2: 'agree'
    }

    label_map=mapping[label]
    input_data = f'author_parent:{author_parent}, author_child:{author_child}, submission_text:{submission}, subreddit:{subreddit}, parent_comment:{pb}, child__reply:{cb}'
    role = 'You are a helpful assistant at (Dis)agreement Detection are very familiar with the way of expression on the Internet.'
    last = 'Your task is to rewrite the counterfactual child_reply to meet the new attitude, while retaining internal coherence and avoids unnecessary changes. You only need to include the rewritten reply in your response.'

    example = '''
           Example:
           comment: Bad study. Chemistry, Physics and Biology textbooks shouldn't be devoting that much space to climate change. 4% or about 600 pages of 15,000 pages between 16 books seems reasonable. That's about 35 or 40 pages a book or almost a whole fucking chapter in what are books that only touch around the edges of the subject.
       reply: Yeah . I don't know what the hell they think that is supposed to indicate?? Why would my physiology class talk about climate change?
       Your response: 
           Climate change has been incorporated into a lot of different subjects recently because of its broad impact on various fields, including biology, chemistry, and physics. The inclusion might feel out of place in a physiology class, but it could be there to highlight how environmental factors influence human health or other biological systems. Some textbooks are adding sections to provide students with a broader understanding of how these disciplines relate to current global issues.
           '''

    demo_neutral = '''
        Here are several subcategories of neutral that can help with disagreement detection:
         1.Inquiry-based Neutral: Maintaining neutrality by asking questions or seeking more information.
    2.Observational Neutral : Offering neutral observations or descriptions without expressing agreement or disagreement.
    3.Informational Neutral : Providing factual or explanatory information without taking a stance.
    4.Speculative Neutral: Maintaining neutrality by offering speculative or hypothetical responses without taking a firm stance.
    5.Emotional Neutral: Expressing neutral sentiments or emotions without taking a stance on the issue.
        '''
    common_dis_agree = '''The following criteria are used to determine agree or disagree:
        agree:If the reply expresses agreement with the original comment, it indicates support or confirmation of the point made in the comment. A reply reinforces or extends the argument of the original comment, usually using an affirmative tone.
        disagree:If the reply expresses a disagreement or rebuttal to the original comment, indicating an opposing position or viewpoint. The response will usually directly present an opinion or rebuttal that is opposed to the original comment.'''

    demo = common_dis_agree+ demo_neutral
    def generate_instruction(label):
        if label == 0:
            instruction = f'Giving you author_parent, author_child\'s respective comment and reply on submission_text under subreddit,\
                                       The child_reply expresses a {label_map} to the parent_comment. Please make changes to the child_reply to express a {mapping_disagree[seed_d]} attitude to the parent_comment.  '
        elif label == 1:
            instruction = f'Giving you author_parent, author_child\'s respective comment and reply on submission_text under subreddit,\
                                              The child_reply expresses a {label_map} to the parent_comment. Please make changes to the child_reply to express a {mapping_neutral[seed_n]} attitude to the parent_comment.  '
        elif label == 2:
            instruction = f'Giving you author_parent, author_child\'s respective comment and reply on submission_text under subreddit,\
                                            The child_reply expresses a {label_map} to the parent_comment. Please make changes to the child_reply to express a {mapping_agree[seed_a]} attitude to the parent_comment.  '
        return instruction

    new_reply = []
    for attitude in [0, 1, 2]:
        instruction = generate_instruction(attitude)
        messages=[{
            'role': 'system', 'content': role + instruction + last+demo+example},
            {'role': 'user', 'content': input_data
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
                print('parent:', pb)
                print('child:', cb)

                print('New_reply:', text)
                print('Old_label is ', label_map)
                print('New label is ', attitude)
                new_reply.append(text)
            else:
                raise ValueError("No choices in the response")

        except Exception as e:
                error_msg = f"Error occurred: {str(e)}"
                print(error_msg)
                return [{'ans': 'error', 'text': error_msg}]  # Return error information as a dictionary

    return new_reply,[0,1,2]

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


def apply_with_progress_and_logging(df, func, log_file='progress_with_time.log', max_workers=20):
    total = len(df)
    with open(log_file, 'w') as f:
        f.write('')

    results = apply_with_concurrency(df, max_workers=max_workers)
    for idx in range(1, len(df) + 1):
        log_progress_with_time(idx, total, log_file)

    return results

base_url = ""
client = OpenAI(api_key="", base_url=base_url)
model_res=''


df = pd.read_csv('train.csv')


tqdm.pandas()

results = apply_with_progress_and_logging(df, lambda row: call_api_with_row(row), max_workers=200)

new_replies,new_labels  = zip(*results)

df_disagree = df.copy()
df_disagree['label'] = [labels[0] for labels in new_labels]  #
df_disagree['body_child'] = [replies[0] for replies in new_replies]  #

df_neutral = df.copy()
df_neutral['label'] = [labels[1] for labels in new_labels]  #
df_neutral['body_child'] = [replies[1] for replies in new_replies]  #

df_agree = df.copy()
df_agree['label'] = [labels[2] for labels in new_labels]  #
df_agree['body_child'] = [replies[2] for replies in new_replies]  #

df_combined = pd.concat([df_disagree, df_neutral, df_agree], ignore_index=True)


print("Combined DataFrame saved:")
print(df_combined)

df_combined.to_csv('Counterfactual_rewrite.csv', index=False)
print(df)
