import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from time import sleep
import random
import re

client = OpenAI(
    api_key= 'your_api_key',
)

def chat_with_gpt(question, history, max_tries=5, model = 'gpt-3.5-turbo'):
    messages = history.copy()
    messages.append({"role": "user", "content": question})
    response = None
    for i in range(max_tries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                n=1,
                stop=None,
                timeout=None
            )
            break
        except Exception as e:
            print(f"Attempt {i+1} failed with error: {e}")
            if i < max_tries:
                sleep(5)
            else:
                raise e
    if response is None:
        reply = None
    else:
        reply = response.choices[0].message.content
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": reply})
    return reply

def extract_option_value(text):
    pattern = re.compile(r"answer is Option ([A-Z])", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).lower()
    else:
        return None

def get_counterfactual_question(question, optionA, optionB, optionC, optionD, counterfactual_label):
    question = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}

Try to explain why the question's answer might be option {counterfactual_label}.

Output Format:
Judgement: The answer is option {counterfactual_label}.
Reasoning: [Your reasoning here]
"""
    return question

def get_critic_question(question, optionA, optionB, optionC, optionD, agent_reply):
    question = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}
Assistant: {agent_reply}

The Assistant's answer maybe wrong. Please persuade the assistant that his answer maybe wrong."""
    return question

def get_revision_question(question, optionA, optionB, optionC, optionD, agent_reply, critic_reply):
    question = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}
Assistant: {agent_reply}

Critic: {critic_reply}

As assistant, please refute the critic's answer and persuade the critic that your answer is correct"""
    return question

df = pd.read_csv('college_mathematics_test.csv', header=None, names=['question','Option_A','Option_B','Option_C','Option_D','answer'])

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    question = row['question']
    optionA = row['Option_A']
    optionB = row['Option_B']
    optionC = row['Option_C']
    optionD = row['Option_D']
    top_1 = row['answer']
    top_2 = random.choice([i for i in ['A', 'B', 'C', 'D'] if i != top_1])

    counterfactual_1_question = get_counterfactual_question(question, optionA, optionB, optionC, optionD, top_1)
    counterfactual_1 = chat_with_gpt(counterfactual_1_question, [])
    df.loc[idx, 'counterfactual_1'] = counterfactual_1

    reflection_1_question = get_critic_question(question, optionA, optionB, optionC, optionD, counterfactual_1)
    reflection_1 = chat_with_gpt(reflection_1_question, [])
    df.loc[idx, 'reflection_1'] = reflection_1

    revision_1_question = get_revision_question(question, optionA, optionB, optionC, optionD, counterfactual_1, reflection_1)
    revision_1 = chat_with_gpt(revision_1_question, [])
    df.loc[idx, 'revision_1'] = revision_1

    counterfactual_2_question = get_counterfactual_question(question, optionA, optionB, optionC, optionD, top_2)
    counterfactual_2 = chat_with_gpt(counterfactual_2_question, [])
    df.loc[idx, 'counterfactual_2'] = counterfactual_2

    reflection_2_question = get_critic_question(question, optionA, optionB, optionC, optionD, counterfactual_2)
    reflection_2 = chat_with_gpt(reflection_2_question, [])
    df.loc[idx, 'reflection_2'] = reflection_2

    revision_2_question = get_revision_question(question, optionA, optionB, optionC, optionD, counterfactual_2, reflection_2)
    revision_2 = chat_with_gpt(revision_2_question, [])
    df.loc[idx, 'revision_2'] = revision_2

    candidates_response = [(counterfactual_1,reflection_1,revision_1), (counterfactual_2,reflection_2,revision_2)]

    judge_question = f"""Question: {question}
Options:
A. {optionA}
B. {optionB}
C. {optionC}
D. {optionD}

Which option is the answer of the question? The results of the analysis for each of the possible options are as follows:
{{
    "Option {top_1}": {{
        "initial analysis": "{candidates_response[0][0]}",
        "critic analysis": "{candidates_response[0][1]}",
        "rebuttle analysis": "{candidates_response[0][2]}"
    }},
    "Option {top_2}": {{
        "initial analysis": "{candidates_response[1][0]}",
        "critic analysis": "{candidates_response[1][1]}",
        "rebuttle analysis": "{candidates_response[1][2]}"
    }}
}}
After thinking the analysis above, do you think which option is the most appropriate answer for the question({question})? Please only give a correct answer and no other replies.

Output Format:
The answer is option [X].
"""
    judge = chat_with_gpt(judge_question, [])
    df.loc[idx, 'judge'] = judge

df.to_csv('college_mathematics_test_result.csv', index=False)
