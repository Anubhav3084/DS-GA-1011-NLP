import json
import collections
import argparse
import random
import numpy as np
import requests
import re

def your_netid():
    YOUR_NET_ID = 'PUT_YOUR_NEW_ID'
    return YOUR_NET_ID

def your_hf_token():
    YOUR_HF_TOKEN = 'PUT_YOUR_TOKEN'
    return YOUR_HF_TOKEN


# for adding small numbers (1-6 digits) and large numbers (7 digits), write prompt prefix and prompt suffix separately.
def your_prompt():
    """Returns a prompt to add to "[PREFIX]a+b[SUFFIX]", where a,b are integers
    Returns:
        A string.
    Example: a=1111, b=2222, prefix='Input: ', suffix='\nOutput: '
    """
    prefix = """1234567+1234567=2469134
4050607+2030405=6081012
6543210+2109876=8653086
3141592+2718281=5859873
5000000+4999999=9999999
5678912+4567891=10246803
8765432+7654321=16419753
9876543+1234567=11111110
2468013+7531987=10000000
8123456+9876544=18000000
9191919+1818181=11010100
"""
    suffix = "="

    return prefix, suffix


def your_config():
    """Returns a config for prompting api
    Returns:
        For both short/medium, long: a dictionary with fixed string keys.
    Note:
        do not add additional keys.
        The autograder will check whether additional keys are present.
        Adding additional keys will result in error.
    """
    config = {
        'max_tokens': 50,
        'temperature': 0.03,   # near-greedy; exactly 0 can raise an error in HF generate
        'top_k': 1,            # always pick the most likely token
        'top_p': 0.99,
        'repetition_penalty': 1,
        'stop': []}

    return config


def your_pre_processing(s):
    return s

    
def your_post_processing(output_string):
    """Return the first run of digits in the model's output; ignore the rest.

    The prompt ends at "=", so the answer is the first number the model writes.
    Anything it generates afterwards (a new question, text) is discarded.
    Returns 0 if no digits are found.
    """
    match = re.search(r"\d{1,}", output_string)
    if match is None:
        return 0
    return int(match.group())
