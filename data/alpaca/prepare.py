# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # Load Alpaca dataset
    dataset = load_dataset("tatsu-lab/alpaca")

    def alpaca_to_text(example):
        # Combine instruction, input, and output into a single text field
        instruction = example['instruction']
        input_text = example['input']
        output = example['output']
        
        # Format: "Instruction: ... Input: ... Response: ..."
        if input_text.strip():
            text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
        else:
            text = f"Instruction: {instruction}\nResponse: {output}"
        
        return {'text': text}
    
    # Apply the conversion
    dataset = dataset.map(
        alpaca_to_text,
        remove_columns=['instruction', 'input', 'output'],
        desc="Converting Alpaca to text format",
        num_proc=num_proc,
    )

    # owt by default only contains the 'train' split, so create a test split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # we now want to tokenize the dataset. first define the encoding functio |