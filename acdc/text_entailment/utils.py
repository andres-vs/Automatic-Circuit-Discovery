from functools import partial
from acdc.docstring.utils import AllDataThings
from acdc.acdc_utils import kl_divergence
import torch
import torch.nn.functional as F
from transformer_lens.HookedEncoder import HookedEncoder
from transformers import AutoTokenizer

from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
import random
from datasets import Dataset
from tqdm import tqdm

# def remove_special_tokens(example):
#     example['input'] = example['input'].replace('[CLS]', '')
#     return {'input': example['input'], 'label': example['label']}

# def tokenize_function(tokenizer, examples, padding, max_length=None):
#     if max_length is not None:
#         return tokenizer(examples["input"], truncation=True, padding=padding, max_length=max_length)
#     else:
#         return tokenizer(examples["input"], truncation=True, padding=padding)

def remove_special_tokens(example):
    return {'input': example['input'].replace('[CLS]', ''), 'label': int(example['label'])}

def tokenize_function(tokenizer, examples, padding, max_length=None):
    tokenized_output = tokenizer(
        examples["input"],
        truncation=True,  # Truncate examples to max_length
        padding=padding,  # Enable consistent padding for batching
        max_length=max_length,
        return_attention_mask=True,  # Generate an attention mask
        return_tensors="pt",  # Return tensors
    )
    tokenized_output['label'] = examples['label']
    return tokenized_output

def get_finetuned_bert_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tl_model = HookedEncoder.from_pretrained(model_name, tokenizer=tokenizer, head_type='classification') #, fold_ln=False)
    tl_model = tl_model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    print(tl_model.cfg.to_dict())
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def invert_query(query):
    if 'not' in query:
        return query.replace('not ', '')
    else:
        return query.replace(' ', ' not ', -1)
def generate_corrupt_examples(examples):
    inputs = []
    labels = []
    for example in examples:
        input_text = example['input']
        theory, query = input_text.split('[SEP]')
        if example['label']:
            if "not" in query:
                theory = theory + ' ' + invert_query(query)
            else:
                theory = theory.replace(query, '')
        else:
            if "not" in query:
                theory = theory.replace(invert_query(query), '')
            else:
                theory = theory + ' ' + query
        inputs.append(theory + '[SEP]' + query)
        labels.append(not example['label'])
    return Dataset.from_dict({'input': inputs, 'label': labels})


def get_all_text_entailment_things(model_name, test_dataset, num_examples, device, metric_name, kl_return_one_element=True, max_length=None):
    tl_model = get_finetuned_bert_model(model_name, device)

    if len(test_dataset) < 2 * num_examples:
        raise ValueError("The test dataset must contain at least 2 * num_examples examples.")

    validation_examples = test_dataset.select(range(num_examples))
    test_examples = test_dataset.select(range(num_examples, 2 * num_examples))

    validation_examples = validation_examples.map(remove_special_tokens)
    test_examples = test_examples.map(remove_special_tokens)

    corrupted_validation_examples = generate_corrupt_examples(validation_examples)
    corrupted_test_examples = generate_corrupt_examples(test_examples)

    tokenized_validation = tokenize_function(tl_model.tokenizer, validation_examples, padding='max_length' if max_length else True)
    tokenized_corrupted_validation = tokenize_function(tl_model.tokenizer, corrupted_validation_examples, padding='max_length' if max_length else True)
    tokenized_test = tokenize_function(tl_model.tokenizer, test_examples, padding='max_length' if max_length else True)
    tokenized_corrupted_test = tokenize_function(tl_model.tokenizer, corrupted_test_examples, padding='max_length' if max_length else True)

    validation_data = tokenized_validation["input_ids"]
    validation_mask = tokenized_validation["attention_mask"]
    validation_patch_data = tokenized_corrupted_validation["input_ids"]
    validation_labels = validation_examples["label"]

    test_data = tokenized_test["input_ids"]
    test_mask = tokenized_test["attention_mask"]
    test_patch_data = tokenized_corrupted_test["input_ids"]
    test_labels = test_examples["label"]

    batch_size = 8
    base_model_logits = []
    for i in tqdm(range(0, len(tokenized_validation["input_ids"]), batch_size)):
        batch_inputs = {
            "input_ids": tokenized_validation["input_ids"][i:i+batch_size],
            "attention_mask": tokenized_validation["attention_mask"][i:i+batch_size]
        }

        with torch.no_grad():
            logits = tl_model(input=batch_inputs['input_ids'], one_zero_attention_mask=batch_inputs['attention_mask'])

        base_model_logits.append(logits)
        del batch_inputs["input_ids"], batch_inputs["attention_mask"], batch_inputs
        del logits
        torch.cuda.empty_cache()

    for i in tqdm(range(0, len(tokenized_test["input_ids"]), batch_size)):
        batch_inputs = {
            "input_ids": tokenized_test["input_ids"][i:i+batch_size],
            "attention_mask": tokenized_test["attention_mask"][i:i+batch_size]
        }

        with torch.no_grad():
            logits = tl_model(input=batch_inputs['input_ids'], one_zero_attention_mask=batch_inputs['attention_mask'])

        base_model_logits.append(logits)
        del batch_inputs["input_ids"], batch_inputs["attention_mask"], batch_inputs
        del logits
        torch.cuda.empty_cache()

    base_model_logits = torch.cat(base_model_logits, dim=0)
    base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)

    base_validation_logprobs = base_model_logprobs[:num_examples, :]
    base_test_logprobs = base_model_logprobs[num_examples:, :]

    del base_model_logits
    del base_model_logprobs
    torch.cuda.empty_cache()

    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=False,
            base_model_probs_last_seq_element_only=False,
            return_one_element=kl_return_one_element,
        )
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=None,
            last_seq_element_only=False,
        ),
    }

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=validation_mask,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=test_mask,
        test_patch_data=test_patch_data,
    )
