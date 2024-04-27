from functools import partial
from acdc.docstring.utils import AllDataThings
from acdc.acdc_utils import kl_divergence
import torch
import torch.nn.functional as F
from transformer_lens.HookedEncoder import HookedEncoder
from transformers import AutoTokenizer

from huggingface_hub import login
from datasets import load_dataset
import random
from datasets import Dataset
from tqdm import tqdm

def remove_special_tokens(example):
    example['input'] = example['input'].replace('[CLS]', '')
    return {'input': example['input'], 'label': example['label']}

def tokenize_function(tokenizer, examples, padding, max_length=None):
    if max_length is not None:
        return tokenizer(examples["input"], truncation=True, padding=padding, max_length=max_length)
    else:
        return tokenizer(examples["input"], truncation=True, padding=padding)

def get_finetuned_bert_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tl_model = HookedEncoder.from_pretrained(model_name, tokenizer=tokenizer) #, fold_ln=False)
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

def get_all_text_entailment_things(model_name, num_examples, device, metric_name, kl_return_one_element=True):
    tl_model=get_finetuned_bert_model(model_name, device)

    login(token="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", add_to_git_credential=True)
    dataset_name = "andres-vs/ruletaker-Att-Noneg-depth0"
    # as you do sentence classification in this task, I used a classification model instead of an autoregressive one
    # model_name = "bert-base-uncased"

    dataset = load_dataset(dataset_name)
    test_size = len(dataset["test"])
    if num_examples*2 < test_size:
        examples = dataset["test"].select(random.sample(range(test_size), num_examples*2))
    else:
        raise ValueError("num_examples cannot exceed half of the test split size.")
    
    examples = examples.map(remove_special_tokens)
    corrupted_examples = generate_corrupt_examples(examples)
    
    tokenized_examples = tokenize_function(tl_model.tokenizer, examples, padding=True)
    print(tokenized_examples)
    print(len(tokenized_examples["input_ids"][0]), len(tokenized_examples["input_ids"][1]), len(tokenized_examples["attention_mask"][0]), len(tokenized_examples["attention_mask"][1]))
    tokenized_corrupted_examples = tokenize_function(tl_model.tokenizer, corrupted_examples, padding="max_length", max_length=len(tokenized_examples["input_ids"][0]))
    
    # validation_data = examples[:num_examples]["input"]
    # validation_patch_data = corrupted_examples[:num_examples]["input"]
    # validation_labels = examples[:num_examples]["label"]
    # test_data = examples[num_examples:]["input"]
    # test_patch_data = corrupted_examples[num_examples:]["input"]
    # test_labels = examples[num_examples:]["label"]
    validation_data = torch.tensor(tokenized_examples["input_ids"][:num_examples])
    valdiation_mask = torch.tensor(tokenized_examples["attention_mask"][:num_examples])
    validation_patch_data = torch.tensor(tokenized_corrupted_examples["input_ids"][:num_examples])
    validation_labels = examples[:num_examples]["label"]
    test_data = torch.tensor(tokenized_examples["input_ids"][num_examples:])
    test_mask = torch.tensor(tokenized_examples["attention_mask"][num_examples:])
    test_patch_data = torch.tensor(tokenized_corrupted_examples["input_ids"][num_examples:])
    test_labels = examples[num_examples:]["label"]


    batch_size = 8
    base_model_logits = []
    # print(-1, torch.cuda.memory_allocated())
    for i in tqdm(range(0, len(tokenized_examples["input_ids"]), batch_size)):
        batch_inputs = {
            "input_ids": torch.tensor(tokenized_examples["input_ids"][i:i+batch_size]),
            "attention_mask": torch.tensor(tokenized_examples["attention_mask"][i:i+batch_size])
        }
        # print(i, "batch_inputs", torch.cuda.memory_allocated())
        # batch_inputs_size = batch_inputs['input_ids'].element_size() * batch_inputs['input_ids'].nelement() / (1024 * 1024 * 1024)
        
        with torch.no_grad():
            logits = tl_model(input=batch_inputs['input_ids'], one_zero_attention_mask=batch_inputs['attention_mask'])[:, -1, :]
        # print(i, "logits", torch.cuda.memory_allocated())
        base_model_logits.append(logits)
        # print(i, "appended", torch.cuda.memory_allocated())
        del batch_inputs["input_ids"], batch_inputs["attention_mask"], batch_inputs
        del logits
        torch.cuda.empty_cache()
        # print(i, "deleted", torch.cuda.memory_allocated())
        # wait = input("(iteration done) Press Enter to continue.")
    # wait = input("(calculated base model logits) Press Enter to continue.")
    base_model_logits = torch.cat(base_model_logits, dim=0)
    # wait = input("(recalculated base model logprobs) Press Enter to continue.")
    base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)
    # wait = input("(calculated base model logprobs) Press Enter to continue.")
    base_validation_logprobs = base_model_logprobs[:num_examples, :]
    base_test_logprobs = base_model_logprobs[num_examples:, :]
    # wait = input("(derived validation and test logprobs) Press Enter to continue.")
    del base_model_logits
    del base_model_logprobs
    torch.cuda.empty_cache()
    # wait = input("(deleted logits and logprob vars) Press Enter to continue.")

    if metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            last_seq_element_only=True,
            base_model_probs_last_seq_element_only=False,
            return_one_element=kl_return_one_element,
        )

    test_metrics = {
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=None,
            last_seq_element_only=True,
        ),
    }

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=valdiation_mask,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=test_mask,
        test_patch_data=test_patch_data,
    )