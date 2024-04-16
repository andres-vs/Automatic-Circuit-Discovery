from functools import partial
from acdc.docstring.utils import AllDataThings
from acdc.acdc_utils import kl_divergence
import torch
import torch.nn.functional as F
from transformer_lens.HookedTransformer import HookedTransformer

from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
import random

def remove_special_tokens(example):
    example['input'] = example['input'].replace('[CLS]', '')
    return {'text': example['input'], 'label': example['label']}

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

def get_bert_base_uncased(device):
    tl_model = HookedTransformer.from_pretrained('bert-base-uncased')
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def get_all_text_entailment_things(num_examples, device, metric_name, kl_return_one_element=True):
    tl_model=get_bert_base_uncased(device)

    login(token="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", add_to_git_credential=True)
    dataset_name = "andres-vs/ruletaker-Att-Noneg-depth0"
    # as you do sentence classification in this task, I used a classification model instead of an autoregressive one
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset(dataset_name)
    test_size = len(dataset["test"])
    if num_examples*2 < test_size:
        examples = dataset["test"].select(random.sample(range(test_size), num_examples*2))
    else:
        raise ValueError("num_examples cannot exceed half of the test split size.")
    
    examples = examples.map(remove_special_tokens)
    tokenized_examples = examples.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    validation_data = tokenized_examples[:num_examples]["input_ids"].to(device)
    validation_labels = tokenized_examples[:num_examples]["label"].to(device)
    test_data = tokenized_examples[num_examples:]["input_ids"].to(device)
    test_labels = tokenized_examples[num_examples:]["label"].to(device)

    # nog niet zeker of dit werkt en hoe het werkt
    with torch.no_grad():
        base_model_logits = tl_model(tokenized_examples)[:, -1, :]
        base_model_logprobs = F.log_softmax(base_model_logits, dim=-1)
    base_validation_logprobs = base_model_logprobs[:num_examples, :]
    base_test_logprobs = base_model_logprobs[num_examples:, :]

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
        validation_mask=None,
        # validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        # test_patch_data=test_patch_data,
    )