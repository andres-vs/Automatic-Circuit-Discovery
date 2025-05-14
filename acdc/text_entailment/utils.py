from functools import partial
from acdc.docstring.utils import AllDataThings
from acdc.acdc_utils import kl_divergence, custom_logit_diff_metric_entailment
import torch
import torch.nn.functional as F
from transformer_lens.HookedEncoder import HookedEncoder
from transformers import AutoTokenizer

from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets, Dataset
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
        rules, facts, query = parse_input_sequence(input_text)
        if example['depth'] == 0:
            relevant_rules, relevant_facts, query = determine_relevant_rules_and_facts(rules, facts, query, depth=0)
            if example['proof_strategy'] == 'proof':
                # make 'random' case by removing query fact from theory
                # Find facts that are not in the relevant facts
                theory_index = relevant_facts[0].theory_index if relevant_facts else None
                non_relevant_facts = [fact for fact in facts if fact not in relevant_facts] + generate_dummy_fact(rules, facts, query, theory_index)
                theory = reconstruct_theory(rules, non_relevant_facts)
            elif example['proof_strategy'] == 'inv-proof':
                # make 'inv-random' case by removing inverse of query fact from theory
                inverted_query = query.invert()
                for fact in facts:
                    if fact == inverted_query:
                        fact_to_remove = fact
                        break
                theory_index = fact_to_remove.theory_index if fact_to_remove else None
                non_relevant_facts = [fact for fact in facts if fact != inverted_query] + generate_dummy_fact(rules, facts, query, theory_index)
                theory = reconstruct_theory(rules, non_relevant_facts)
            elif example['proof_strategy'] == 'random':
                # make 'proof' case by adding query fact to theory
                theory = theory + ' ' + str(query)
            elif example['proof_strategy'] == 'inv-random':
                # make 'inv-proof' case by adding inverse of query fact to theory
                theory = theory + ' ' + str(query.invert())
            inputs.append(theory + '[SEP]' + str(query))
            labels.append(not example['label'])

        elif example['depth'] == 1:
            relevant_rules, relevant_facts, query = determine_relevant_rules_and_facts(rules, facts, query, depth=1)
            if example['proof_strategy'] == 'proof':
                # make 'rconc' case by removing relevant fact essential for inferring query from the theory
                # Find facts that are not in the relevant facts
                theory_index = relevant_facts[0].theory_index if relevant_facts else None
                non_relevant_facts = [fact for fact in facts if fact not in relevant_facts] + generate_dummy_fact(rules, facts, query, theory_index)
                theory = reconstruct_theory(rules, non_relevant_facts)
            elif example['proof_strategy'] == 'inv-proof':
                # make 'inv-rconc' case by removing relevant fact essential for inferring query from the theory
                # Find facts that are not in the relevant facts
                theory_index = relevant_facts[0].theory_index if relevant_facts else None
                non_relevant_facts = [fact for fact in facts if fact not in relevant_facts] + generate_dummy_fact(rules, facts, query, theory_index)
                theory = reconstruct_theory(rules, non_relevant_facts)
            elif example['proof_strategy'] == 'rconc':
                # make 'proof' case by adding relevant fact(s) to theory to make premise true
                # determine necessary facts to add to the theory
                facts_to_add = add_necessary_facts(relevant_rules, relevant_facts, query, num_reasoning_steps=1)
                # Add the necessary facts to the theory
                theory = reconstruct_theory(rules, facts + facts_to_add)
            elif example['proof_strategy'] == 'inv-rconc':
                # make 'inv-proof' case by adding relevant fact(s) to theory to make premise true
                # determine necessary facts to add to the theory
                facts_to_add = add_necessary_facts(relevant_rules, relevant_facts, query, num_reasoning_steps=1)
                # Add the necessary facts to the theory
                theory = reconstruct_theory(rules, facts + facts_to_add)
            else:
                raise ValueError(f"Unknown proof strategy: {example['proof_strategy']}")
            inputs.append(theory + '[SEP]' + str(query))
            labels.append(not example['label'])
        
        elif example['depth'] == 2:
            relevant_rules, relevant_facts, query = determine_relevant_rules_and_facts(rules, facts, query, depth=2)
            if example['proof_strategy'] == 'proof':
                # make 'rconc' case by removing relevant fact or rule essential for inferring query from the theory
                # Find facts that are not in the relevant facts
                theory_index = relevant_facts[0].theory_index if relevant_facts else None
                non_relevant_facts = [fact for fact in facts if fact not in relevant_facts] + generate_dummy_fact(rules, facts, query, theory_index)
                theory = reconstruct_theory(rules, non_relevant_facts)
            elif example['proof_strategy'] == 'inv-proof':
                # make 'inv-rconc' case by removing relevant fact essential for inferring query from the theory
                # Find facts that are not in the relevant facts
                theory_index = relevant_facts[0].theory_index if relevant_facts else None
                non_relevant_facts = [fact for fact in facts if fact not in relevant_facts] + generate_dummy_fact(rules, facts, query, theory_index)
                theory = reconstruct_theory(rules, non_relevant_facts)
            elif example['proof_strategy'] == 'rconc':
                # make 'proof' case by adding relevant fact(s) to theory to make premise true
                # determine necessary facts to add to the theory
                facts_to_add = add_necessary_facts(relevant_rules, facts, query, num_reasoning_steps=2)
                # Add the necessary facts to the theory
                theory = reconstruct_theory(rules, facts + facts_to_add)
            elif example['proof_strategy'] == 'inv-rconc':
                # make 'inv-proof' case by adding relevant fact(s) to theory to make premise true
                # determine necessary facts to add to the theory
                facts_to_add = add_necessary_facts(relevant_rules, facts, query.invert(), num_reasoning_steps=2)
                # Add the necessary facts to the theory
                theory = reconstruct_theory(rules, facts + facts_to_add)
            else:
                raise ValueError(f"Unknown proof strategy: {example['proof_strategy']}")
            inputs.append(theory + '[SEP]' + str(query))
            labels.append(not example['label'])

        elif example['depth'] > 2:
            inputs.append(input_text)
            labels.append(example['label'])

    return Dataset.from_dict({'input': inputs, 'label': labels})

def generate_dummy_fact(rules, facts, query, theory_index=None):
    """
    Generates a dummy fact that won't interfere with the entailment prediction.
    Uses an entity different from the inverted query's entity to avoid interference.
    
    Args:
        rules (list): List of Rule objects
        facts (list): List of Fact objects
        inverted_query (Query): The inverted query object
        
    Returns:
        list: A list containing one dummy Fact object
    """
    # List of possible entities and properties
    entities = ["Anne", "Bob", "Charlie", "Dave", "Erin", "Fiona", "Gary", "Harry"]
    properties = ["red", "blue", "green", "kind", "nice", "big", "cold", "young", 
                    "round", "rough", "white", "smart", "quiet", "furry"]
    
    # Filter out entities that match the inverted query to avoid interference
    available_entities = [entity for entity in entities if entity != query.entity]
    
    if not available_entities:
        available_entities = entities  # Fallback if somehow all entities are filtered out
    
    dummy_fact_unique = False
    while not dummy_fact_unique:
        # Choose a random entity and property
        entity = random.choice(available_entities)
        property = random.choice(properties)
        
        # Create a dummy fact
        dummy_fact = Fact(entity=entity, property=property, negative=False, theory_index=theory_index)

        dummy_fact_unique = not any(fact == dummy_fact for fact in facts)
    
    return [dummy_fact]

def add_necessary_facts(rules, facts, query, num_reasoning_steps=1):
    """
    Adds the necessary facts to make the query true based on the number of reasoning steps.
    
    Args:
        rules (list): List of Rule objects
        facts (list): List of Fact objects
        query (Query): The query to make true
        num_reasoning_steps (int): Number of reasoning steps (1 or 2)
        
    Returns:
        list: List of facts needed to add to make the query true
    """
    facts_to_add = []
    
    if num_reasoning_steps == 0:
        # For depth 0, just add the query itself as a fact
        fact_version = Fact(entity=query.entity, property=query.property, negative=query.negative)
        facts_to_add.append(fact_version)
        
    elif num_reasoning_steps == 1:
        # Find rules that can directly infer the query
        for rule in rules:
            # is_general_entity = rule.consequence_entity in ["[ANY_PERSON]", "[ANY_THING]"]
            # consequence_property_match = any(prop == query.property for prop in rule.consequence_property)
            
            # if ((rule.consequence_entity == query.entity or is_general_entity) and 
            #     consequence_property_match and
            #     rule.consequence_negative == query.negative):
                
            # For rules with multiple premises
            if len(rule.premises) > 1:
                for premise in rule.premises:
                    # Check if this premise is already in facts
                    premise_exists = any(
                        fact.entity == query.entity and 
                        fact.property == premise.property and
                        fact.negative == premise.negative
                        for fact in facts
                    )
                    
                    if not premise_exists:
                        # Create a fact from the premise
                        entity = query.entity if premise.entity in ["[ANY_PERSON]", "[ANY_THING]"] else premise.entity
                        facts_to_add.append(Fact(
                            entity=entity,
                            property=premise.property,
                            negative=premise.negative
                        ))
            # For rules with simple premise structure
            else:
                for prop in rule.premise_property:
                    # Check if this premise is already in facts
                    premise_exists = any(
                        fact.entity == query.entity and 
                        fact.property == prop and
                        fact.negative == rule.premise_negative
                        for fact in facts
                    )
                    
                    if not premise_exists:
                        facts_to_add.append(Fact(
                            entity=query.entity,
                            property=prop,
                            negative=rule.premise_negative
                        ))
                
            # Once we've added facts for one rule, we're done
            if facts_to_add:
                break
    
    elif num_reasoning_steps == 2:
        # First, find rules that directly infer the query
        final_rules = []
        for rule in rules:
            is_general_entity = rule.consequence_entity in ["[ANY_PERSON]", "[ANY_THING]"]
            consequence_property_match = any(prop == query.property for prop in rule.consequence_property)
            
            if ((rule.consequence_entity == query.entity or is_general_entity) and 
                consequence_property_match and
                rule.consequence_negative == query.negative):
                final_rules.append(rule)
        
        # For each final rule, find intermediate rules and necessary facts
        one_int_rule_found = False
        for final_rule in final_rules:
            # Identify the premises we need to make true
            needed_premises = []
            for prop in final_rule.premise_property:
                needed_premises.append(Fact(
                    # questionable? should needed premises not always have query.entity?
                    entity=query.entity if final_rule.premise_entity in ["[ANY_PERSON]", "[ANY_THING]"] else final_rule.premise_entity,
                    property=prop,
                    negative=final_rule.premise_negative
                ))
            
            for premise in needed_premises:
                premise_added = False
                        
                # Check if the premise already exists in facts
                if any(fact.entity == query.entity and 
                        fact.property == premise.property and
                        fact.negative == premise.negative
                        for fact in facts + facts_to_add):
                    premise_added = True
                    continue
                
                # Try to find an intermediate rule to derive this premise
                for int_rule in rules:
                    rule_is_general_entity = int_rule.consequence_entity in ["[ANY_PERSON]", "[ANY_THING]"]
                    premise_is_general_entity = premise.entity in ["[ANY_PERSON]", "[ANY_THING]"]
                    consequence_property_match = any(prop == premise.property for prop in int_rule.consequence_property)
                    
                    if ((int_rule.consequence_entity == premise.entity or rule_is_general_entity or premise_is_general_entity) and
                        consequence_property_match and
                        int_rule.consequence_negative == premise.negative):
                        
                        # Found a rule that can derive the premise
                        one_int_rule_found = True
                        int_needed_premises = []
                        if int_rule.premises:
                            int_needed_premises = int_rule.premises
                        else:
                            for prop in int_rule.premise_property:
                                int_needed_premises.append(Fact(
                                    entity=premise.entity if int_rule.premise_entity in ["[ANY_PERSON]", "[ANY_THING]"] else int_rule.premise_entity,
                                    property=prop,
                                    negative=int_rule.premise_negative
                                ))
                        
                        all_premises_exist = True
                        for int_premise in int_needed_premises:
                            if not any(fact.entity == query.entity and 
                                        fact.property == int_premise.property and
                                        fact.negative == int_premise.negative
                                        for fact in facts + facts_to_add):
                                facts_to_add.append(Fact(
                                    entity=query.entity,
                                    property=int_premise.property,
                                    negative=int_premise.negative
                                ))
                                all_premises_exist = False

                        if all_premises_exist or facts_to_add:
                            premise_added = True
                            break
                # If no rule can derive the premise, add it directly
                if not premise_added:
                    facts_to_add.append(premise)

            # If we've found a way to satisfy this rule using at least one intermediate rule and not directly satisfy a final rule, we're done
            if facts_to_add and one_int_rule_found and not added_facts_satisfy_a_final_rule(facts + facts_to_add, final_rules, query):
                break
            else:
                facts_to_add = []

    
    return facts_to_add

def added_facts_satisfy_a_final_rule(facts, final_rules, query):
    """
    Checks if the facts after facts_to_add was added directly satisfy any of the final rules,
    which would make the query immediately true without requiring an intermediate reasoning step/rule.
    
    Args:
        facts (list): List of Fact objects in the theory, including the newly added facts
        final_rules (list): List of Rule objects that directly conclude the query
        query (Fact): The query fact to be satisfied

    Returns:
        bool: True if any final rule is fully satisfied by facts, False otherwise
    """
    for rule in final_rules:
        all_premises_satisfied = True
        for premise in rule.premises:
            premise_satisfied = False
            for fact in facts:
                if (fact.entity == premise.entity or (premise.entity in ["[ANY_PERSON]", "[ANY_THING]"]) and fact.entity == query.entity) and \
                    fact.property == premise.property and \
                    fact.negative == premise.negative:
                    premise_satisfied = True
                    break
            if not premise_satisfied:
                all_premises_satisfied = False
                break
        
        if all_premises_satisfied:
            return True
    
    return False

def reconstruct_theory(rules, facts):
    """
    Reconstructs a theory text from a list of Rule and Fact objects.
    
    Args:
        rules (list): List of Rule objects
        facts (list): List of Fact objects
        
    Returns:
        str: The reconstructed theory text with statements in the original order
    """
    # Combine rules and facts for sorting
    all_statements = []
    
    # Add rules to the list
    for rule in rules:
        # if hasattr(rule, 'theory_index') and rule.theory_index is not None:
        all_statements.append((rule.theory_index, str(rule)))
    
    # Add facts to the list
    for fact in facts:
        # if hasattr(fact, 'theory_index') and fact.theory_index is not None:
        all_statements.append((fact.theory_index, str(fact)))
    
    # Sort by index
    all_statements.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))
    
    # Join all statements into a single string
    theory_text = ' '.join([statement for _, statement in all_statements])
    
    # If no statements had indices, just concatenate them in the order given
    if not all_statements:
        theory_text = ' '.join([str(rule) for rule in rules] + [str(fact) for fact in facts])
    
    return theory_text

def extract_rules_and_facts(input_sequence):
    """
    Extracts rules and facts from a text entailment input sequence.
    
    Args:
        input_sequence (str): The input text containing rules and facts, followed by a query after [SEP].
    
    Returns:
        tuple: (rules, facts, query) where:
            - rules (list): List of rule statements (if-then statements)
            - facts (list): List of factual statements
            - query (str): The query statement that follows [SEP]
    """
    if '[SEP]' not in input_sequence:
        raise ValueError("Input sequence must contain [SEP] to separate premises from query")
    
    theory, query = input_sequence.split('[SEP]', 1)
    theory = theory.strip()
    query = query.split('.')[0].strip()
    
    # Split premises into individual statements
    statements = [s.strip() for s in theory.split('.') if s.strip()]
    
    rules = []
    facts = []
    
    for i, statement in enumerate(statements):
        # Check if statement is a rule (contains "if", "then", "all", etc.)
        if any(rule_indicator in statement.lower() for rule_indicator in ["if", "then", "all", "any", "every", "when"]):
            rules.append((statement, i))
        # Check for implicit rule format: "<adjective> <plural noun> are <property>"
        elif " are " in statement and any(plural in statement.lower() for plural in ["people", "things", "dogs", "cats", "animals"]):
            # Make sure it's not a specific entity (fact about specific people)
            parts = statement.split(" are ", 1)
            if not parts[0].strip()[0].isupper() or "people" in parts[0].lower() or "things" in parts[0].lower():
                rules.append((statement, i))
            else:
                facts.append((statement, i))
        else:
            facts.append((statement, i))
    
    return rules, facts, query

class Fact:
    def __init__(self, entity=None, property=None, negative=False, is_query=False, theory_index=None):
        self.entity = entity        # The subject of the fact (e.g., "John", "Mary")
        self.property = property    # The property of the subject (e.g., "is tall", "likes cake")
        self.negative = negative    # Whether the fact is negated (e.g., "John is not tall")
        self.is_query = is_query    # Whether this fact is a query
        self.theory_index = theory_index    # Index of the fact in the list of the theory's statements (if applicable)

    @classmethod
    def from_string(cls, statement, theory_index=None, is_query=False):
        statement = statement.strip()
        
        # Check if statement is negative
        negative = False
        if "not" in statement.lower():
            negative = True
        
        # Basic parsing - split on first verb or "is"/"are"
        parts = statement.split(" is ", 1) if " is " in statement else statement.split(" are ", 1)
        
        if len(parts) == 2:
            entity = parts[0].strip()
            property = parts[1].strip()
            
            # Handle "not" in the property part
            if "not" in property:
                property = property.replace("not", "").strip()
                negative = True
                
            return cls(entity=entity, property=property, negative=negative, is_query=is_query, theory_index=theory_index)
        else:
            # Fallback for more complex statements
            return cls(entity=statement, property="", negative=negative, is_query=is_query, theory_index=theory_index)
        
    def invert(self):
        """Inverts the fact (e.g., "John is tall" becomes "John is not tall")."""
        if self.is_query:
            return Query(entity=self.entity, property=self.property, negative=not self.negative)
        else:
            return Fact(entity=self.entity, property=self.property, negative=not self.negative, theory_index=self.theory_index)
    
    def __str__(self):
        if not self.property:
            return self.entity
        
        connector = "is not" if self.negative else "is"
        return f"{self.entity} {connector} {self.property}."

    def __eq__(self, other):
        """
        Determine if two Facts are equal based on entity and property.
        Two facts are considered equal if they have the same entity and property,
        regardless of whether they are negative or positive.
        
        Args:
            other (Fact): The other Fact object to compare with
            
        Returns:
            bool: True if the facts have the same entity and property, False otherwise
        """
        if not isinstance(other, Fact):
            return False
        
        return self.entity == other.entity and self.property == other.property and self.negative == other.negative

class Rule:
    def __init__(self, premise_entity=None, premise_property=None, premise_negative=False,
                 consequence_entity=None, consequence_property=None, consequence_negative=False,
                 format_type=None, premises=None, theory_index=None):
        self.premise_entity = premise_entity
        # Convert premise_property to a list if it's not already
        self.premise_property = premise_property if isinstance(premise_property, list) else ([premise_property] if premise_property else [])
        self.premise_negative = premise_negative
        self.consequence_entity = consequence_entity
        # Convert consequence_property to a list if it's not already
        self.consequence_property = consequence_property if isinstance(consequence_property, list) else ([consequence_property] if consequence_property else [])
        self.consequence_negative = consequence_negative
        self.format_type = format_type  # To store the format type of the rule (if-then, all-are, etc.)
        self.premises = premises or []  # For multiple conditions (e.g., "if X is Y and X is Z then...")
        self.theory_index = theory_index    # Index of the fact in the list of the theory's statements (if applicable)

    @classmethod
    def from_string(cls, statement, theory_index=None):
        statement = statement.strip().rstrip('.')
        
        # Handle different rule formats
        if "if" in statement.lower() and "then" in statement.lower():
            # Format: "If X is Y, then Z is W"
            parts = statement.lower().split("if ", 1)[1].split(" then ", 1)
            premise_str = parts[0].strip().rstrip(',')
            consequence_str = parts[1].strip()
            
            # Special case for generic entities with multiple properties
            if (premise_str.lower().startswith("someone is") or premise_str.lower().startswith("something is")) and " and " in premise_str:
                entity = "someone" if premise_str.lower().startswith("someone") else "something"
                # Extract all properties after "is"
                properties_part = premise_str.lower().split(f"{entity} is", 1)[1].strip()
                
                # Split properties on "and"
                property_list = [p.strip() for p in properties_part.split(" and ")]
                
                # Create premises for each property
                premises = []
                for prop in property_list:
                    premises.append(Fact(entity="[ANY_PERSON]" if entity == "someone" else "[ANY_THING]", 
                                       property=prop, negative=False))
                
                # Parse consequence
                consequence_fact = Fact.from_string(consequence_str)
                
                # Handle "they" or other pronouns in consequence
                if consequence_fact.entity.lower() in ["they", "he", "she", "it", "them"]:
                    consequence_fact.entity = "[ANY_PERSON]" if entity == "someone" else "[ANY_THING]"
                
                rule = cls(
                    premise_entity="[ANY_PERSON]" if entity == "someone" else "[ANY_THING]",
                    premise_property=property_list,
                    premise_negative=False,
                    consequence_entity=consequence_fact.entity,
                    consequence_property=[consequence_fact.property],
                    consequence_negative=consequence_fact.negative,
                    format_type="if_then",
                    premises=premises,
                    theory_index=theory_index
                )
                return rule
            # Handle multiple conditions in premise for other cases
            elif " and " in premise_str:
                conditions = premise_str.split(" and ")
                premises = []
                property_list = []
                
                for condition in conditions:
                    fact = Fact.from_string(condition.strip())
                    premises.append(fact)
                    property_list.append(fact.property)
                
                # Get the last entity for consistency
                premise_entity = premises[-1].entity if premises else "[UNKNOWN]"
                
                # Parse consequence
                consequence_fact = Fact.from_string(consequence_str)
                
                # Handle "they" or other pronouns in consequence
                if consequence_fact.entity.lower() in ["they", "he", "she", "it", "them"]:
                    consequence_fact.entity = premise_entity
                
                rule = cls(
                    premise_entity=premise_entity,
                    premise_property=property_list,
                    premise_negative=premises[0].negative if premises else False,
                    consequence_entity=consequence_fact.entity,
                    consequence_property=[consequence_fact.property],
                    consequence_negative=consequence_fact.negative,
                    format_type="if_then",
                    premises=premises,
                    theory_index=theory_index
                )
                return rule
            else:
                # Single condition
                premise_fact = Fact.from_string(premise_str)
                consequence_fact = Fact.from_string(consequence_str)
                
                # # Handle "they" or other pronouns in consequence
                # if consequence_fact.entity.lower() in ["they", "he", "she", "it", "them"]:
                #     consequence_fact.entity = premise_fact.entity

                # Handle "they" or other pronouns in consequence
                if consequence_fact.entity.lower() in ["they", "he", "she", "it", "them"]:
                    consequence_fact.entity = "[ANY_PERSON]" if premise_fact.entity == "someone" else "[ANY_THING]"
                if premise_fact.entity.lower() == "someone":
                    premise_fact.entity = "[ANY_PERSON]"
                elif premise_fact.entity.lower() == "something":
                    premise_fact.entity = "[ANY_THING]"
                rule = cls(
                    premise_entity=premise_fact.entity,
                    premise_property=[premise_fact.property],
                    premise_negative=premise_fact.negative,
                    consequence_entity=consequence_fact.entity,
                    consequence_property=[consequence_fact.property],
                    consequence_negative=consequence_fact.negative,
                    format_type="if_then",
                    premises=[premise_fact],
                    theory_index=theory_index
                )
                return rule
        
        # Handle "All X are Y" format
        elif statement.lower().startswith("all "):
            parts = statement.lower().split("all ", 1)[1].split(" are ", 1)
            entity_desc = parts[0].strip()  # e.g., "nice, cold people"
            property_type = parts[1].strip()  # e.g., "rough"
            
            # Handle entity with multiple properties
            entity_base = "people" if "people" in entity_desc else "things"
            properties = [p.strip() for p in entity_desc.replace(entity_base, "").split(",") if p.strip()]
            
            # Create a fact for each property
            premises = []
            for prop in properties:
                premises.append(Fact(entity="[ANY_PERSON]" if entity_base == "people" else "[ANY_THING]", 
                                     property=prop.strip(), negative=False))
            
            consequence = Fact(entity="[ANY_PERSON]" if entity_base == "people" else "[ANY_THING]", 
                              property=property_type, negative=False)
            
            return cls(
                premise_entity="[ANY_PERSON]" if entity_base == "people" else "[ANY_THING]",
                premise_property=properties,
                consequence_entity="[ANY_PERSON]" if entity_base == "people" else "[ANY_THING]",
                consequence_property=[property_type],
                format_type="all_are",
                premises=premises,
                theory_index=theory_index
            )
        
        # Handle implied "All" format (e.g., "Nice people are blue")
        elif " are " in statement.lower() or " is " in statement.lower():
            separator = " are " if " are " in statement.lower() else " is "
            parts = statement.lower().split(separator, 1)
            entity_desc = parts[0].strip()  # e.g., "nice people"
            property_type = parts[1].strip()  # e.g., "blue"
            
            entity_base = "people" if "people" in entity_desc else "things"
            property_desc = entity_desc.replace(entity_base, "").strip()
            
            # Split properties if there are multiple (comma or and separated)
            property_list = []
            if "," in property_desc or " and " in property_desc:
                # Replace " and " with "," for consistent splitting
                property_desc = property_desc.replace(" and ", ", ")
                property_list = [p.strip() for p in property_desc.split(",") if p.strip()]
            else:
                property_list = [property_desc]
                
            premises = []
            for prop in property_list:
                premises.append(Fact(entity="[ANY_PERSON]" if entity_base == "people" else "[ANY_THING]", 
                              property=prop, negative=False))
            
            return cls(
                premise_entity="[ANY_PERSON]" if entity_base == "people" else "[ANY_THING]",
                premise_property=property_list,
                consequence_entity="[ANY_PERSON]" if entity_base == "people" else "[ANY_THING]",
                consequence_property=[property_type],
                format_type="implied_all",
                premises=premises,
                theory_index=theory_index
            )
        
        # Fallback for other rule formats
        else:
            return cls(
                premise_entity="[UNKNOWN]",
                premise_property=[statement],
                consequence_entity="[UNKNOWN]",
                consequence_property=["[UNKNOWN]"],
                format_type="unknown",
                theory_index=theory_index
            )
    
    def __str__(self):
        if self.format_type == "if_then":
            if len(self.premises) > 1:
                # Multiple conditions
                premise_strs = []
                premise = self.premises[0]
                if premise.entity == "[ANY_PERSON]" or premise.entity == "[ANY_THING]":
                    if premise.entity == "[ANY_PERSON]":
                        premise_strs.append(f"someone is{' not' if premise.negative else ''} {premise.property}")
                    elif premise.entity == "[ANY_THING]":
                        premise_strs.append(f"something is{' not' if premise.negative else ''} {premise.property}")
                    for premise in self.premises[1:]:
                        if premise.entity == "[ANY_PERSON]":
                            premise_strs.append(f"{premise.property}")
                        elif premise.entity == "[ANY_THING]":
                            premise_strs.append(f"{premise.property}")
                else:
                    for premise in self.premises:
                        premise_strs.append(f"{premise.entity} is{' not' if premise.negative else''} {premise.property}")
                premise_str = " and ".join(premise_strs)
                
            else:
                # Single condition
                if self.premise_entity == "[ANY_PERSON]":
                    premise_str = f"someone is{' not' if self.premise_negative else''} {' and '.join(self.premise_property)}"
                elif self.premise_entity == "[ANY_THING]":
                    premise_str = f"something is{' not' if self.premise_negative else''} {' and '.join(self.premise_property)}"
                else:
                    premise_str = f"{self.premise_entity} is{' not' if self.premise_negative else''} {' and '.join(self.premise_property)}"
            
            if self.consequence_entity == "[ANY_PERSON]":
                consequence_str = f"they are{' not' if self.consequence_negative else''} {' and '.join(self.consequence_property)}"
            elif self.consequence_entity == "[ANY_THING]":
                consequence_str = f"it is{' not' if self.consequence_negative else''} {' and '.join(self.consequence_property)}"
            else:
                consequence_str = f"{self.consequence_entity} is{' not' if self.consequence_negative else''} {' and '.join(self.consequence_property)}"
            
            return f"If {premise_str} then {consequence_str}."
        
        elif self.format_type == "all_are" or self.format_type == "implied_all":
            # Construct entity description with properties
            entity_desc = ", ".join(self.premise_property)
            
            entity_base = "people" if self.premise_entity == "[ANY_PERSON]" else "things"
            
            if self.format_type == "all_are":
                return f"All {entity_desc} {entity_base} are {' and '.join(self.consequence_property)}."
            else:
                return f"{entity_desc.capitalize()} {entity_base} are {' and '.join(self.consequence_property)}."
        
        else:
            return "Unknown rule format."



class Query(Fact):
    def __init__(self, entity=None, property=None, negative=False):
        super().__init__(entity=entity, property=property, negative=negative, is_query=True)
    
    @classmethod
    def from_string(cls, statement):
        return Fact.from_string(statement, is_query=True)


def parse_input_sequence(input_sequence):
    """
    Parse an input sequence into structured Rule, Fact, and Query objects.
    
    Args:
        input_sequence (str): The input text containing rules and facts, followed by a query after [SEP].
    
    Returns:
        tuple: (rules, facts, query) where:
            - rules (list): List of Rule objects
            - facts (list): List of Fact objects
            - query (Query): The query object
    """
    raw_rules, raw_facts, raw_query = extract_rules_and_facts(input_sequence)
    
    rules = [Rule.from_string(rule_str, theory_index=theory_index) for (rule_str, theory_index) in raw_rules]
    facts = [Fact.from_string(fact_str, theory_index=theory_index) for (fact_str, theory_index) in raw_facts]
    query = Query.from_string(raw_query)
    
    return rules, facts, query

def determine_relevant_rules_and_facts(rules, facts, query, depth=0):
    """
    Determines which rules and facts are relevant to a query based on the specified depth.
    
    Args:
        input_sequence (str): The input text containing rules, facts, and query.
        depth (int): The depth of reasoning required (0, 1, or 2).
    
    Returns:
        tuple: (relevant_rules, relevant_facts, query) 
    """
    # Parse the input sequence into structured objects
    # rules, facts, query = parse_input_sequence(input_sequence)
    relevant_rules = []
    relevant_facts = []
    
    # Depth 0: Only consider the fact that is exactly the query or its inverse
    if depth == 0:
        for fact in facts:
            # Check if fact matches query exactly or is its inverse
            if (fact.entity == query.entity and fact.property == query.property):
                relevant_facts.append(fact)
    
    # Depth 1: Consider rules that directly lead to the query
    elif depth == 1:
        # Find rules that can directly infer the query
        for rule in rules:
            # Check if rule consequence matches query entity and property
            is_general_entity = rule.consequence_entity in ["[ANY_PERSON]", "[ANY_THING]"]
            consequence_property_match = False
            
            # Check if any consequence property matches the query property
            for prop in rule.consequence_property:
                if prop == query.property:
                    consequence_property_match = True
                    break
            
            if ((rule.consequence_entity == query.entity or is_general_entity) and 
                consequence_property_match):
                relevant_rules.append(rule)
        
        # Find facts that satisfy the premises of these rules
        for rule in relevant_rules:
            for fact in facts:
                # Handle different rule types
                is_general_premise = rule.premise_entity in ["[ANY_PERSON]", "[ANY_THING]"]
                
                # For rules with multiple premises
                if rule.premises:
                    for premise in rule.premises:
                        if ((fact.entity == query.entity) and
                            fact.property == premise.property and
                            fact.negative == premise.negative):
                            if fact not in relevant_facts:
                                relevant_facts.append(fact)
                # For rules with simple premise structure
                else:
                    for prop in rule.premise_property:
                        if ((fact.entity == query.entity) and
                            fact.property == prop and
                            fact.negative == rule.premise_negative):
                            if fact not in relevant_facts:
                                relevant_facts.append(fact)
    
    # Depth 2: Consider two-step reasoning
    elif depth == 2:
        # First, find rules that directly infer the query (final rules)
        final_rules = []
        for rule in rules:
            is_general_entity = rule.consequence_entity in ["[ANY_PERSON]", "[ANY_THING]"]
            consequence_property_match = any(prop == query.property for prop in rule.consequence_property)
            
            if ((rule.consequence_entity.lower() == query.entity.lower() or is_general_entity) and 
                consequence_property_match):
                # and rule.consequence_negative == query.negative):
                final_rules.append(rule)
                relevant_rules.append(rule)
        
        # Find rules that can infer the premises of the final rules (intermediate rules)
        intermediate_rules = []
        for final_rule in final_rules:
            premises_to_check = final_rule.premises if final_rule.premises else [
                Fact(entity=final_rule.premise_entity, property=prop, negative=final_rule.premise_negative)
                for prop in final_rule.premise_property
            ]
            
            for premise in premises_to_check:
                for rule in rules:
                    rule_is_general_entity = rule.consequence_entity in ["[ANY_PERSON]", "[ANY_THING]"]
                    premise_is_general_entity = premise.entity in ["[ANY_PERSON]", "[ANY_THING]"]
                    consequence_property_match = any(prop == premise.property for prop in rule.consequence_property)
                    
                    if ((rule.consequence_entity == premise.entity or rule_is_general_entity or premise_is_general_entity) and
                        consequence_property_match and
                        rule.consequence_negative == premise.negative):
                        intermediate_rules.append(rule)
                        if rule not in relevant_rules:
                            relevant_rules.append(rule)
        
        # Find facts that satisfy the premises of the intermediate rules
        for rule in intermediate_rules:
            
            # Handle different rule types
            is_general_premise = rule.premise_entity in ["[ANY_PERSON]", "[ANY_THING]"]
            
            # For rules with multiple premises
            if rule.premises:
                rule_relevant_facts = []
                all_premises_satisfied = True
                for premise in rule.premises:
                    premise_satisfied = False
                    for fact in facts:
                        if ((fact.entity.lower() == premise.entity.lower() or is_general_premise) and
                            fact.entity.lower() == query.entity.lower() and
                            fact.property == premise.property and
                            fact.negative == premise.negative):
                            premise_satisfied = True
                            if fact not in relevant_facts:
                                rule_relevant_facts.append(fact)
                    if not premise_satisfied:
                        all_premises_satisfied = False
                        break
                if all_premises_satisfied:
                    if len(rule_relevant_facts) > 1:
                        relevant_facts.append(random.choice(rule_relevant_facts))
                    else:
                        relevant_facts.extend(rule_relevant_facts)
            # For rules with simple premise structure
            else:
                for prop in rule.premise_property:
                    if ((fact.entity == rule.premise_entity or is_general_premise) and
                        fact.property == prop and
                        fact.negative == rule.premise_negative):
                        if fact not in relevant_facts:
                            relevant_facts.append(fact)
    
    return relevant_rules, relevant_facts, query

def get_all_text_entailment_things(model_name, test_dataset, num_examples, device, metric_name, kl_return_one_element=True, max_length=None):
    tl_model = get_finetuned_bert_model(model_name, device)

    if len(test_dataset) < 2 * num_examples:
        raise ValueError("The test dataset must contain at least 2 * num_examples examples.")

    validation_examples = test_dataset.select(range(num_examples))
    test_examples = test_dataset.select(range(num_examples, 2 * num_examples))

    # validation_examples = validation_examples.map(remove_special_tokens)
    # test_examples = test_examples.map(remove_special_tokens)
    print("generating corrupted examples")
    # Generate corrupted examples for validation and test sets
    corrupted_validation_examples = generate_corrupt_examples(validation_examples)
    corrupted_test_examples = generate_corrupt_examples(test_examples)

    # Concatenate normal and corrupted examples for validation and test sets
    combined_validation_examples = concatenate_datasets([validation_examples, corrupted_validation_examples])
    combined_test_examples = concatenate_datasets([test_examples, corrupted_test_examples])

    # Tokenize the combined datasets
    tokenized_combined_validation = tokenize_function(tl_model.tokenizer, combined_validation_examples, padding='max_length' if max_length else 'longest')
    tokenized_combined_test = tokenize_function(tl_model.tokenizer, combined_test_examples, padding='max_length' if max_length else 'longest')

    # Split the tokenized outputs back into normal and corrupted examples
    tokenized_validation = {
        "input_ids": tokenized_combined_validation["input_ids"][:num_examples],
        "attention_mask": tokenized_combined_validation["attention_mask"][:num_examples],
        "label": tokenized_combined_validation["label"][:num_examples]
    }
    tokenized_corrupted_validation = {
        "input_ids": tokenized_combined_validation["input_ids"][num_examples:],
        "attention_mask": tokenized_combined_validation["attention_mask"][num_examples:],
        "label": tokenized_combined_validation["label"][num_examples:]
    }

    tokenized_test = {
        "input_ids": tokenized_combined_test["input_ids"][:num_examples],
        "attention_mask": tokenized_combined_test["attention_mask"][:num_examples],
        "label": tokenized_combined_test["label"][:num_examples]
    }
    tokenized_corrupted_test = {
        "input_ids": tokenized_combined_test["input_ids"][num_examples:],
        "attention_mask": tokenized_combined_test["attention_mask"][num_examples:],
        "label": tokenized_combined_test["label"][num_examples:]
    }

    # tokenized_validation = tokenize_function(tl_model.tokenizer, validation_examples, padding='max_length' if max_length else True)
    # tokenized_corrupted_validation = tokenize_function(tl_model.tokenizer, corrupted_validation_examples, padding='max_length' if max_length else True)
    # tokenized_test = tokenize_function(tl_model.tokenizer, test_examples, padding='max_length' if max_length else True)
    # tokenized_corrupted_test = tokenize_function(tl_model.tokenizer, corrupted_test_examples, padding='max_length' if max_length else True)

    validation_data = tokenized_validation["input_ids"]
    validation_mask = tokenized_validation["attention_mask"]
    validation_patch_data = tokenized_corrupted_validation["input_ids"]
    validation_labels = validation_examples["label"]
    validation_wrong_labels = corrupted_validation_examples["label"]

    test_data = tokenized_test["input_ids"]
    test_mask = tokenized_test["attention_mask"]
    test_patch_data = tokenized_corrupted_test["input_ids"]
    test_labels = test_examples["label"]
    test_wrong_labels = corrupted_test_examples["label"]

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
    elif metric_name == "custom_logit_diff":
        validation_metric = partial(
            custom_logit_diff_metric_entailment,
            correct_labels=validation_labels,
            wrong_labels=validation_wrong_labels,
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
        "custom_logit_diff": partial(
            custom_logit_diff_metric_entailment,
            correct_labels=test_labels,
            wrong_labels=test_wrong_labels,
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
