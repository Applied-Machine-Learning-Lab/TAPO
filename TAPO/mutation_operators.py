import random
import re
from TAPO.types import Population, EvolutionUnit
from typing import List
from TAPO.thinking_styles import thinking_styles
from TAPO import gsm
from openai import OpenAI
from dotenv import load_dotenv
from rich import print

load_dotenv()
# Dataset in JSON format, please put the path of your dataset here
dataset_examples = gsm.read_jsonl('YOUR_PATH.jsonl')


# Direct Mutation mutators
def zero_order_prompt_gen(unit: EvolutionUnit, problem_description: str, model: OpenAI, model_id, **kwargs):
    """Generates a new task-prompt P by concatenating the problem description D with the prompt 
    'a list of 100 hints:'. New task-prompt P is the first generated hint.
    """
    result = model.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": problem_description + " An ordered list of 100 hints: ",
            },
        ],
    )

    # Search for the pattern "anything after 1. and before 2."
    pattern = r"1\.(.*?)2\."
    match = re.search(pattern, result.choices[0].message.content, re.DOTALL)

    if match:
        # return the first match
        unit.P = match.group(1).strip()
    else:
        unit.P = ""
    return unit


def first_order_prompt_gen(unit: EvolutionUnit, model: OpenAI, model_id, **kwargs):
    """Concatenate the mutation prompt M to the parent task-prompt P and pass it to the LLM to produce P
    """
    completion = model.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": unit.M + " " + unit.P,
            },
        ],
    )
    unit.P = completion.choices[0].message.content
    return unit


def lineage_based_mutation(unit: EvolutionUnit, elites: List[EvolutionUnit], model: OpenAI, model_id, **kwargs):
    """Using the stored history of best units, provide the LLM this list in chronological order to produce a novel prompt as continuation.
    """
    HEADING = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY \n "
    # made a choice not to format it with newlines, could change later.
    ITEMS = "\n".join(["{}. {}".format(i + 1, x.P) for i, x in enumerate(elites)])
    completion = model.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": HEADING + ITEMS,
            },
        ],
    )
    unit.P = completion.choices[0].message.content
    return unit


# Hypermutation
def zero_order_hypermutation(unit: EvolutionUnit, problem_description, model: OpenAI, model_id, **kwargs):
    """ Concatenate the original problem_description to a randomly sampled thinking-style and feed it to the LLM to generate a new mutation-prompt.
    """
    RANDOM_THINKING_STYLE = random.sample(thinking_styles, 1)[0]
    completion = model.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": problem_description + " " + RANDOM_THINKING_STYLE,
            },
        ],
    )
    unit.M = completion.choices[0].message.content
    return unit


def first_order_hypermutation(unit: EvolutionUnit, model: OpenAI, model_id, **kwargs):
    """ Concatenate the hyper-mutation prompt "Please summarize and improve the following instruction:"
    to a mutation-prompt to that the LLM generates a new mutation-prompt. This new mutation-prompt is then 
    instantly applied to the task-prompt of that unit.
    """
    HYPER_MUTATION_PROMPT = "Please summarize and improve the following instruction: "
    completion_M = model.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": HYPER_MUTATION_PROMPT + unit.M,
            },
        ],
    )
    unit.M = completion_M.choices[0].message.content
    completion_P = model.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": unit.M + " " + unit.P,
            },
        ],
    )
    unit.P = completion_P.choices[0].message.content
    return unit


# Lamarckian Mutation
def working_out_task_prompt(unit: EvolutionUnit, model: OpenAI, model_id, **kwargs):
    """ A 'lamarckian' mutation operator similar to instruction induction in APE.
    As far as I can understand, give it both the Q and A from the gsm8k dataset, 
    concatenated between 'I gave a friend an instruction and some advice. Here
    are the correct examples of his workings out ' and 'The instruction was: '
    The idea is to let the LLM reverse-engineer the task-prompt.
    """
    RANDOM_WORKING_OUT = random.sample(dataset_examples, 1)[0]
    completion = model.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content":
                    "I gave a friend an instruction and some advice. Here are the correct examples of his workings out " +
                    RANDOM_WORKING_OUT['question'] + " " + RANDOM_WORKING_OUT['answer'] + " The instruction was: "
                ,
            },
        ],
    )
    unit.P = completion.choices[0].message.content
    return unit


MUTATORS = [
    zero_order_prompt_gen,
    first_order_prompt_gen,
    lineage_based_mutation,
    zero_order_hypermutation,
    first_order_hypermutation,
    working_out_task_prompt
]


def mutate(population: Population, model: OpenAI) -> Population:
    # make an index pair
    indices = [i for i in range(len(population.units))]
    random.shuffle(indices)
    pairs = [indices[2 * x:2 * x + 2] for x in range(len(indices) // 2)]

    # Binary tournament genetic algorithm
    for i in range(len(pairs)):
        first_unit = population.units[pairs[i][0]]
        second_unit = population.units[pairs[i][1]]

        if first_unit.fitness >= second_unit.fitness:
            # winner gets mutated.
            mutation_input = first_unit
        else:
            mutation_input = second_unit
        data = {
            'unit': mutation_input,
            'model': model,
            'elites': population.elites,
            'problem_description': population.problem_description,
            'model_id': "gpt-4o-2024-08-06"
        }
        random_mutator = random.sample(MUTATORS, 1)[0]
        print(f"mutating:{i}")
        # print(f"MUTATING: {mutation_input} with {random_mutator.__name__}")
        random_mutator(**data)
    return population
