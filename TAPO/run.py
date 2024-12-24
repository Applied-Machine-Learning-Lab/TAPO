from openai import OpenAI
from rich import print
import os
from typing import List
from datetime import datetime

from TAPO.mutation_operators import mutate
from TAPO.types import EvolutionUnit, Population
from TAPO.evaluate import _evaluate_fitness

answer_request = "Answer the math question with consise solution process and The solution should end with a final answer in the format '\n#### number'. eg.'A pen costs $1.20 + $0.30 = $<<1.20+0.30=1.50>>1.50.\nSo, 8 pens cost $1.50 x 8 = $<<8*1.5=12>>12.\n#### 12'"


def create_population(tp_set: List, mutator_set: List, problem_description: str):
    data = {
        'size': len(tp_set) * len(mutator_set),
        'problem_description': problem_description,
        'elites': [],
        'units': [EvolutionUnit(**{
            'T': t,
            'M': m,
            'P': '',
            'fitness': 0,
            'history': []
        }) for t in tp_set for m in mutator_set],
        'result_dir': ''
    }
    return Population(**data)


def init_run(population: Population, client: OpenAI, model_id, question_set, num_evals):
    # Initialise the current time to store the result
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(os.getcwd(), "Output")
    current_time_dir = os.path.join(output_dir, now)
    if not os.path.exists(current_time_dir):
        os.makedirs(current_time_dir)
    population.result_dir = current_time_dir

    # Initialize prompt
    prompts = []
    for unit in population.units:
        template = f"{unit.T} {unit.M} INSTRUCTION: {population.problem_description} INSTRUCTION MUTANT = "
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": template,
                },
            ],
        )
        prompts.append(completion.choices[0].message.content)
    # Store the generated prompt to population
    assert len(prompts) == population.size
    for i, item in enumerate(prompts):
        population.units[i].P = item
    print("done initial prompting")

    # result processing
    result_list = []
    for unit in population.units:
        for index, example in enumerate(question_set):
            description = example['question']
            completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": answer_request + unit.P + description}, ],
            )
            result_list.append(completion.choices[0].message.content)
    print("done initialization\n")
    # result evaluating
    _evaluate_fitness(population, result_list, question_set, num_evals)
    print("done initial evaluation")
    return population


def run_for_n(n: int, population: Population, client: OpenAI, model_id, question_set, num_evals):
    # Runs the genetic algorithm for n generations
    for i in range(n):
        print(f"================== Population {i} ================== \n")

        # Perform a variation operation to generate a new prompt word
        mutate(population, client)
        print("done mutation\n")

        # result processing
        result_list = []
        for unit in population.units:
            for index, example in enumerate(question_set):
                description = example['question']
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "user", "content": answer_request + unit.P + description},],
                )
                result_list.append(completion.choices[0].message.content)
        print(f"done processing {i}\n")

        # results evaluating
        _evaluate_fitness(population, result_list, question_set, num_evals)
        print(f"done evaluation {i}")

    return population
