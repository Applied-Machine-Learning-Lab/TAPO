from openai import OpenAI
import argparse
import random
from rich import print

from TAPO import gsm
from TAPO.mutation_prompts import mutation_prompts
from TAPO.thinking_styles import thinking_styles
from TAPO.run import init_run, run_for_n, create_population

# dataset in JSONL format, please put the path of your dataset here
dataset_examples = gsm.read_jsonl('YOUR_PATH.jsonl')

# the API of OpenAI, put your API here, and write your model id.
client = OpenAI(api_key='YOUR_API_KEY')
model_id = "YOUR_MODEL_ID"

# definition of TAPO
parser = argparse.ArgumentParser(description='Run the TAPO Algorithm. Number of units is mp * ts.')
parser.add_argument('-mp', '--num_mutation_prompts', type=int, default=2)
parser.add_argument('-ts', '--num_thinking_styles', type=int, default=4)
parser.add_argument('-e', '--num_evals', type=int, default=50)
parser.add_argument('-n', '--simulations', type=int, default=3)
parser.add_argument('-p', '--problem', default="Solve the math word problem.")

# choose thinking styles, mutation prompts and dataset examples
args = vars(parser.parse_args())
tp_set = random.sample(mutation_prompts, int(args['num_mutation_prompts']))
mutator_set = random.sample(thinking_styles, int(args['num_thinking_styles']))
question_set = random.sample(dataset_examples, int(args['num_evals']))

# create population
p = create_population(tp_set=tp_set, mutator_set=mutator_set, problem_description=args['problem'])

# start evolution
init_run(p, client, model_id, question_set, int(args['num_evals']))
run_for_n(int(args['simulations']), p, client, model_id, question_set, int(args['num_evals']))

print("done processing! final gen:")
print(p.units)
