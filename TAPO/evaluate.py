import re
import os
from datetime import datetime
import csv
import numpy as np
import torch
from rich import print
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.util import ngrams
from typing import Tuple
from rouge import Rouge
import nltk

from TAPO.types import Population


# initialization of model used for evaluation
model = SentenceTransformer('all-MiniLM-L6-v2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-large')


def get_embedding(text: str) -> list:
    # Get the embeddings of text
    return model.encode(text)


def calculate_similarity(embedding1: list, embedding2: list) -> float:
    # Calculates the cosine similarity between two embedded vectors
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity


def calculate_perplexity(text: str) -> float:
    # An index for calculating the difficulty of text
    tokenize_input = gpt2_tokenizer.encode(text, return_tensors="pt")
    loss = gpt2_model(tokenize_input, labels=tokenize_input).loss
    return torch.exp(loss).item()


def evaluate_fluency(generated_answer: str, fluency_threshold: float = 5.0) -> bool:
    # Evaluate the fluency of the generated answer
    perplexity = calculate_perplexity(generated_answer)
    return perplexity <= fluency_threshold


def calculate_ngram_diversity(text: str, n: int = 2) -> float:
    # Calculate the n-gram diversity of a given text
    words = text.split()
    if len(words) < n:
        # If the number of words is less than n, an n-gram cannot be formed
        return 0.0  # 如果单词数量少于n，无法形成n-gram

    all_ngrams = list(ngrams(words, n))
    unique_ngrams = set(all_ngrams)

    if len(all_ngrams) == 0:
        return 0.0

    diversity = len(unique_ngrams) / len(all_ngrams)
    return diversity


def calculate_complexity(text: str) -> Tuple[int, int, int]:
    # Calculates the complexity of the text, including length, syntactic complexity and logical steps
    words = text.split()
    length = len(words)

    # Calculating syntactic complexity (e.g. number of clauses)
    sentences = nltk.sent_tokenize(text)
    grammar_complexity = sum(len(nltk.word_tokenize(sentence)) for sentence in sentences)

    # Logical steps (can be defined for specific tasks)
    logic_steps = text.count('.')

    return length, grammar_complexity, logic_steps


def evaluate_relevance_coverage_and_overlap(generated_answer: str, correct_answer: str) -> Tuple[float, float, int]:
    # Assesses the relevance, coverage and word overlap of generated text
    rouge = Rouge()
    scores = rouge.get_scores(generated_answer, correct_answer)[0]

    # Calculate coverage
    correct_words = set(re.findall(r'\b\w+\b', correct_answer.lower()))
    generated_words = set(re.findall(r'\b\w+\b', generated_answer.lower()))

    coverage = len(generated_words.intersection(correct_words)) / len(correct_words)

    # Determine if words are the same (ignoring case and punctuation)
    has_common_word = 1 if correct_words.intersection(generated_words) else 0

    return scores['rouge-1']['f'], coverage, has_common_word


def _evaluate_fitness(population: Population, results, question_set, num_evals) -> Population:

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{now}.csv"
    result_dir = population.result_dir
    filepath = os.path.join(result_dir, filename)

    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['unit_index', 'question', 'generated_answer', 'correct_answer', 'prompt', 'total_fitness',
                      'similarity', 'log_perplexity', 'diversity', 'complexity', 'relevance', 'coverage',
                      'has_common_word']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # Initialise fitness
        for unit in population.units:
            unit.fitness = 0
        # Evaluating
        for unit_index, fitness_results in enumerate(results):
            generated_answer = fitness_results
            print(f"evaluating: {unit_index}")
            correct_answer = question_set[unit_index % num_evals]['answer']  # 序号取余才是真正的问题标号
            question = question_set[unit_index % num_evals]['question']

            # similarity evaluation
            similarity = calculate_similarity(get_embedding(generated_answer), get_embedding(correct_answer))

            # Perplexity evaluation
            perplexity = calculate_perplexity(generated_answer)
            log_perplexity = np.log(perplexity + 0.1)

            diversity = calculate_ngram_diversity(generated_answer)

            # Complexity evaluation
            length, grammar_complexity, logic_steps = calculate_complexity(generated_answer)

            # Relevance and Coverage evaluation
            relevance, coverage, has_common_word = evaluate_relevance_coverage_and_overlap(generated_answer, correct_answer)

            # Output results
            weights = {
                'similarity_weight': 1,
                'log_perplexity_weight': -0.1,
                'diversity_weight': 0,
                'complexity_weight': 0,
                'relevance_weight': 0,
                'coverage_weight': 0,
                'has_common_word': 0
            }

            unit.fitness += (weights['similarity_weight'] * similarity
                             + weights['log_perplexity_weight'] * log_perplexity +
                             weights['diversity_weight'] * diversity +
                             weights['complexity_weight'] * (length + grammar_complexity + logic_steps) +
                             weights['relevance_weight'] * relevance +
                             weights['coverage_weight'] * coverage +
                             weights['has_common_word'] * has_common_word)

            # Save the results
            writer.writerow({
                'unit_index': unit_index,
                'question': question,
                'generated_answer': generated_answer,
                'correct_answer': correct_answer,
                'prompt': unit.P,
                'total_fitness': unit.fitness,
                'similarity': similarity,
                'log_perplexity': log_perplexity,
                'diversity': diversity,
                'complexity': length + grammar_complexity + logic_steps,
                'relevance': relevance,
                'coverage': coverage,
                'has_common_word': has_common_word

            })
        # save the best result on population
        elite_fitness = max(unit.fitness for unit in population.units)
        population.elites.extend([unit for unit in population.units if unit.fitness == elite_fitness])

    return population

