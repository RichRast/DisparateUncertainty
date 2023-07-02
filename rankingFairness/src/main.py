import numpy as np
import argparse

parser = argparse.ArgumentParser(description="ranking with simulated dataset with bernoulli probs for merit")
parser.add_argument(
    "--num_groups",
    type=int,
    default='2',
)
parser.add_argument(
    "--proportion of candidates in groups",
    type=int,
    default=5,
)
parser.add_argument(
    "--number of candidates in first group",
    type=int,
    default=5,
)
parser.add_argument(
    "--probs in each group",
    type=int,
    default=5,
)
parser.add_argument(
    "--top_k",
    type=int,
    default=5,
)
parser.add_argument(
    "--setting",
    type=str,
    default=None,
    choices=[
        'offline',
        'online'
    ]

)
parser.add_argument(
    "--ranker",
    type=str,
    default=None,
    choices=[
        'PRP',
        'Uniform',
        'EOR',
    ]
)