import pydantic
from typing import List

class EvolutionUnit(pydantic.BaseModel):
    T: str
    M: str
    P: str
    fitness: float
    history: List[str]

class Population(pydantic.BaseModel):
    size: int
    problem_description: str
    elites: List[EvolutionUnit]
    units: List[EvolutionUnit]
    result_dir: str