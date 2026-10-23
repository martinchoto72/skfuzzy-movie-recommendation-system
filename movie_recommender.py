"""Fuzzy movie recommendation system built with scikit-fuzzy."""

from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


@dataclass(frozen=True)
class Movie:
    title: str
    rating: float        # 0-10
    genre_match: float   # 0-100 (съвпадение с любимите жанрове)


def create_variables():
    """Създава входните и изходната променлива с функциите на принадлежност."""
    rating = ctrl.Antecedent(np.arange(0, 10.1, 0.1), "rating")
    genre = ctrl.Antecedent(np.arange(0, 101, 1), "genre_match")
    recommend = ctrl.Consequent(np.arange(0, 101, 1), "recommendation")

    rating["low"] = fuzz.trapmf(rating.universe, [0, 0, 4, 6])
    rating["medium"] = fuzz.trimf(rating.universe, [4, 6, 8])
    rating["high"] = fuzz.trapmf(rating.universe, [6, 8, 10, 10])

    genre["low"] = fuzz.trapmf(genre.universe, [0, 0, 30, 50])
    genre["medium"] = fuzz.trimf(genre.universe, [30, 50, 70])
    genre["high"] = fuzz.trapmf(genre.universe, [50, 70, 100, 100])

    recommend["no"] = fuzz.trapmf(recommend.universe, [0, 0, 20, 40])
    recommend["maybe"] = fuzz.trimf(recommend.universe, [30, 50, 70])
    recommend["yes"] = fuzz.trapmf(recommend.universe, [60, 80, 100, 100])

    return rating, genre, recommend


def create_system(rating, genre, recommend):
    """Създава базата от правила."""
    rules = [
        ctrl.Rule(rating["high"] & genre["high"], recommend["yes"]),
        ctrl.Rule(rating["high"] & genre["medium"], recommend["yes"]),
        ctrl.Rule(rating["medium"] & genre["high"], recommend["yes"]),
        ctrl.Rule(rating["medium"] & genre["medium"], recommend["maybe"]),
        ctrl.Rule(rating["high"] & genre["low"], recommend["maybe"]),
        ctrl.Rule(rating["medium"] & genre["low"], recommend["no"]),
        ctrl.Rule(rating["low"], recommend["no"]),
    ]
    return ctrl.ControlSystem(rules)


class MovieRecommender:
    """Изчислява степен на препоръка (0-100) за даден филм."""

    def __init__(self) -> None:
        self.rating, self.genre, self.recommend = create_variables()
        self.system = create_system(self.rating, self.genre, self.recommend)
        self.simulation = ctrl.ControlSystemSimulation(self.system)

    def score(self, rating: float, genre_match: float) -> float:
        if not 0 <= rating <= 10:
            raise ValueError(f"rating must be in [0, 10], got {rating}")
        if not 0 <= genre_match <= 100:
            raise ValueError(f"genre_match must be in [0, 100], got {genre_match}")

        self.simulation.input["rating"] = rating
        self.simulation.input["genre_match"] = genre_match
        self.simulation.compute()
        return float(self.simulation.output["recommendation"])

    def rank(self, movies: list[Movie]) -> list[tuple[Movie, float]]:
        scored = [(m, self.score(m.rating, m.genre_match)) for m in movies]
        return sorted(scored, key=lambda pair: pair[1], reverse=True)