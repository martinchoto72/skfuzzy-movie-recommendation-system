"""Generates PNG figures for the README.

Outputs (in docs/images/):
- membership_rating.png
- membership_genre_match.png
- membership_recommendation.png
- control_surface.png
"""

import os

import matplotlib

matplotlib.use("Agg")  # работи без прозорци (headless)
import matplotlib.pyplot as plt
import numpy as np

from movie_recommender import MovieRecommender

OUTPUT_DIR = os.path.join("docs", "images")


def save_figure(fig, filename: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_membership_plots(recommender: MovieRecommender) -> None:
    variables = [
        (recommender.rating, "membership_rating.png"),
        (recommender.genre, "membership_genre_match.png"),
        (recommender.recommend, "membership_recommendation.png"),
    ]
    for variable, filename in variables:
        variable.view()
        save_figure(plt.gcf(), filename)


def save_control_surface(recommender: MovieRecommender, steps: int = 40) -> None:
    ratings = np.linspace(0, 10, steps)
    genres = np.linspace(0, 100, steps)

    surface = np.zeros((steps, steps))
    for i, genre in enumerate(genres):
        for j, rating in enumerate(ratings):
            surface[i, j] = recommender.score(rating, genre)

    R, G = np.meshgrid(ratings, genres)
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(R, G, surface, levels=20, cmap="viridis")
    fig.colorbar(contour, ax=ax, label="Степен на препоръка")
    ax.set_xlabel("Рейтинг на филма (0–10)")
    ax.set_ylabel("Съвпадение с жанровете (%)")
    ax.set_title("Повърхност на управлението")
    save_figure(fig, "control_surface.png")


def main() -> None:
    recommender = MovieRecommender()
    save_membership_plots(recommender)
    save_control_surface(recommender)


if __name__ == "__main__":
    main()