from movie_recommender import Movie, MovieRecommender

MOVIES = [
    Movie("Inception", 8.8, 90),
    Movie("Romantic Comedy X", 5.2, 85),
    Movie("Obscure Thriller", 7.5, 40),
    Movie("Classic Drama", 8.0, 70),
    Movie("Bad Action Film", 3.1, 60),
]


def main() -> None:
    recommender = MovieRecommender()
    print(f"{'Movie':<20}{'Rating':>8}{'Match':>8}{'Score':>8}")
    print("-" * 44)
    for movie, score in recommender.rank(MOVIES):
        print(f"{movie.title:<20}{movie.rating:>8.1f}{movie.genre_match:>8.0f}{score:>8.1f}")


if __name__ == "__main__":
    main()