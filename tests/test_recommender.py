import pytest

from movie_recommender import Movie, MovieRecommender


@pytest.fixture(scope="module")
def recommender() -> MovieRecommender:
    return MovieRecommender()


def test_high_rating_and_match_is_recommended(recommender):
    assert recommender.score(9, 90) > 70


def test_low_rating_is_not_recommended(recommender):
    assert recommender.score(2, 90) < 40


def test_ranking_puts_best_movie_first(recommender):
    movies = [Movie("A", 5, 40), Movie("B", 9, 90)]
    assert recommender.rank(movies)[0][0].title == "B"


@pytest.mark.parametrize("rating, match", [(11, 50), (-1, 50), (5, 101)])
def test_invalid_input_raises(recommender, rating, match):
    with pytest.raises(ValueError):
        recommender.score(rating, match)