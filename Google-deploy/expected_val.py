def calc_expected_val(team: str, odds: int, true_prob: float) -> dict:
    if not isinstance(team, str):
        raise TypeError("team must be a string")
    if not isinstance(odds, int):
        raise TypeError("odds must be an integer")
    if not isinstance(true_prob, float):
        raise TypeError("true_prob must be a float")

    implied_prob = 0
    decimal_odds = 0

    if odds < 0:  # 'minus' odds
        implied_prob = -(odds) / ((-(odds)) + 100)
        decimal_odds = (100 / -(odds)) + 1
    else:
        implied_prob = 100 / (odds + 100)
        decimal_odds = (odds / 100) + 1

    edge = (true_prob * decimal_odds) - 1

    return {
        team: {
            "prob_winning": round(
                true_prob, 4
            ),  # Rounded to 4 decimal places for clarity
            "edge": round(edge, 4),
            "odds": odds,
        }
    }


# TODO: uncomment for example
# print(calc_expected_val("DET", -120, 0.60))
