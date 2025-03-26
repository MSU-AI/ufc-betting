
team_aliases = {
    "ATL": ["Atlanta Hawks", "Hawks"],
    "BOS": ["Boston Celtics", "Celtics"],
    "BRK": ["Brooklyn Nets", "Nets"],
    "CHO": ["Charlotte Hornets", "Hornets"],
    "CHI": ["Chicago Bulls", "Bulls"],
    "CLE": ["Cleveland Cavaliers", "Cavaliers", "Cavs"],
    "DAL": ["Dallas Mavericks", "Mavericks"],
    "DEN": ["Denver Nuggets", "Nuggets"],
    "DET": ["Detroit Pistons", "Pistons"],
    "GSW": ["Golden State Warriors", "GS Warriors", "Golden State", "Warriors"],
    "HOU": ["Houston Rockets", "Rockets"],
    "IND": ["Indiana Pacers", "Pacers"],
    "LAC": ["Los Angeles Clippers", "LA Clippers", "Clippers"],
    "LAL": ["Los Angeles Lakers", "LA Lakers", "Lakers"],
    "MEM": ["Memphis Grizzlies", "Grizzlies"],
    "MIA": ["Miami Heat", "Heat"],
    "MIL": ["Milwaukee Bucks", "Bucks"],
    "MIN": ["Minnesota Timberwolves", "Timberwolves"],
    "NOP": ["New Orleans Pelicans", "Pelicans"],
    "NYK": ["New York Knicks", "NY Knicks", "Knicks"],
    "OKC": ["Oklahoma City Thunder", "OKC Thunder", "Thunder"],
    "ORL": ["Orlando Magic", "Magic"],
    "PHI": ["Philadelphia 76ers", "Philly 76ers", "76ers", "Sixers"],
    "PHO": ["Phoenix Suns", "Suns"],
    "POR": ["Portland Trail Blazers", "Trail Blazers", "Blazers", "Portland Blazers"],
    "SAC": ["Sacramento Kings", "Sacramento"],
    "SAS": ["San Antonio Spurs", "San Antonio", "Spurs"],
    "TOR": ["Toronto Raptors", "Toronto", "Raptors"],
    "UTA": ["Utah Jazz", "Jazz"],
    "WAS": ["Washington Wizards", "Washington", "Wizards"],
}

# reverse lookup for aliases to acronyms, in case we need it
alias_to_acronym = {
    alias.lower(): code for code, names in team_aliases.items() for alias in names
}

acronym_to_canonical = {
    code: names[0] for code, names in team_aliases.items()
}

def convert_team_name(name):
    """Convert team name or acronym to acronym or full name."""
    if not isinstance(name, str):
        raise ValueError("Input must be a string")
    name = name.strip()
    upper = name.upper()
    if upper in acronym_to_canonical:
        return acronym_to_canonical[upper]
    lower = name.lower()
    if lower in alias_to_acronym:
        return alias_to_acronym[lower]
    raise ValueError(f"Could not convert team name or acronym: {name}")


if __name__ == "__main__":
    # bascically 3 letter acronym converts to full name
    # every otherthing converts to acronym
    test_names = [
        "76ers", "Philadelphia 76ers", "PHI",
        "Toronto", "Raptors", "TOR",
        "Golden State", "GSW", "Warriors",
        "Lakers", "LAL", "New York Knicks", "NYK",
        "Boston Celtics",
        "Golden State Warriors",
        "Warriors",
        "LA Lakers",
        "76ers",
        "Philadelphia 76ers",
        "Pacers",
        "Indiana Pacers",
        "NY Knicks",
        "Orlando Magic",
        "Spurs",
        "Toronto Raptors",
        "PHI",
        "TOR"
    ]

    for name in test_names:
        try:
            result = convert_team_name(name)
            print(f"{name} -> {result}")
        except ValueError as e:
            print(f"Error: {e}")
