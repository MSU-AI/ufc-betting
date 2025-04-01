def convert_team_name(full_name):
    """Convert full team name to team acronym."""
    
    # Dictionary mapping various team name formats to their acronyms
    team_name_mapping = {
        # Full names to acronyms
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
        
        # Common variations
        "Nets": "BKN",
        "Hornets": "CHA",
        "Warriors": "GSW",
        "Clippers": "LAC",
        "Lakers": "LAL",
        "Pelicans": "NOP",
        "Knicks": "NYK",
        "Thunder": "OKC",
        "76ers": "PHI",
        "Sixers": "PHI",
        "Suns": "PHX",
        "Trail Blazers": "POR",
        "Blazers": "POR",
        "Spurs": "SAS",
        
        # City-only variations
        "Atlanta": "ATL",
        "Boston": "BOS",
        "Brooklyn": "BKN",
        "Charlotte": "CHA",
        "Chicago": "CHI",
        "Cleveland": "CLE",
        "Dallas": "DAL",
        "Denver": "DEN",
        "Detroit": "DET",
        "Golden State": "GSW",
        "Houston": "HOU",
        "Indiana": "IND",
        "LA Lakers": "LAL",
        "LA Clippers": "LAC",
        "Memphis": "MEM",
        "Miami": "MIA",
        "Milwaukee": "MIL",
        "Minnesota": "MIN",
        "New Orleans": "NOP",
        "New York": "NYK",
        "Oklahoma City": "OKC",
        "Orlando": "ORL",
        "Philadelphia": "PHI",
        "Phoenix": "PHX",
        "Portland": "POR",
        "Sacramento": "SAC",
        "San Antonio": "SAS",
        "Toronto": "TOR",
        "Utah": "UTA",
        "Washington": "WAS",
    }
    
    try:
        # First try direct lookup
        if full_name in team_name_mapping:
            return team_name_mapping[full_name]
        
        # Try case-insensitive lookup
        full_name_lower = full_name.lower()
        return next(v for k, v in team_name_mapping.items() 
                   if k.lower() == full_name_lower)
            
    except (KeyError, StopIteration):
        raise ValueError(f"Could not convert team name: {full_name}")

# Example usage
if __name__ == "__main__":
    # Test some conversions
    test_names = [
        "Boston Celtics",
        "Golden State Warriors",
        "Warriors",
        "LA Lakers",
        "76ers"
    ]
    
    for name in test_names:
        try:
            acronym = convert_team_name(name)
            print(f"{name} -> {acronym}")
        except ValueError as e:
            print(f"Error: {e}")