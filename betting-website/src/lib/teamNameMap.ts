export const teamNameMap: Record<string, string> = {
  "Nets": "Brooklyn Nets",
  "Celtics": "Boston Celtics",
  "Pistons": "Detroit Pistons",
  "Bulls": "Chicago Bulls",
  "Thunder": "Oklahoma City Thunder",
  "Rockets": "Houston Rockets",
  "Pacers": "Indiana Pacers",
  "Grizzlies": "Memphis Grizzlies",
  "Bucks": "Milwaukee Bucks",
  "Pelicans": "New Orleans Pelicans",
  "Spurs": "San Antonio Spurs",
  "Heat": "Miami Heat",
  "Cavaliers": "Cleveland Cavaliers",
  "Mavericks": "Dallas Mavericks",
  "Knicks": "New York Knicks",
  "Hawks": "Atlanta Hawks",
  "Trail Blazers": "Portland Trail Blazers",
  "Lakers": "Los Angeles Lakers",
  "Magic": "Orlando Magic",
  "Suns": "Phoenix Suns",
  "76ers": "Philadelphia 76ers",
  "Raptors": "Toronto Raptors",
  "Clippers": "Los Angeles Clippers",
  "Hornets": "Charlotte Hornets",
  "Wizards": "Washington Wizards",
  "Warriors": "Golden State Warriors",
  "Kings": "Sacramento Kings",
  "Jazz": "Utah Jazz",
  "Nuggets": "Denver Nuggets",
  "Timberwolves": "Minnesota Timberwolves",
};

export const getTeamLogo = (team: string) => {
  const fullTeamName = teamNameMap[team] || team; 
  return `/logos/${encodeURIComponent(fullTeamName)}.svg`;
};
