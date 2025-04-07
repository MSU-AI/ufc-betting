import sqlite3
import csv
import os

conn = sqlite3.connect('nba.sqlite')

#create a cursor
cur = conn.cursor()

# Add this after creating the cursor
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print("Available tables:", tables)


print("Player table schema:")
# Get schema of the Player table
cur.execute("PRAGMA table_info(Player)")
schema = cur.fetchall()

for column in schema:
    print(f"Column: {column[1]}, Type: {column[2]}")


print("Game table schema:")
# Get schema of the Game table
cur.execute("PRAGMA table_info(Game)")
schema = cur.fetchall()

# Print the schema information
for column in schema:
    print(f"Column: {column[1]}, Type: {column[2]}")

# Split the queries and execute them separately

# Drop the TeamGames view if it exists
cur.execute("DROP VIEW IF EXISTS TeamGames;")

create_view_query = """
CREATE VIEW TeamGames AS
SELECT 
    game_id,
    season_id,
    game_date as date,
    team_id_home AS team,
    'home' AS location,
    CASE WHEN wl_home = 'W' THEN 1 ELSE 0 END as win,
    pts_home AS pts,
    reb_home AS reb,
    ast_home AS ast,
    stl_home AS stl,
    blk_home AS blk,
    tov_home AS tov,
    fgm_home AS fgm,
    fga_home AS fga,
    fg_pct_home AS fg_pct,
    fg3m_home AS fg3m,
    fg3a_home AS fg3a,
    fg3_pct_home AS fg3_pct,
    ftm_home AS ftm,
    fta_home AS fta,
    ft_pct_home AS ft_pct,
    oreb_home AS oreb,
    dreb_home AS dreb
FROM game
UNION ALL
SELECT 
    game_id,
    season_id,
    game_date as date,
    team_id_away AS team,
    'away' AS location,
    CASE WHEN wl_away = 'W' THEN 1 ELSE 0 END as win,
    pts_away AS pts,
    reb_away AS reb,
    ast_away AS ast,
    stl_away AS stl,
    blk_away AS blk,
    tov_away AS tov,
    fgm_away AS fgm,
    fga_away AS fga,
    fg_pct_away AS fg_pct,
    fg3m_away AS fg3m,
    fg3a_away AS fg3a,
    fg3_pct_away AS fg3_pct,
    ftm_away AS ftm,
    fta_away AS fta,
    ft_pct_away AS ft_pct,
    oreb_away AS oreb,
    dreb_away AS dreb
FROM game;
"""

# Create queries for 3, 5, and 10 game averages
create_3game_query = """
WITH SeasonDates AS (
    SELECT 
        season_id,
        MIN(game_date) as season_start,
        MAX(game_date) as season_end
    FROM game 
    WHERE game_date >= '2015-10-27'
    GROUP BY season_id
),
TeamAverages AS (
    SELECT 
        game_id,
        team,
        season_id,
        AVG(CAST(win as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS win_pct,
        AVG(pts) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg_pts,
        AVG(reb) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg_reb,
        AVG(ast) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg_ast,
        AVG(stl) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg_stl,
        AVG(blk) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg_blk,
        AVG(CAST(fg_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg_fg_pct,
        AVG(CAST(fg3_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg_fg3_pct,
        AVG(CAST(ft_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS avg_ft_pct
    FROM TeamGames
    WHERE date >= '2015-10-27' AND (date < '2020-03-11' OR date > '2020-10-11')
)
SELECT 
    g.game_id,
    g.game_date as date,
    g.season_id as season,
    g.team_id_home,
    g.team_abbreviation_home,
    g.team_id_away,
    g.team_abbreviation_away,
    g.wl_home,
    ha.win_pct AS home_win_pct,
    ha.avg_pts AS home_avg_pts,
    ha.avg_reb AS home_avg_reb,
    ha.avg_ast AS home_avg_ast,
    ha.avg_stl AS home_avg_stl,
    ha.avg_blk AS home_avg_blk,
    ha.avg_fg_pct AS home_avg_fg_pct,
    ha.avg_fg3_pct AS home_avg_fg3_pct,
    ha.avg_ft_pct AS home_avg_ft_pct,
    aa.win_pct AS away_win_pct,
    aa.avg_pts AS away_avg_pts,
    aa.avg_reb AS away_avg_reb,
    aa.avg_ast AS away_avg_ast,
    aa.avg_stl AS away_avg_stl,
    aa.avg_blk AS away_avg_blk,
    aa.avg_fg_pct AS away_avg_fg_pct,
    aa.avg_fg3_pct AS away_avg_fg3_pct,
    aa.avg_ft_pct AS away_avg_ft_pct
FROM game as g
JOIN SeasonDates sd ON g.season_id = sd.season_id
LEFT JOIN TeamAverages ha 
    ON g.game_id = ha.game_id AND ha.team = g.team_id_home
LEFT JOIN TeamAverages aa 
    ON g.game_id = aa.game_id AND aa.team = g.team_id_away
WHERE g.game_date >= sd.season_start
    AND g.game_date <= sd.season_end
    AND g.game_date >= '2015-10-27' 
    AND (g.game_date < '2020-03-11' OR g.game_date > '2020-10-11')
ORDER BY g.game_date;
"""

create_5game_query = """
WITH SeasonDates AS (
    SELECT 
        season_id,
        MIN(game_date) as season_start,
        MAX(game_date) as season_end
    FROM game 
    WHERE game_date >= '2015-10-27'
    GROUP BY season_id
),
TeamAverages AS (
    SELECT 
        game_id,
        team,
        season_id,
        AVG(CAST(win as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS win_pct,
        AVG(pts) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_pts,
        AVG(reb) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_reb,
        AVG(ast) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_ast,
        AVG(stl) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_stl,
        AVG(blk) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_blk,
        AVG(CAST(fg_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_fg_pct,
        AVG(CAST(fg3_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_fg3_pct,
        AVG(CAST(ft_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_ft_pct
    FROM TeamGames
    WHERE date >= '2015-10-27' AND (date < '2020-03-11' OR date > '2020-10-11')
)
SELECT 
    g.game_id,
    g.game_date as date,
    g.season_id as season,
    g.team_id_home,
    g.team_abbreviation_home,
    g.team_id_away,
    g.team_abbreviation_away,
    g.wl_home,
    ha.win_pct AS home_win_pct,
    ha.avg_pts AS home_avg_pts,
    ha.avg_reb AS home_avg_reb,
    ha.avg_ast AS home_avg_ast,
    ha.avg_stl AS home_avg_stl,
    ha.avg_blk AS home_avg_blk,
    ha.avg_fg_pct AS home_avg_fg_pct,
    ha.avg_fg3_pct AS home_avg_fg3_pct,
    ha.avg_ft_pct AS home_avg_ft_pct,
    aa.win_pct AS away_win_pct,
    aa.avg_pts AS away_avg_pts,
    aa.avg_reb AS away_avg_reb,
    aa.avg_ast AS away_avg_ast,
    aa.avg_stl AS away_avg_stl,
    aa.avg_blk AS away_avg_blk,
    aa.avg_fg_pct AS away_avg_fg_pct,
    aa.avg_fg3_pct AS away_avg_fg3_pct,
    aa.avg_ft_pct AS away_avg_ft_pct
FROM game as g
JOIN SeasonDates sd ON g.season_id = sd.season_id
LEFT JOIN TeamAverages ha 
    ON g.game_id = ha.game_id AND ha.team = g.team_id_home
LEFT JOIN TeamAverages aa 
    ON g.game_id = aa.game_id AND aa.team = g.team_id_away
WHERE g.game_date >= sd.season_start
    AND g.game_date <= sd.season_end
    AND g.game_date >= '2015-10-27' 
    AND (g.game_date < '2020-03-11' OR g.game_date > '2020-10-11')
ORDER BY g.game_date;
"""

create_10game_query = """
WITH SeasonDates AS (
    SELECT 
        season_id,
        MIN(game_date) as season_start,
        MAX(game_date) as season_end
    FROM game 
    WHERE game_date >= '2015-10-27'
    GROUP BY season_id
),
TeamAverages AS (
    SELECT 
        game_id,
        team,
        season_id,
        AVG(CAST(win as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS win_pct,
        AVG(pts) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_pts,
        AVG(reb) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_reb,
        AVG(ast) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_ast,
        AVG(stl) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_stl,
        AVG(blk) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_blk,
        AVG(CAST(fg_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_fg_pct,
        AVG(CAST(fg3_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_fg3_pct,
        AVG(CAST(ft_pct as FLOAT)) OVER (PARTITION BY team, season_id ORDER BY date ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS avg_ft_pct
    FROM TeamGames
    WHERE date >= '2015-10-27' AND (date < '2020-03-11' OR date > '2020-10-11')
)
SELECT 
    g.game_id,
    g.game_date as date,
    g.season_id as season,
    g.team_id_home,
    g.team_abbreviation_home,
    g.team_id_away,
    g.team_abbreviation_away,
    g.wl_home,
    ha.win_pct AS home_win_pct,
    ha.avg_pts AS home_avg_pts,
    ha.avg_reb AS home_avg_reb,
    ha.avg_ast AS home_avg_ast,
    ha.avg_stl AS home_avg_stl,
    ha.avg_blk AS home_avg_blk,
    ha.avg_fg_pct AS home_avg_fg_pct,
    ha.avg_fg3_pct AS home_avg_fg3_pct,
    ha.avg_ft_pct AS home_avg_ft_pct,
    aa.win_pct AS away_win_pct,
    aa.avg_pts AS away_avg_pts,
    aa.avg_reb AS away_avg_reb,
    aa.avg_ast AS away_avg_ast,
    aa.avg_stl AS away_avg_stl,
    aa.avg_blk AS away_avg_blk,
    aa.avg_fg_pct AS away_avg_fg_pct,
    aa.avg_fg3_pct AS away_avg_fg3_pct,
    aa.avg_ft_pct AS away_avg_ft_pct
FROM game as g
JOIN SeasonDates sd ON g.season_id = sd.season_id
LEFT JOIN TeamAverages ha 
    ON g.game_id = ha.game_id AND ha.team = g.team_id_home
LEFT JOIN TeamAverages aa 
    ON g.game_id = aa.game_id AND aa.team = g.team_id_away
WHERE g.game_date >= sd.season_start
    AND g.game_date <= sd.season_end
    AND g.game_date >= '2015-10-27' 
    AND (g.game_date < '2020-03-11' OR g.game_date > '2020-10-11')
ORDER BY g.game_date;
"""

create_season_query = """
WITH SeasonDates AS (
    SELECT 
        season_id,
        MIN(game_date) as season_start,
        MAX(game_date) as season_end
    FROM game 
    WHERE game_date >= '2015-10-27'
    GROUP BY season_id
),
TeamAverages AS (
    SELECT 
        game_id,
        team,
        season_id,
        AVG(CAST(win as FLOAT)) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS win_pct,
        AVG(pts) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_pts,
        AVG(reb) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_reb,
        AVG(ast) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_ast,
        AVG(stl) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_stl,
        AVG(blk) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_blk,
        AVG(CAST(fg_pct as FLOAT)) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_fg_pct,
        AVG(CAST(fg3_pct as FLOAT)) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_fg3_pct,
        AVG(CAST(ft_pct as FLOAT)) OVER (
            PARTITION BY team, season_id 
            ORDER BY date 
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS avg_ft_pct
    FROM TeamGames
    WHERE date >= '2015-10-27' AND (date < '2020-03-11' OR date > '2020-10-11')
)
SELECT 
    g.game_id,
    g.game_date as date,
    g.season_id as season,
    g.team_id_home,
    g.team_abbreviation_home,
    g.team_id_away,
    g.team_abbreviation_away,
    g.wl_home,
    ha.win_pct AS home_win_pct,
    ha.avg_pts AS home_avg_pts,
    ha.avg_reb AS home_avg_reb,
    ha.avg_ast AS home_avg_ast,
    ha.avg_stl AS home_avg_stl,
    ha.avg_blk AS home_avg_blk,
    ha.avg_fg_pct AS home_avg_fg_pct,
    ha.avg_fg3_pct AS home_avg_fg3_pct,
    ha.avg_ft_pct AS home_avg_ft_pct,
    aa.win_pct AS away_win_pct,
    aa.avg_pts AS away_avg_pts,
    aa.avg_reb AS away_avg_reb,
    aa.avg_ast AS away_avg_ast,
    aa.avg_stl AS away_avg_stl,
    aa.avg_blk AS away_avg_blk,
    aa.avg_fg_pct AS away_avg_fg_pct,
    aa.avg_fg3_pct AS away_avg_fg3_pct,
    aa.avg_ft_pct AS away_avg_ft_pct
FROM game as g
JOIN SeasonDates sd ON g.season_id = sd.season_id
LEFT JOIN TeamAverages ha 
    ON g.game_id = ha.game_id AND ha.team = g.team_id_home
LEFT JOIN TeamAverages aa 
    ON g.game_id = aa.game_id AND aa.team = g.team_id_away
WHERE g.game_date >= sd.season_start
    AND g.game_date <= sd.season_end
    AND g.game_date >= '2015-10-27' 
    AND (g.game_date < '2020-03-11' OR g.game_date > '2020-10-11')
ORDER BY g.game_date;
"""

# Execute the create view query first
cur.execute(create_view_query)

# os to get directory of this file and prepend to filemames

current_dir = os.path.dirname(os.path.abspath(__file__))


# Execute each query and save to separate CSV files
for query, filename in [
    (create_season_query, os.path.join(current_dir, 'team_game_stats_season.csv')),
    (create_3game_query, os.path.join(current_dir, 'team_game_stats_3game.csv')),
    (create_5game_query, os.path.join(current_dir, 'team_game_stats_5game.csv')),
    (create_10game_query, os.path.join(current_dir, 'team_game_stats_10game.csv'))
]:
    cur.execute(query)
    results = cur.fetchall()
    print(f"Number of rows for {filename}: {len(results)}")
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([i[0] for i in cur.description])  # Write header
        writer.writerows(results)

# Close the connection
conn.close()
