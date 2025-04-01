import pandas as pd
import numpy as np
from datetime import datetime
from team_name_converter import convert_team_name
import os

def process_odds_data(file_path):
    odds_df = pd.read_csv(file_path)
    odds_df = odds_df[odds_df['date'] >= '2019-10-01']
    
    odds_df['team'] = odds_df['team'].apply(convert_team_name)
    odds_df['opponent'] = odds_df['opponent'].apply(convert_team_name)
    
    #filter rows where home/visitor is vs
    odds_df = odds_df[odds_df['home/visitor'] == 'vs']
    
    return odds_df

#get current dir
current_dir = os.path.dirname(os.path.abspath(__file__))


processed_odds_df = process_odds_data(os.path.join(current_dir, 'oddsData.csv'))
processed_odds_df.to_csv(os.path.join(current_dir, 'processed_odds_data.csv'), index=False)




