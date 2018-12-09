import numpy as np
import pandas as pd

events_dataset = pd.read_csv('./football_events/events_cleaned.csv')
game_info_dataset = pd.read_csv('./football_events/game_info_cleaned.csv')

merged = events_dataset.merge(game_info_dataset, on= 'id_odsp')
pd.DataFrame.to_csv(merged, path_or_buf='./football_events/merged.csv', float_format='%.0f')

