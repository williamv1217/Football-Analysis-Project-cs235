import numpy as np
import pandas as pd

events_dataset = pd.read_csv('./football-events/events_cleaned.csv')
events_dataset.info()
game_info_dataset = pd.read_csv('./football-events/game_info_cleaned.csv')
game_info_dataset.info()

