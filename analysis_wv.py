import pandas as pd
import plotly as py
import plotly.graph_objs as go


# py.offline.plot({
#     "data": [go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
#     "layout": go.Layout(title="hello no world")
# }, auto_open=False)



data = pd.read_csv('./football_events/merged.csv')

data.info()