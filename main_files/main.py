from main_files.predictions import prediction
from main_files.analysis import analysis
# import plotly as py
# py.offline.init_notebook_mode(connected=True)


if __name__ == '__main__':
    print('--------------------------------------------------------------')
    print('-----------------------ANALYSIS SECTION-----------------------')
    print('--------------------------------------------------------------')
    analysis()
    print('--------------------------------------------------------------')
    print('-----------------------PREDICTION SECTION---------------------')
    print('--------------------------------------------------------------')
    prediction()
