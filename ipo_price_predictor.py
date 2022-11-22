import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

def predict_ret(vars, ret_type, params):
    df = pd.read_csv('final_data_test.csv')

    # set up variables
    X = df[vars]
    y = df[ret_type]

    # initialize linear regression model
    LR = linear_model.LinearRegression()
    LR.fit(X, y)

    # predict returns based on ret_type
    return LR.predict([params])

if __name__ == "__main__":
    vars = ["RISK FACTORS Complexity", 
            "RISK FACTORS Positive Sentiment", "RISK FACTORS Neutral Sentiment", "RISK FACTORS Negative Sentiment",
            "RISK FACTORS Subjectivity",
            "Happy", "Angry", "Surprise", "Sad", "Fear",
            "S-1 Size", "Photos Frequencies", "non-gaap Frequency", "adjusted Frequency", "unknown Frequency", "uncertain Frequency", "uncommon Frequency"]

    ret_type = '1-Week Returns'
    params = [-7.7, 0.1, 0.8, 0.4, 0.4, 0.09, 0.02, 0.13, 0.11, 0.65, 100000, 4, 4, 5, 5, 4, 4]

    print(predict_ret(vars, ret_type, params))
