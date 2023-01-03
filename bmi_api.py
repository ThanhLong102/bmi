from flask import Flask
from flask import request
from pycaret.regression import *
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
reg = ExtraTreesRegressor()


class Bmi:
    def __int__(self, sex, age, height, weight):
        self.sex = sex
        self.age = age
        self.height = height
        self.weight = weight


@app.route("/bmi-perdic", methods=['POST'])
def bmi_predict():
    init()
    # result = reg.predict(request.get_json())
    # data = [['sex', 'age', 'height', 'weight'],
    #         [request.get_json()['sex'], request.get_json()['age'], request.get_json()['height'],
    #          request.get_json()['weight']]]
    print(request)
    df2 = pd.DataFrame.from_dict([request.get_json()['bmi']])
    predict = reg.predict(df2)
    print(predict[0])
    return {'result': str(predict[0])}


def init():
    df = pd.read_csv("./bmi_data.csv")
    df.isnull().sum()
    df = df.dropna()
    df.reset_index(inplace=True, drop=True)

    # Apply label encoder to each column with categorical data
    label_encoder = LabelEncoder()
    t = (df.dtypes == "object")
    object_cols = list(t[t].index)

    for i in object_cols:
        df[i] = label_encoder.fit_transform(df[i])

    df["Height(Inches)"] = df["Height(Inches)"].apply(lambda hei: hei * 2.54)
    df["Weight(Pounds)"] = df["Weight(Pounds)"].apply(lambda wei: wei / 2.205)
    df.rename(columns={"Height(Inches)": "Height", "Weight(Pounds)": "Weight"}, inplace=True)
    X = df.drop("BMI", axis=1)
    Y = df["BMI"]
    reg.fit(X, Y)


if __name__ == "__main__":
    app.run()
