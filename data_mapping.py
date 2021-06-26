import pandas
from sklearn import tree

data_train = pandas.read_csv('dl_train.csv')
data_ans = pandas.read_csv('dl_status.csv')

auto_tree = tree.DecisionTreeClassifier()
tree.DecisionTreeClassifier.fit(
    auto_tree, 
    data_train.drop(columns='LSOA'), 
    data_ans.drop(columns='LSOA')
    )

auto_club = auto_tree.predict(
    data_train.drop(columns='LSOA'), 
    data_ans.drop(columns='LSOA')
    )

club_predictions = auto_club(data_train)

def test_predictions(predictions, answers):
    correct = 0
    for pred, ans in zip(predictions, answers):
        if pred == ans:
            correct +=1
    print('%s/%s correct!'%(correct, len(answers)))

test_predictions(club_predictions, data_ans)