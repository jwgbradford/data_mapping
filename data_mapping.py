import pandas
from sklearn import tree

dl_training = pandas.read_csv(
    'dl_train.csv',
    sep=',',
    index_col=0,
    engine='c'
    )

dl_status = pandas.read_csv(
    'dl_status.csv',
    sep=',',
    index_col=0,
    engine='c'
    )

dl_uk = pandas.read_csv(
    'dl_uk.csv',
    sep=',',
    index_col=0,
    engine='c'
    )

auto_tree = tree.DecisionTreeClassifier()
auto_tree = auto_tree.fit(
    dl_training, 
    dl_status
    )

auto_club = auto_tree.predict(dl_uk)

pos_clubs = dl_uk

pos_clubs['pos_clubs'] = auto_club
pos_clubs.to_csv('pos_clubs.csv')