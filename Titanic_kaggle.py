import pandas as pd

url_train = 'https://raw.githubusercontent.com/AnnaReyc/Kaggle_Titanic/master/train.csv'
dataset = pd.read_csv(url_train)
#dataset.info() #видим, что типов object несколько: для обучения не нужны Name
dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 1
dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 2 #преобразуем для поля Sex 1-male 2-female
dataset.loc[dataset['Embarked'] == 'S'] = 1
dataset.loc[dataset['Embarked'] == 'C'] = 2
dataset.loc[dataset['Embarked'] == 'Q'] = 3
#dataset.fillna(0) #заменить все значения integer NaN на 0
#print(dataset[dataset['Embarked'].isnull()]) #ищем значения float NaN
dataset['Embarked'] = pd.to_numeric(dataset['Embarked'], errors='coerce')
dataset = dataset.dropna(subset=['Embarked'])
dataset['Embarked'].astype(int)
dataset.info()
