import pandas as pd

url_train = 'https://raw.githubusercontent.com/AnnaReyc/Kaggle_Titanic/master/train.csv'
dataset = pd.read_csv(url_train)
#dataset.info() #видим, что типов object несколько: для обучения не нужны Name
dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 1
dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 2 #преобразуем для поля Sex 1-male 2-female
dataset.loc[dataset['Embarked'] == 'S', 'Embarked'] = 1
dataset.loc[dataset['Embarked'] == 'C', 'Embarked'] = 2
dataset.loc[dataset['Embarked'] == 'Q', 'Embarked'] = 3
#dataset.fillna(0) #заменить все значения integer NaN на 0
#print(dataset[dataset['Age'].isnull()]) #ищем значения float NaN
dataset['Embarked'] = pd.to_numeric(dataset['Embarked'], errors='coerce')
dataset = dataset.dropna(subset=['Embarked'])
dataset['Age'] = pd.to_numeric(dataset['Age'], errors='coerce')
dataset = dataset.dropna(subset=['Age'])
#dataset['Embarked'].astype(int)
dataset = dataset.drop(columns=['Cabin', 'Name'])#удаляем столбцы с object (axis=1 for columns, axis=0 for rows

print(dataset)
