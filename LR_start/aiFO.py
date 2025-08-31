import pandas as pd
import numpy as np
import seaborn as sns  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42) 

num_samples = 500  

# данные
years_of_experience = np.random.randint(2, 21, size=num_samples)
intercept = 60_000 
slope = (200_000 - intercept) / 18
salaries = slope * years_of_experience + intercept + np.random.normal(0, 10_000, num_samples)

# создаем DataFrame
data = {'Years_of_experience': years_of_experience, 'Salary': salaries}
df = pd.DataFrame(data)

# график
plt.figure(figsize=(10,6))
sns.scatterplot(x='Years_of_experience', y='Salary', data=df, label="Data with noise")
sns.regplot(x='Years_of_experience', y='Salary', data=df, scatter=False, color="red", label="Regression line")

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Salary vs Experience')
plt.legend()
plt.show()

# модель
X = df[['Years_of_experience']]   # исправлено
Y = df[['Salary']]                # исправлено

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13)

lr = LinearRegression()
lr.fit(X_train, Y_train)

print("R2 train:", lr.score(X_train, Y_train))

# прогноз
Y_pred = lr.predict(X_test) 

# метрики
print("MAE:", mean_absolute_error(Y_test, Y_pred))
print("MSE:", mean_squared_error(Y_test, Y_pred))
print("R2 test:", r2_score(Y_test, Y_pred))
