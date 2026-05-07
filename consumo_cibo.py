import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_theme()
sns.set(rc={'figure.figsize':(15, 9)})
sns.set(font_scale=1.5)

#The food_consumption.csv dataset from the data folder contains data on the several countries' food consumption per food category and their respective CO2 emissions. Load it to a DataFrame named food and check its contents.
food = pd.read_csv('food_consumption.csv')
print(food.head())

#Use the .describe() method on the food DataFrame to produce descriptive statistics about the consumption metric for each class in the food_category variable. Which food_category has the highest median value of food_consumption?
print(food.groupby('food_category')['consumption'].describe())
highest_median = food.groupby('food_category')['consumption'].median().idxmax()
print(f"The food category with the highest median value of food consumption is: {highest_median}")

#In a single chart, plot one boxplot for each food_category (11 in total) using the variable co2_emission as the metric. By looking at the chart, which food_category has the highest interquartile range (IQR)?
plt.figure(figsize=(12, 6))
sns.boxplot(x='food_category', y='co2_emission', data=food)
plt.xticks(rotation=45)
plt.title('CO2 Emission by Food Category')
plt.xlabel('Food Category')
plt.ylabel('CO2 Emission')
plt.show()

#Looking at the chart from the previous question, which is the food_category with the highest median co2_emission value?
co2_emission_stats = food.groupby('food_category')['co2_emission'].describe()
print(co2_emission_stats)

co2_emission_stats['IQR'] = co2_emission_stats['75%'] - co2_emission_stats['25%']
highest_iqr_category = co2_emission_stats['IQR'].idxmax()
highest_median_co2_category = co2_emission_stats['50%'].idxmax()

print(f"\nThe food_category with the highest interquartile range (IQR) for CO2 emission is: {highest_iqr_category}")
print(f"The food_category with the highest median CO2 emission is: {highest_median_co2_category}")

#Consider the consumption of "poultry" and "fish" across all available countries; looking at the table from question 1, the average of poultry consumption (21.22) seems to be higher than that of fish consumption (17.29), but is this difference statistically significant? Create a permutation test in order to assess the null hypothesis that there is no difference between the two means. Do you accept or reject the null hypothesis? Explain why.
consumption_poultry = food[food['food_category'] == 'poultry']['consumption']
consumption_fish = food[food['food_category'] == 'fish']['consumption'] 
observed_diff = consumption_poultry.mean() - consumption_fish.mean()
combined = np.concatenate([consumption_poultry, consumption_fish])
n_permutations = 10000
count_extreme = 0

for i in range(n_permutations):
    np.random.shuffle(combined)
    permuted_poultry = combined[:len(consumption_poultry)]
    permuted_fish = combined[len(consumption_poultry):]
    permuted_diff = permuted_poultry.mean() - permuted_fish.mean()

    if abs(permuted_diff) >= abs(observed_diff):
        count_extreme += 1
p_value = count_extreme / n_permutations
print(f"Observed difference in means: {observed_diff}")
print(f"P-value from permutation test: {p_value}")

#The distributions.csv dataset from the data folder contains data drawn from 6 probability distributions – one per column. Load it to a DataFrame named distributions and check its contents.
distributions = pd.read_csv('distributions.csv')
print(distributions.head())

#Use the .describe() and info() methods on the distributions DataFrame to produce some preliminary information about each column. What is the data type of each column?
print(distributions.describe())
print(distributions.info())

#Loop through each column of distributions and for each column produce a plot to show the distribution.
distributions.hist(bins=30, figsize=(15, 10))
plt.suptitle('Distribution of Each Column', fontsize=16)
plt.show() 