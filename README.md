# SARS-COV-2  (COVID-19) Analysis & Prediction

##### Author: Dionisis Mastavrslis
     - Email:    dionisis.mastavralis@gmail.com
     - GitHub:   https://github.com/mastavralis/dm-covid-19-greece
     - License:  About the Analysis and Prediction Model: All Rights Reserved.

###### About the DATA
- Data Authority: JOHNS HOPKINS UNIVERSITY AND MEDICINE
- Data Source: https://github.com/CSSEGISandData/COVID-19
- Data renewal frequency: Daily at 23:59


+ Data Prediction Methods
    - Linear Regression
    - Polynomial Rregression

###### Notes:
* The analysis is customised to read, analyze and predict data only for Greece. However, changing the "country" parameter on In [139] the Model can fetch and analyze data for other countries as well, but with a probability of wrong data visualization or a malfunction of the Model. Thus, it is not recommended by the author to perform analysis for other countries but Greece.

## Tools & Libraries
+ Required Tools
    - Python >= 3
    - Jupyter Notebook / Anacond
    - matplotlib > 0.24

+ Required/Optional Libraries
    - datetime (R)
    - pandas   (R)
    - numpy    (R)
    - matplotlib.pyplot (R)
    - seaborn (O)
    - glob    (O)
    - __ future __ | absolute_import, division, print_function, unicode_literals (O)
    - pandas.plotting | register_matplotlib_converters (R)
    - sklearn.linear_model | LinearRegression    (R)
    - sklearn.preprocessing | PolynomialFeatures (R)
    
## How to run

Open an anaconda CMD and run: ***jupyter notebook***
A locall link will be created in order to navigate to the Jupyter project through a web browser.
