# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:50:11 2023

@author: Salman Saleem
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import scipy as sc

def load_data(file_name):
    """Load data from a CSV file and prepare it for analysis.
    
    Args:
    file_name (str): the name of the CSV file to load
    
    Returns:
    A tuple of three pandas DataFrames: 
    1. the original data as loaded from the CSV file
    2. a cleaned version of the data with missing values filled and duplicates removed
    3. a transposed version of the cleaned data with country names as column headers
    """
    # Load data from a CSV file, skip any bad lines.
    main_data = pd.read_csv(file_name, on_bad_lines='skip')
    
    # Transpose the data frame and set the country name as the columns.
    main_data_copy = main_data.loc[main_data['Country Name'].drop_duplicates(keep='last').index].transpose()
    main_data_copy.columns = main_data_copy.loc['Country Name']
    main_data_copy = main_data_copy[1:]
    
    return main_data, main_data.drop(main_data.columns[1:4], axis=1), main_data_copy

def cleaning(data):
    """Clean the given data by filling missing values with the column mean and removing duplicates.
    
    Args:
    data (DataFrame): the data to clean
    
    Returns:
    A cleaned version of the input data
    """
    # Fill missing values with the column mean.
    for col in data.columns[1:]:
        data[col] = data[col].fillna(data[col].mean())
    
    # Remove duplicates.
    data = data.drop_duplicates()
    
    return data


def get_country_years(data):
    """Select a list of countries and years to analyze.
    
    Args:
    data (DataFrame): the data to select from
    
    Returns:
    A tuple containing a list of years to analyze and a list of countries to select
    """
    # Get the years to analyze and a list of countries to select.
    Cus_years = list(data.columns[53:])
    Country = random.choices(data['Country Name'].unique(), k=13)
    
    return Cus_years, Country


def plot_bar(dataframe, x, title, kind='barh', stacked=False, fontsize=20, figsize=(30, 15)):
    """Plot a horizontal bar chart.
    
    Args:
    dataframe (DataFrame): the data to plot
    x (str): the column to plot on the x-axis
    title (str): the title of the plot
    kind (str): the type of plot to create (default 'barh')
    stacked (bool): whether or not to stack the bars (default False)
    fontsize (int): the font size to use for the plot (default 20)
    figsize (tuple): the size of the plot in inches (default (30, 15))
    """
    # Plot a horizontal bar chart.
    ax = dataframe.plot(x=x, title=title, kind=kind, stacked=stacked, fontsize=fontsize, figsize=figsize)
    ax.set_title(title, fontsize=30)
    
    
def R_e_c_analysis(data):
    """
    Selects renewable energy consumption data for the given countries and years from the input dataset.

    Args:
    data (pd.DataFrame): Input dataset containing renewable energy consumption data.

    Returns:
    pd.DataFrame: A modified dataframe containing renewable energy consumption values for each selected country.
    """
    # Select only the renewable energy consumption data.
    r_e_c_df = data[data['Indicator Name'] == 'Renewable energy consumption (% of total final energy consumption)']

    country_lis=[]
    for cont in Country:
        # Append the renewable energy consumption values for each selected country.
        country_lis.append(list(r_e_c_df[r_e_c_df['Country Name'] == cont][Cus_years].values[0]))

    modified_df = pd.DataFrame(country_lis)
    modified_df.insert(0, 'Country_Name', Country)
    modified_df.columns = [modified_df.columns[0]] + Cus_years

    return modified_df


def population_grow_anlysis(data):
    """
    Selects population growth data for the given countries and years from the input dataset.

    Args:
    data (pd.DataFrame): Input dataset containing population growth data.

    Returns:
    pd.DataFrame: A modified dataframe containing population growth values for each selected country.
    """
    # Select only the population growth data.
    pop_grow = data[data['Indicator Name'] == 'Population growth (annual %)']
    country_lis=[]
    for cont in Country:
        # Append the population growth values for each selected country.
        country_lis.append(list(pop_grow[pop_grow['Country Name'] == cont][Cus_years].values[0]))

    modified_df = pd.DataFrame(country_lis)
    modified_df.insert(0, 'Country_Name', Country)
    modified_df.columns = [modified_df.columns[0]] + col
    return modified_df

def stats_using_scipy(data1):
    """
    Calculates skewness, standard error of the mean, kurtosis, kurtosis statistic, and kurtosis p-value for the given data using scipy.

    Args:
    data1 (pd.DataFrame): Input dataset for which the statistics are to be calculated.

    Returns:
    pd.DataFrame: A dataframe containing the calculated statistics.
    """
    # Compute skewness of each column in the data, except for the first column (which contains the name of the country)
    skew_df = [sc.stats.skew(data1[i]) for i in data1.columns[1:]]
    # Compute standard error of the mean of each column in the data, except for the first column
    sem_df = [sc.stats.sem(data1[i]) for i in data1.columns[1:]]
    # Compute kurtosis of each column in the data, except for the first column
    kurtosis_df = [sc.stats.kurtosis(data1[i]) for i in data1.columns[1:]]

    # Compute the kurtosis statistic of each column in the data, except for the first column
    kurtosistestST_df = [sc.stats.kurtosistest(data1[i])[0] for i in data1.columns[1:]]
    # Compute the kurtosis p-value of each column in the data, except for the first column
    kurtosistestPVal_df = [sc.stats.kurtosistest(data1[i])[1] for i in data1.columns[1:]]
    
    # Create a pandas DataFrame from the computed statistics with the columns 'skew', 'standard_error_mean',
     # Create a pandas DataFrame from the computed statistics with the columns 'skew', 'standard_error_mean', 'kurtosis', 'kurtosis_statistic', 'kurtosis_pval'
    panda_statis_df = pd.DataFrame(index=data1.columns[1:],data=list(zip(skew_df,sem_df,kurtosis_df,kurtosistestST_df,kurtosistestPVal_df)),
             columns='skew,standard_error_mean,kurtosis,kurtosis_statistic,kurtosis_pval'.split(','))
    # Return the DataFrame
    return panda_statis_df


def stats_using_numpy(data1):
    """
    Computes the statistical properties (mean, median, standard deviation, variance, variance ignoring NaN values) 
    of each column (except the first one) of the input dataframe using NumPy functions.
    
    Parameters:
    data1 (pandas.DataFrame): The input dataframe.
    
    Returns:
    pandas.DataFrame: A dataframe containing the calculated statistics.
    """
    # Calculate mean for each column (except the first one) of the input dataframe
    mean_df = [np.mean(data1[i]) for i in data1.columns[1:]]
    # Calculate median for each column (except the first one) of the input dataframe
    median_df = [np.median(data1[i]) for i in data1.columns[1:]]
    # Calculate standard deviation for each column (except the first one) of the input dataframe
    std_df = [np.std(data1[i]) for i in data1.columns[1:]]
    # Calculate variance for each column (except the first one) of the input dataframe
    var_df = [np.var(data1[i]) for i in data1.columns[1:]]

    # Calculate variance ignoring NaN values for each column (except the first one) of the input dataframe
    varNan_df = [np.nanvar(data1[i]) for i in data1.columns[1:]]
    
    # Create a pandas dataframe to store the calculated statistics
    panda_statis_df_numpy = pd.DataFrame(index=data1.columns[1:],data=list(zip(mean_df,median_df,std_df,var_df,varNan_df)),
             columns='mean,median,stanard deviation,variance,nan_varinace'.split(','))
    # Return the pandas dataframe containing the calculated statistics
    return panda_statis_df_numpy


def analysis_urban_pop_year(data):
    """
    Filters the input dataframe to only include rows with Country Name in the list of countries of interest, 
    and filters the dataframe to only include rows with Indicator Name equal to 'Urban population (% of total population)' 
    and select columns for the three years of interest (1960, 2000, and 2021) and Country Name.
    
    Parameters:
    data (pandas.DataFrame): The input dataframe.
    
    Returns:
    pandas.DataFrame: The resulting dataframe.
    """
    # Filter the input dataframe to only include rows with Country Name in the list of countries of interest
    df_urban_pop = data[data['Country Name'].isin(Country)].drop_duplicates(keep='first')
    # Filter the dataframe to only include rows with Indicator Name equal to 'Urban population (% of total population)'
    # and select columns for the three years of interest (1960, 2000, and 2021) and Country Name
    df_urban_pop = df_urban_pop[df_urban_pop['Indicator Name'] == 'Urban population (% of total population)'][['Country Name','1960','2000','2021']].set_index('Country Name')
    # Return the resulting dataframe
    return df_urban_pop


def correlatation_analysi_indicators(data):
    """
    Filters the given data based on the custom indicator list and returns the correlation matrix for the transposed data.

    Parameters:
    data (pandas.DataFrame): Input data for correlation analysis.

    Returns:
    pandas.DataFrame: Correlation matrix for the filtered data.

    """
    # Filters the data based on the given custom indicator list
    df_corr_indicator = data[data['Indicator Name'].isin(custom_indicator_lis)]
    # Filters the data to get only the data for Finland, sets the index to the Indicator Name
    df_corr_indicator = df_corr_indicator[df_corr_indicator['Country Name'] == 'Finland'].set_index('Indicator Name')
    # Drops the unnecessary columns and transposes the data
    df_corr_indicator = df_corr_indicator.drop(['Country Name','Country Code','Indicator Code'],axis=1).transpose()
    # Returns the correlation matrix for the transposed data
    return df_corr_indicator.corr()

def plot_correlation_heatmap(datafram_corr,title,cmap='tab20'):
    """
    Plots a heatmap for the given correlation matrix.

    Parameters:
    datafram_corr (pandas.DataFrame): Correlation matrix for the analysis.
    title (str): Title of the plot.
    cmap (str): Color map for the plot.

    Returns:
    None
    """
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(datafram_corr, vmin=-1, vmax=1, annot=True, cmap=cmap)
    heatmap.set_title(title, fontdict={'fontsize':22}, pad=12)

def plot_line(data,title):
    """
    Plots a line chart for the given data.

    Parameters:
    data (pandas.DataFrame): Data for the analysis.
    title (str): Title of the plot.

    Returns:
    None
    """
    ax = data.plot(figsize=(20,10))
    ax.set_title(title)

def analysis_coutry_arableLand_wrt_years(data):
    """
    Filters the data based on the given indicator name and returns the transposed data for the given country names.

    Parameters:
    data (pandas.DataFrame): Input data for analysis.

    Returns:
    pandas.DataFrame: Transposed data for the given country names.
    """
    # Filters the data based on the given indicator name
    df_arable_country=data[data['Indicator Name']=='Arable land (% of land area)']
    # Drops the unnecessary columns except the Country Name column and transposes the data
    df_arable_country = df_arable_country.drop(['Indicator Name', 'Country Code','Indicator Code','1960'],axis=1)
    # Returns the transposed data for the given country names and fills any NaN values with 0
    return df_arable_country[df_arable_country['Country Name'].isin(Country)].set_index('Country Name').transpose().fillna(0)

# Define a list of custom indicators to be used in the analysis
custom_indicator_lis = ['Urban population (% of total population)',
'School enrollment, primary and secondary (gross), gender parity index (GPI)',
'Agriculture, forestry, and fishing, value added (% of GDP)',
'Forest area (% of land area)', 'Forest area (sq. km)',
'Electric power consumption (kWh per capita)',
'CO2 emissions from liquid fuel consumption (% of total)',
'Arable land (% of land area)']


# Load data from the CSV file
file = r'C:\Users\munee\Desktop\SSK assignment\Assignment 2\dataset\datasource.csv'
data0,data1,data2 = load_data(file)

# Clean data1
data1 = cleaning(data1)

# Print information about data1, including the columns and data types
print(data1.info())

# Print the first few rows of data1
print(data1.head())

# Print the first few rows of data2
print(data2.head())

# Print descriptive statistics of data0
print(data0.describe())

# Get a list of years and countries
Cus_years,Country = get_country_years(data0)
col = list(data0.columns[53:])

# Perform R_E_C analysis and plot the result in a bar chart
R_E_C_Analysis_Result_DF = R_e_c_analysis(data0) 
title = 'Renewable energy consumption (% of total final energy consumption) Per Year Country VS Country'
plot_bar(R_E_C_Analysis_Result_DF,'Country_Name',title) 

# Plot the result of R_E_C analysis in a horizontal bar chart
title = 'Renewable energy consumption (% of total final energy consumption) Per Year Country VS Country'
plot_bar(R_E_C_Analysis_Result_DF,'Country_Name',title,'bar')  

# Perform population growth analysis and plot the result in a bar chart
df_population_grow_anlysis = population_grow_anlysis(data0)
title = 'Population growth annually'
plot_bar(R_E_C_Analysis_Result_DF,'Country_Name',title,'bar') 

# Calculate descriptive statistics using scipy and print the result
panda_statis_df = stats_using_scipy(data1)
print(panda_statis_df.head())

# Calculate descriptive statistics using numpy and print the result
panda_statis_df_numpy = stats_using_numpy(data1)
print(panda_statis_df_numpy.head())

# Analyze urban population over the years and print the result
analysis_urban_pop_year_df = analysis_urban_pop_year(data0)
print(analysis_urban_pop_year_df)

# Perform correlation analysis of selected indicators and plot the correlation heatmap
correlatation_analysi_indicators_df = correlatation_analysi_indicators(data0)

# Plot a heatmap for correlation analysis results
plot_correlation_heatmap(correlatation_analysi_indicators_df,'Correlation Heatmap')

# Analyze arable land of selected countries over the years and plot the result in a line chart
analysis_coutry_arableLand_wrt_years_df = analysis_coutry_arableLand_wrt_years(data0)

#Plot a line graph for arable land analysis results
plot_line(analysis_coutry_arableLand_wrt_years_df,'Arable land wrt Years and country')

