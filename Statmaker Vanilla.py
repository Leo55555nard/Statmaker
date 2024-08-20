
#Imports and file path
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from tkinter import Canvas, NW


#Expanded colors
colors = ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3", "#FFC0CB", "#A52A2A", "#800080", "#00FFFF", "#FFD700", "#800000"]

dummylist=["Test, Jess", "Soap, Jo", "Doe, Jane","Seep, Jan","Sweet, Janet"]

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    if getattr(sys, 'frozen', False):  # If running as a bundle
        base_path = sys._MEIPASS
    else:  # If running in a normal Python environment
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

#Checking file type and missing columns
def Prerror(filepath,main=True):
    #filepath=str(filepath)
    #filepath2=str(filepath2)
    if not filepath.endswith(".xlsx"):
        return "File Error: Not an Excel file"
    
    try:
        df = pd.read_excel(filepath)
    except FileNotFoundError:
        return f"File Error: File '{filepath}' not found or unable to read"
    except Exception as e:
        return f"File Error: {str(e)}"
    #Columns needed in excel file 
    if main==True:
        needed = sorted(["Last name, First name", "Location", "Employment Status","Division","Department", "Gender","Degree","Major/Specialization",'Job Title'])
        df_cols = sorted(df.columns)
    
        missing_columns = [col for col in needed if col not in df_cols]
    
        if missing_columns:
            missing_message = f"Missing column(s): {', '.join(missing_columns)}. Necessary columns are: {', '.join(needed)}"
            return missing_message
    
    
def remove_dummies(names, Full_time_only, Except_interns, df):

    # Format handling
    if not isinstance(names, (str, list)):
        return "Insert dummy names as string or list of strings"
    if not isinstance(df, pd.DataFrame):
        return "Insert the file as type DataFrame"
    
    # Create a copy of the DataFrame to avoid modifying the original
    finaldf = df.copy()
    
    # Dropping observations based on the provided list of names
    if isinstance(names, str):
        finaldf = finaldf[finaldf['Last name, First name'] != names]
    elif isinstance(names, list):
        finaldf = finaldf[~finaldf['Last name, First name'].isin(names)]
    
    # Dropping observations based on Employment Status
    if Full_time_only:
        finaldf = finaldf[finaldf['Employment Status'] == "Full-Time"]
    
    # Dropping interns
    if Except_interns:
        finaldf = finaldf[finaldf['Employment Status'] != "Intern"]

    return finaldf

def unique_vars(column,df):
    varname=np.unique(df[str(column)].fillna(""))
    return varname

#count employees per condition and per employment status
#needs more work
def create_count(column,df):
    unique=unique_vars(str(column),df)
    dic={}
    
    for i in unique:
        count=np.sum(df[column] == i)
        dic.update({i:count})
            #Removing entries with 0 value
    for key in unique:
        if dic[key]==0:
            del(dic[key])
    return dic
#Allows to keep the latest/highest education based on rank
def max_edu(df):
    educations = {
        'nan': 0,
        "Associate's": 1,
        'Certificate': 4,
        "Bachelor's": 3,
        'Honours': 4,
        'Professional Diploma': 4,
        'Post-graduate Diploma': 4,
        "Master's": 7,
        'Post Masters': 7,
        'Doctorate': 7,
        'Fellowship': 7 
    }
    
    
    # Create a new column for the degree ranks
    df["Degree Rank"] = df["Degree"].map(educations)
    
    # Sort the df by Degree Rank with higher ones coming first
    df = df.sort_values(by="Degree Rank", ascending=False)
    
    # Delete rows with duplicate names and keep the first occurrence of a name
    # Thus only the highest education of an employee stays in the df
    df.drop_duplicates(subset="Last name, First name", keep='first', inplace=True)
    
    return df


def classify_degree(degree):
#Engineering, sciences, commerce, human sciences, law
#Certificates: CFA, CAIA, CA(SA)

    keywords_dict = {
        'Commerce': ['accountancy','marketing', 'accounting', 'accountant', 'finance', 'economics', 'economical', 'business', 'advertisement','hospitality','banking', 'fund', 'investment', 'commerce', 'MBA', 'marketing'],
        'Engineering': ['engineering', 'energy efficient', 'energy efficiency', 'energy'],
        'Human Sciences': ['government', 'social responsibility', 'risk', 'development', 'biodiversity', 'environmental', 'environment', 'policy','psychology', 'human resources', 'arts',   'development',  'environmental management', 'public management'],
        'Law': ['law', 'legal', 'tax'],
        'Sciences': ['mathematics', 'computer science', 'science', 'logic', 'statistics', 'chemistry', 'biology', 'physics', 'geology', 'geography'],
        'Other': []
    }
    for field, keywords in keywords_dict.items():
        for keyword in keywords:
            if keyword.lower() in degree.lower():
                return field
    return 'Other'


def mba_count(column, df):
    # Initialize the dictionary to hold counts for "Master's" and MBA related degrees
    dic = {"Master's": 0, "MBA": 0}

    # Count occurrences of "Master's"
    dic["Master's"] = np.sum(df["Degree"] == "Master's")
    
    # Count occurrences of "MBA" and "Master of Business Administration"
    # This should be seen as a keyword search
    mba_count = np.sum(df[column].str.contains("MBA", case=False, na=False)) + \
                np.sum(df[column].str.contains("Master of Business Administration", case=False, na=False))
    dic["MBA"] = mba_count

    # Remove entries with 0 value
    dic = {k: v for k, v in dic.items() if v > 0}
    
    return dic

def PieM(column,df):
    data=mba_count(column,df)
    x=list(data.keys())
    y=list(data.values())
    fig=plt.figure(figsize=(14, 8))  # Optional: setting the figure size
    plt.pie(y,labels=x,labeldistance=1.2,rotatelabels=False,pctdistance=1.1,autopct='%1.f%%')
    plt.title(f'Distribution of {column} for {np.sum(y)} employees')
    # left to do: Adjust the layout to center the pie chart
    
    return fig


def Pie(column,df):
    data=create_count(column,df)
    x=list(data.keys())
    y=list(data.values())
    fig=plt.figure(figsize=(14, 8))  # Optional: setting the figure size
    plt.pie(y,labels=x,labeldistance=1.2,rotatelabels=False,pctdistance=1.1,autopct='%1.f%%',colors=colors)
    plt.title(f'Distribution of {column} for {np.sum(y)} employees')
    # left to do: Adjust the layout to center the pie chart
    # Adjusting text properties

    
    return fig

#Creating stacked bar charts
def stacked_bar(column1, column2, df):

    # Group by column1 and column2, then count occurrences
    grouped = df.groupby([column1, column2]).size().unstack(fill_value=0)
    
    # Determine order of x-axis labels based on the longer column
    if len(grouped) >= len(df[column2].unique()):
        longer_column = column1
        shorter_column = column2
    else:
        longer_column = column2
        shorter_column = column1
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
    grouped.plot(kind='bar', stacked=True, ax=ax, color=colors[:len(grouped.columns)])
    
    # Set labels and title
    ax.set_xlabel(longer_column)
    ax.set_ylabel('Count')
    ax.set_title(f'Stacked Bar Chart of {column2} across {column1}')
    
    # Move legend outside the plot
    ax.legend(title=shorter_column, loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig

def stacked_barM(column1, column2, df):
    # Group by column1 and column2, then count occurrences
    grouped = df.groupby([column1, column2]).size().unstack(fill_value=0)
    
    # Determine order of x-axis labels based on the longer column
    if len(grouped) >= len(df[column2].unique()):
        longer_column = column1
        shorter_column = column2
    else:
        longer_column = column2
        shorter_column = column1
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
    grouped.plot(kind='bar', stacked=True, ax=ax)
    
    # Set labels and title
    ax.set_xlabel(longer_column)
    ax.set_ylabel('Count')
    ax.set_title(f'Stacked Bar Chart of {column2} across {column1}')
    
    # Move legend outside the plot
    ax.legend(title=shorter_column, loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig

#For now: all names that are in the list stay, which are MBA, Master of Business Administartion, etc
#
def keepmba(Full_time_only, Except_interns, df):
    keywords = ['MBA', 'Master of Business Administration', 'Masters of Business Administration']

    finaldf = df.copy()

    # Keeping observations where the 'Major/Specialization' includes any of the keywords
    finaldf = finaldf[finaldf['Major/Specialization'].str.contains('|'.join(keywords), case=False, na=False)]

    # Dropping observations based on Employment Status
    if Full_time_only:
        finaldf = finaldf[finaldf['Employment Status'] == "Full-Time"]

    # Dropping interns
    if Except_interns:
        finaldf = finaldf[finaldf['Employment Status'] != "Intern"]
    
    # Setting Major/Specialization to "MBA" for all remaining entries
    finaldf["Major/Specialization"] = "MBA"
    
    return finaldf


def unique_vars2(column,df):
    varname=np.unique(df[str(column)].fillna(""))
    return varname

def mba_count(column, df):
    # Initialize the dictionary to hold counts for "Master's" and MBA related degrees
    dic = {"Master's": 0, "MBA": 0}

    # Count occurrences of "Master's"
    dic["Master's"] = np.sum(df["Degree"] == "Master's")
    
    # Count occurrences of "MBA" and "Master of Business Administration"
    # This should be seen as a keyword search
    mba_count = np.sum(df[column].str.contains("MBA", case=False, na=False)) + \
                np.sum(df[column].str.contains("Master of Business Administration", case=False, na=False))
    dic["MBA"] = mba_count

    # Remove entries with 0 value
    dic = {k: v for k, v in dic.items() if v > 0}
    
    return dic

def PieMBA(column,column2,df,df2):
    data=create_count(column2,df) #contains some departments
    data2=create_count(column2,df2)#big, contains all departments
    dic=data2.copy()
    for key in data2.keys():
        if data.get(key,False) ==False:
            del(dic[key])
        else:
            k=(data.get(key,data2[key])/data2[key])

            dic.update({key:k})
    del(dic["CEO Office"])
    x=list(dic.keys())
    y=list(dic.values())
    total=np.sum(df[column]=='MBA')
    fig=plt.figure(figsize=(14, 8))  # Optional: setting the figure size
    plt.pie(y,labels=x,labeldistance=1.2,rotatelabels=False,pctdistance=1.1,autopct='%1.f%%')
    plt.title(f"Proportion of MBA per {column2} ({total-1} MBA's excluding CEO)")
    # left to do: Adjust the layout to center the pie chart
    plt.show()
    return fig

def add_text_page(df):

    HR = np.sum((df['Job Title'] == "Human Resources Officer") | (df['Job Title'] == "Human Resources Manager")| (df['Job Title'] == "Head of Human Resources")) 
    HRR=HR/df.shape[0]
    average_age=np.nanmean(df["Age"])
    median_age=np.nanmedian(df["Age"])
    average_length=np.nanmean(df["Length of service: Years"])
    median_length=np.nanmedian(df["Length of service: Years"])
    std_age=np.std(df['Age'])
    #employees above 50


    dic={'HR to employee Ratio':HRR,'HR':HR,'All':df.shape[0],'Average Age':average_age,"Median Age":median_age,"Average length of service in years":average_length,
         "Median length of service in years":median_length,'Standard Deviation Age':std_age}

    fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
    ax.text(0.5, 0.5, dic, transform=ax.transAxes, fontsize=12, va='center', ha='center', wrap=True)
    ax.axis('off')
    return fig

def stacked_funnel(column, df):
    # Generate years as strings
    max=int(np.max(df[column])+1)
    min=int(np.min(df[column]))
    years = np.arange(min, max)
    colors=['#0197B4']
    # Calculate counts of each age in the specified column
    counts = df[column].value_counts()
    
    # Ensure all years are present even if count is zero
    y = [counts.get(year, 0) for year in years]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(years, y, label=column,color=colors)
    ax.set_xlabel('Years')
    ax.set_ylabel('Count')
    ax.set_title(f'Stacked Funnel Chart for {column}')
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)
    plt.tight_layout()
    return fig
#Creates new columns to classify Revenue generating and Business enabling (definitions differ thus 2 columns)
#Also creates age group column
def frontVback(df):

    conditions = [
    (df["Department"] == "CEO Office"),
    (df["Department"] =="Investment and Asset Management"),
    (df["Department"]  =="New Ventures"),
    (df["Department"]  =="Finance and Fund Administration"),
    (df["Department"]  =="Operations"),
    (df["Department"]  =="Structuring and Valuations"),
    (df["Department"]  =="ESG"),
    (df["Department"]  =="Development and Construction Management"),
    (df["Department"]  =="Assurance")
    ]
# Define the corresponding choices for each condition
    choices = [
        "Revenue generating",
        "Revenue generating",
        "Revenue generating",
        "Business enabling",
        "Business enabling",
        "Business enabling",
        "Business enabling",
        "Business enabling",
        "Business enabling"
    ]

# Use np.select to apply the conditions and choices
    df["Revenue generating & Business enabling (Def 1)"] = np.select(conditions, choices, default="Unknown")

    conditions = [
    (df["Department"] == "CEO Office"),
    (df["Department"] =="Investment and Asset Management"),
    (df["Department"]  =="New Ventures"),
    (df["Department"]  =="Structuring and Valuations"),
    (df["Department"]  =="ESG"),
    (df["Department"]  =="Finance and Fund Administration"),
    (df["Department"]  =="Operations"),
    (df["Department"]  =="Development and Construction Management"),
    (df["Department"]  =="Assurance")
    ]
# Define the corresponding choices for each condition
    choices = [
        "Revenue generating",
        "Revenue generating",
        "Revenue generating",
        "Revenue generating",
        "Revenue generating",
        "Business enabling",
        "Business enabling",
        "Business enabling",
        "Business enabling"
    ]

# Use np.select to apply the conditions and choices
    df["Revenue generating & Business enabling (Def 2)"] = np.select(conditions, choices, default="Unknown")


    #return stacked_bar("Age","Length of service: Group",df)
    conditions = [
    (df["Age"] <= 20),
    (df["Age"] <= 30),
    (df["Age"] <= 40),
    (df["Age"] <= 50),
    (df["Age"] <= 60),
    (df["Age"] > 60)
    ]

# Define the corresponding choices for each condition
    choices = [
        'Up to 20 years old',
        '21 to 30 years old',
        '31 to 40 years old',
        '41 to 50 years old',
        '51 to 60 years old',
        'above 60 years old'
    ]

# Use np.select to apply the conditions and choices for regions
    df["Age group"] = np.select(conditions, choices, default="NA")



def stacked_bar_stats(column1,column2,df):
    finaldf = df.copy()
    data = finaldf[column1].groupby(finaldf[column2]).median()
    grand_mean = finaldf[column1].mean()
    grand_median = finaldf[column1].median()
    std_dev = finaldf[column1].groupby(finaldf[column2]).std()
    data_min = finaldf[column1].groupby(finaldf[column2]).min()
    data_max = finaldf[column1].groupby(finaldf[column2]).max()

    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed
    
    data.plot(kind='bar', stacked=True, ax=ax, color='#0197B4', legend=False)  # Exclude column1 from the legend

    # Set labels and title
    ax.set_xlabel(column2)
    ax.set_ylabel(f'Median {column1}')
    ax.set_title(f'Stacked Bar Chart of Median {column1} across {column2}')

    # Create a secondary y-axis that has the same scale as the first axis
    ax2 = ax.twinx()

    # Add error bars based on standard deviation
    ax2.errorbar(data.index, data, yerr=std_dev, fmt='none', capsize=5, ecolor='#024D5F', elinewidth=1, alpha=1)

    # Add grand mean and grand median lines on the secondary y-axis (for legend purposes)
    ax2.axhline(grand_mean, color='#FAB822', linestyle='--', linewidth=2, label=f'Grand Mean: {grand_mean:.2f}')
    ax2.axhline(grand_median, color='#D6D6D9', linestyle='--', linewidth=2, label=f'Grand Median: {grand_median:.2f}')

    # Hide the secondary y-axis ticks and labels
    ax2.yaxis.set_visible(False)

    # Set the same y-axis limits for both primary and secondary y-axes
    ax.set_ylim(bottom=min(ax.get_ylim()[0], ax2.get_ylim()[0]), top=max(ax.get_ylim()[1], ax2.get_ylim()[1]))
    ax2.set_ylim(ax.get_ylim())

    # Hide the secondary y-axis spines
    ax2.spines['right'].set_color('none')

    # Move legend outside the plot
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set smaller font size for x-axis ticks
    ax.tick_params(axis='x', labelsize=7)

    # Show the plot
    plt.tight_layout()
    return fig


def PieT(column,df):
    data=create_count(column,df)
    x=list(data.keys())
    y=list(data.values())
    fig=plt.figure(figsize=(14, 8))  # Optional: setting the figure size
    plt.pie(y,labels=x,labeldistance=1.2,rotatelabels=False,pctdistance=1.1,autopct='%1.f%%',colors=colors)
    plt.title(f'Distribution of {column} for {np.sum(y)} terminated employees')
    # left to do: Adjust the layout to center the pie chart
    plt.show()
    return fig

pd.options.mode.chained_assignment = None  # default='warn'

def ToMonth(column1,df,column2='Hire Date Month'):
# Assuming 'Adds' is your DataFrame and 'Hire Date' is a column in it
    df[column1] = pd.to_datetime(df[column1])
    df[column2]= df[column1].copy(deep=True)
# Create a new column 'Hire Date Month' with the year and month as strings
    df[column2] = df[column2].dt.to_period('M').astype(str)

def ToYear(column1,df,column2='Hire Date Year'):
# Assuming 'Adds' is your DataFrame and 'Hire Date' is a column in it
    df[column1] = pd.to_datetime(df[column1])
    df[column2]= df[column1].copy(deep=True)
# Create a new column 'Hire Date Month' with the year and month as strings
    df[column2] = df[column2].dt.to_period('Y').astype(str)
    #df.loc[:, column2] = df[column1].dt.to_period('Y').astype(str)

def approxTurn(dfa,dft):
    Termmonths=np.unique(dft["Termination Date Month"])
    dic={}
    for month in Termmonths:
        Rate=np.sum(dft['Termination Date Month']==month)/np.sum(dfa['Hire Date Month']<=month)*100
        dic.update({month:Rate})
    
    Rate=list(dic.values())
    Month=list(dic.keys())
    fig=plt.figure(figsize=(10, 6))
    plt.plot(Month, Rate, marker='o', linestyle='-', color='#0197B4')
    plt.title('Monthly Employee Turnover Rate')
    plt.xlabel('Month')
    plt.ylabel('Turnover Rate (%)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def add_figure_to_frame(fig, frame):
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)




#Main calculating function
def Dashboard3(filepath,filepath2 ,names, Full_time_only=True, Except_interns=True,save_as_pdf=False):
    if isinstance(filepath2, list):
        filepath2 = ";".join(filepath2)  # Join list elements if it's a list
    
    print(f"Debug: filepath2 is now: {filepath2} (type: {type(filepath2)})")
    # Checking for missing columns or wrong format
    error_message = Prerror(filepath)
    error_message2=Prerror(filepath2,False)
    if error_message:
        return error_message
    if error_message2:
        return error_message2
    # Reading the file as df
    df = pd.read_excel(filepath)
    #Additions and Terminations excel sheet
    turnover=pd.read_excel(filepath2,sheet_name=['Additions', 'Terminations'])
    dft=turnover["Terminations"]
    dfa=turnover["Additions"]

    
    #Removing dummies from addition sheet
    if isinstance(names, str):
        dfa = dfa[dfa['Name'] != names]
    elif isinstance(names, list):
        dfa=dfa[~dfa['Name'].isin(names)]

        #Removing dummies from addition sheet
    if isinstance(names, str):
        dft = dft[dft['Name'] != names]
        dft2=dft.copy()
    elif isinstance(names, list):
        dft=dft[~dft['Name'].isin(names)]
        dft2=dft.copy()
        #Changing full time defined term to full time
    dft['Employment Status'] = np.where(dft['Employment Status'] == 'Full-Time Defined Term', 'Full-Time', dft['Employment Status'])
    dfa['Employment Status'] = np.where(dfa['Employment Status'] == 'Full-Time Defined Term', 'Full-Time', dfa['Employment Status'])
    if Except_interns==True:
        dft = dft[dft['Employment Status'] != "Intern"]
        dfa = dfa[dfa['Employment Status'] != "Intern"]
    if Full_time_only==True:
        dft = dft[dft['Employment Status'] == "Full-Time"]
        dfa = dfa[dfa['Employment Status'] == "Full-Time"]  

    #Creating suitable columns for turnover rate tables 
    ToMonth('Hire Date',dfa)
    ToMonth('Hire Date',dft)
    ToMonth('Termination Date',dft,'Termination Date Month')
    #Creating suitable columns for finding first year resigners
    ToYear("Termination Date",dft,'Termination Date Year')
    ToYear("Hire Date",dft,'Hire Date Year')

    # Fill missing values with "NA" for all columns except "Age"
    df = df.apply(lambda col: col.fillna("NA") if col.name != "Age" else col)
    #Changing full time defined term to full time
    df['Employment Status'] = np.where(df['Employment Status'] == 'Full-Time Defined Term', 'Full-Time', df['Employment Status'])
    # Removing dummies and eventually non Full-Timers
    # Save result as finaldf
    ddf = remove_dummies(dummylist, Full_time_only, Except_interns, df)
    # Creating df with only MBA
    mbadf = keepmba(Full_time_only, Except_interns, ddf)
    # Keeps the highest education and removes employees without education!
    finaldf = max_edu(ddf)
    finaldf["Degree Field"] = finaldf['Major/Specialization'].apply(lambda x: classify_degree(x))
    #Adding front vs back office distinction column and Age groups
    frontVback(finaldf)
    

    text1=add_text_page(finaldf)

    fig1=approxTurn(dfa,dft)
    
    fig2 = stacked_bar("Degree", "Gender", finaldf)

    fig3=stacked_bar('Department',"Degree Field",finaldf)
    
    fig4=stacked_bar('Department','Major/Specialization',mbadf)

    fig5=stacked_bar('Length of service: Years','Degree Field',finaldf)

    fig6=stacked_bar('Length of service: Years',"Is Supervisor",finaldf)

    fig7=stacked_bar('Degree',"Degree Field",finaldf)

    fig8=stacked_bar_stats('Age','Department',finaldf)

    fig9=PieT("Employment Status",dft2)

    fig10=stacked_bar("Department","Employment Status",dft)

    fig11 = Pie("Degree", finaldf)

    fig12 = stacked_funnel('Age',finaldf)

    fig13= stacked_funnel('Length of service: Years',finaldf)

    fig14=stacked_bar('Department','Length of service: Years',finaldf)

    fig15=Pie('Revenue generating & Business enabling (Def 1)',finaldf)

    fig16=Pie('Revenue generating & Business enabling (Def 2)',finaldf)

    fig17=stacked_bar('Age group','Revenue generating & Business enabling (Def 1)',finaldf)

    fig18=stacked_bar('Age group','Revenue generating & Business enabling (Def 2)',finaldf)

    fig19=stacked_bar('Length of service: Years','Revenue generating & Business enabling (Def 1)',finaldf)

    fig20=stacked_bar('Length of service: Years','Revenue generating & Business enabling (Def 2)',finaldf)

    fig21=stacked_bar('Degree','Revenue generating & Business enabling (Def 1)',finaldf)

    fig22=stacked_bar('Degree','Revenue generating & Business enabling (Def 2)',finaldf)

    fig23=Pie("Role Grade", finaldf)

    fig24=stacked_bar('Length of service: Years','Role Grade',finaldf)

    fig25=stacked_bar_stats('Length of service: Years','Role Grade',finaldf)

    if save_as_pdf == True:
        
        filename=f'''Statistics_Full-Time status{Full_time_only}_ExcludeInterns{Except_interns}.pdf'''
        # Create pdf and charts
    
        pdf = PdfPages(filename)
        # Pie charts
        # Employees without degree get excluded
        
        pdf.savefig(fig1, bbox_inches='tight')

        pdf.savefig(fig2, bbox_inches='tight')

        pdf.savefig(fig3, bbox_inches='tight')

        pdf.savefig(fig4, bbox_inches='tight')

        pdf.savefig(fig5, bbox_inches='tight')

        pdf.savefig(fig6, bbox_inches='tight')

        pdf.savefig(fig7, bbox_inches='tight')

        pdf.savefig(fig8, bbox_inches='tight')

        pdf.savefig(fig9, bbox_inches='tight')

        pdf.savefig(fig10, bbox_inches='tight')

        pdf.savefig(fig11, bbox_inches='tight')

        pdf.savefig(fig12, bbox_inches='tight')

        pdf.savefig(fig13, bbox_inches='tight')

        pdf.savefig(fig14, bbox_inches='tight')

        pdf.savefig(fig15, bbox_inches='tight')

        pdf.savefig(fig16, bbox_inches='tight')

        pdf.savefig(fig17, bbox_inches='tight')

        pdf.savefig(fig18, bbox_inches='tight')

        pdf.savefig(fig19, bbox_inches='tight')

        pdf.savefig(fig20, bbox_inches='tight')

        pdf.savefig(fig21, bbox_inches='tight')

        pdf.savefig(fig22, bbox_inches='tight')

        pdf.savefig(fig23, bbox_inches='tight')

        pdf.savefig(fig24, bbox_inches='tight')

        pdf.savefig(fig25, bbox_inches='tight')

        pdf.savefig(text1, bbox_inches='tight')

        # Finishes pdf creation
        pdf.close()
    
        return pdf
    

    return fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10,fig11,fig12,fig13,fig14,fig15,fig16,fig17,fig18,fig19,fig20,fig21,fig22,fig23,fig24,fig25
        


def submit_form():
    # Retrieve and strip input values from the form
    filepath_raw = filepath_entry.get().strip()
    filepath2_raw = filepath_entry2.get().strip()

    # Remove any potential extra quotes around paths
    filepath_raw = filepath_raw.strip('"')
    filepath2_raw = filepath2_raw.strip('"')

    # Normalize file paths
    filepath = os.path.normpath(filepath_raw)
    filepath2 = os.path.normpath(filepath2_raw)
    #Get other arguments
    full_time_only = full_time_only_var.get()
    except_interns = except_interns_var.get()
    save_as_pdf = save_as_pdf_var.get()
    # Print paths to debug
    print(f"Normalized Filepath: '{filepath}'")
    print(f"Normalized Filepath2: '{filepath2}'")

    # Validate file paths
    if not os.path.isfile(filepath):
        print(f"Error: The file at '{filepath}' does not exist or cannot be accessed.")
        return

    if not os.path.isfile(filepath2):
        print(f"Error: The file at '{filepath2}' does not exist or cannot be accessed.")
        return

    # Retrieve additional form data
    full_time_only = full_time_only_var.get()
    except_interns = except_interns_var.get()
    save_as_pdf = True

    # Debug print statements
    print(f"Full Time Only: {full_time_only}")
    print(f"Except Interns: {except_interns}")
    print(f"Save as PDF: {save_as_pdf}")

    # Example list of names, replace with actual data as needed
    names_list = ["Test, Jess", "Soap, Jo", "Doe, Jane", "Seep, Jan", "Sweet, Janet"]

    #Give preview windows
    figure_list = Dashboard3(filepath, filepath2, ["Test, Jess", "Soap, Jo", "Doe, Jane", "Seep, Jan", "Sweet, Janet"], full_time_only, except_interns, save_as_pdf)

        # Clear the main window
    for widget in window.winfo_children():
        widget.destroy()

        # Update the window title
    window.title("Dashboard3 Output")
    window.geometry("2200x2200")

        # Create a Canvas widget
    canvas = tk.Canvas(window)
    canvas.pack(side="left", fill="both", expand=True)

        # Create a frame inside the canvas
    fig_frame = tk.Frame(canvas)

        # Add the frame to the canvas
    canvas.create_window((0, 0), window=fig_frame, anchor="nw")

        # Add all figures to the frame
    for figure in figure_list:
        add_figure_to_frame(fig=figure, frame=fig_frame)


    # Call the Dashboard3 function with the retrieved data
    return Dashboard3(filepath, filepath2, names_list, full_time_only, except_interns, save_as_pdf)




# Main window setup
window = tk.Tk()
window.title("Input for Corporate Employment Statistics")
window.geometry("1280x848")

# Image and canvas setup
image_path = resource_path("Hand.jpg")
image = Image.open(image_path)
img = ImageTk.PhotoImage(image)
canvas = tk.Canvas(window, width=image.width, height=image.height)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, anchor=tk.NW, image=img)

# Create a white rectangle
rectangle_width = 800
rectangle_height = 400
rectangle_x1 = (image.width - rectangle_width) // 4
rectangle_y1 = (image.height - rectangle_height) // 2
rectangle_x2 = rectangle_x1 + rectangle_width
rectangle_y2 = rectangle_y1 + rectangle_height
canvas.create_rectangle(rectangle_x1, rectangle_y1, rectangle_x2, rectangle_y2, fill="white")

# Form setup
style = ttk.Style()
style.configure("TLabel", font=('Arial', 12), background="white")
style.configure("TEntry", font=('Arial', 12))
style.configure("TCheckbutton", font=('Arial', 12), background="white")
style.configure("TButton", font=('Arial', 12), padding=10)

center_x = (rectangle_x1 + rectangle_x2) // 2
center_y = (rectangle_y1 + rectangle_y2) // 2
start_y = center_y - 100
y_offset = 50

filepath_label = ttk.Label(window, text="Filepath:", style="TLabel")
canvas.create_window(center_x - 250, start_y, anchor="nw", window=filepath_label)

filepath_entry = ttk.Entry(window, width=70, style="TEntry")
#filepath_entry.insert(0, "C:\\Users\\LeonardSugg\\Downloads\\TEST4_(Climate_Fund_Managers) (1).xlsx")
canvas.create_window(center_x - 150, start_y, anchor="nw", window=filepath_entry)

filepath_label2 = ttk.Label(window, text="Filepath (Additions & Terminations):", style="TLabel")
canvas.create_window(center_x - 350, start_y-50, anchor="nw", window=filepath_label2)

filepath_entry2 = ttk.Entry(window, width=70, style="TEntry")
#filepath_entry2.insert(0, "C:\\Users\\LeonardSugg\\Downloads\\Additions_Terminations_(Climate_Fund_Managers).xlsx")
canvas.create_window(center_x -90, start_y-50, anchor="nw", window=filepath_entry2)

full_time_only_var = tk.BooleanVar(value=True)
full_time_only_check = ttk.Checkbutton(window, text="Full Time status only", variable=full_time_only_var, style="TCheckbutton", padding=10)
canvas.create_window(center_x - 150, start_y + y_offset, anchor="nw", window=full_time_only_check)

save_as_pdf_var = tk.BooleanVar(value=False)
save_as_pdf_check = ttk.Checkbutton(window, text="Save as PDF", variable=save_as_pdf_var, style="TCheckbutton", padding=10)
canvas.create_window(center_x - 150, start_y + 2 * y_offset, anchor="nw", window=save_as_pdf_check)

except_interns_var = tk.BooleanVar(value=True)
except_interns_check = ttk.Checkbutton(window, text="Exclude Intern status", variable=except_interns_var, style="TCheckbutton", padding=10)
canvas.create_window(center_x - 150, start_y + 3 * y_offset, anchor="nw", window=except_interns_check)

submit_button = ttk.Button(window, text="Submit", command=submit_form, style="TButton")
canvas.create_window(center_x, start_y + 4 * y_offset, anchor="n", window=submit_button)

# Start the main event loop
window.mainloop()



