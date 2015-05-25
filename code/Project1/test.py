import pandas as pd
import numpy as np


from numpy.random import randn


print np.log2(0.5)
df = pd.DataFrame({'Name' : ['ALEX', 'ALEX', 'ALEX', 'ALEX',
                           'ALEX', 'bar', 'foo', 'foo'],
                    'Sex' : ['M', 'F', 'M', 'F',
                           'F', 'M', 'M', 'M'],
                    'DOB' : [2001, 2002,2001, 2002,2001, 2002,2001, 2002,],
                    'Count' : [2.1, 30.786989, 4.0, 2.0, 30.56565, 4.0,2.0, 90]})

print df.ix[1]
grouped = df.groupby(['Name'])



for name, group in grouped:
    print(name)
    print(group) 

            
def func(df):
    
    df_dob = df.groupby('DOB', as_index=False).aggregate({'Count' : np.sum})

    df_1980 = df_dob[df_dob['DOB'] == 2001]

    if not df_1980.empty:
        count = float(df_1980.Count)



    #print df_dob
    #df_dob = df_dob.sort('DOB')  
    
    min_count = float(df_dob.iloc[0].Count)
    max_count = float(df_dob.iloc[-1].Count) 
    percent_change = 0.0
    if (min_count != max_count):
        percent_change = (max_count - min_count)*100/min_count
    return pd.Series([percent_change, min_count, max_count], index=['percent_change', 'min_count', 'max_count'])

print grouped.apply(func)