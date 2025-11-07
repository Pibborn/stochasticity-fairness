import numpy as np
import pandas as pd

def generate_data(n,path):
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    S = np.random.randint(0, 2, size=n)
    y = (x1 + x2 >0).astype(int)

    data = np.column_stack((x1,x2,S,y))
    data = np.repeat(data,2,axis=0)

    for i in range(0,len(data),2):
        s_value = data[i,2]
        if s_value == 0:
            data[i,3] = 0
            data[i+1,3]=1
            
    df = pd.DataFrame(data,columns=["x1","x2","s","y"])
    df.to_csv(path)

if __name__=="__main__":
    generate_data(1000,"../data/01_raw/toydata_uncertainty.csv")