"""
@last edit: Thu May 30 16:15:10
@author: Joker_C
@Description: Essembled Data Preprocessing
"""
import pandas as pd
import numpy as np

# 数据清洗函数封装
class data_preprocessing():
    # 自动顺序执行所有清洗操作
    def auto_clean(df=pd.DataFrame()):
        df=data_preprocessing.One_Hot(df)
        df=data_preprocessing.drop_Allnull_Col(df)
        df=data_preprocessing.drop_Allsame_Col(df)
        print ("Auto cleansing encoding has been done!")
        return df

    # 独热编码    
    def One_Hot(df=pd.DataFrame()):
        """
        Intro: 接受一个DataFrame对象，挑出其文本分类特征（object对象列）进行onehot编码，并与数值特征拼接。返回编码后的数据框。
        Note: 对于null/nan值，不分裂成单独的特征，会被隐藏。
        """
        ToSplit = df.select_dtypes(include=[object])   # 选出需要进行One Hot的列，再进行One Hot与原特征拼接
        df.drop(ToSplit.columns,axis=1,inplace=True)
        for i in ToSplit.columns:
            ToSplit = ToSplit.drop([i],axis=1).join(pd.get_dummies(ToSplit[i],prefix=i))
        df = df.join(ToSplit)
        print ("One-hot encoding has been done!")
        return df
	
    # 删除全空列
    def drop_Allnull_Col(df=pd.DataFrame()):
        """
        Intro: 接受一个DataFrame对象，删除所有值均为空的列。
        """
        droplist = np.isnan(df).any()
        droplists = []
        for i in droplist.index:
            if droplist[i] == True:
                droplists.append(i)
                print ("Drop all null feature {0}!".format(i))
        df.drop(droplists,axis=1,inplace=True)
        print ("Drop all null feature has been done!")
        return df

    # 删除值全部一样的feature
    def drop_Allsame_Col(df=pd.DataFrame()):
        """
        Intro: 接受一个DataFrame对象，删除所有值均相等的列。
        """
        droplists = []
        for i in df.columns:
            if df.loc[:,i].max() == df.loc[:,i].min():
                droplists.append(i)
                print ("Drop all same feature {0}!".format(i))
        df.drop(droplists,axis=1,inplace=True)
        print ("Drop all same feature has been done!")
        return df


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    print ("Below is pre_clean result:")
    X = pd.DataFrame([[1,0,'上海',1000],[2,0,'北京',1000],[3,0,'上海',1000]],columns=['编号','江苏','城市名称','贡献度'])
    X.replace(0,np.nan,inplace=True)
    print (X,'\n\nStart Cleansing!')
    X = data_preprocessing.auto_clean(X)
    # X=data_preprocessing.One_Hot(X)
    # X=data_preprocessing.drop_Allnull_Col(X)
    # X=data_preprocessing.drop_Allsame_Col(X)
    print ("Cleansing Done!\n\nBelow is post_clean result:")
    print (X)

