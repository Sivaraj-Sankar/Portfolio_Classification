import logging
from abc import ABC, abstractmethod
from typing import Union
from typing_extensions import Annotated
from typing import Tuple



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler




class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    
    """
    
    @abstractmethod
    def handle_data(self,data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data
    
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame: 
        #return super().handle_data(data)
        """
        Preprocess data
        
        """
        try:
            data['job'].fillna('Missing',inplace=True)
            data['education'].fillna(method='ffill',inplace=True)
            data['contact'].fillna('cellular',inplace=True)
            data['poutcome'].fillna('failure',inplace=True)
            data.drop(columns = ['day','month'],inplace=True)
            #data.drop('target',inplace=True)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e 
        
class DataEncodingStrategy(DataStrategy):
    """
    Strategy for Encoding Categorical and Feature Engineering the data
    
    """ 
    @classmethod
    def AgeGroup(cls,encode_data: pd.Series) -> pd.Series:
        """
        Args:
            data: to encode the data
        Returns:
            pd.Series the encoded data        
        """
        ageGroup_bin = [0,25,35,45,55,65,75,85,100]
        ageGroup_labels = ['0-25','25-35','35-45','45-55','55-65','65-75','75-85','85-100']
        ageGroup = pd.cut(encode_data,bins=ageGroup_bin,labels=ageGroup_labels,right=False)
        return ageGroup
    
    def LabelEncoding_AgeGroup(cls, encode_data: pd.DataFrame,Age_Group_dic: dict) -> pd.Series:
        """
        Args:
            data: to encode the data
            Age_Group_dic: to Bin the Age
        Returns:
            pd.Series the encoded data        
        """
        labelencoder = LabelEncoder()
        encode_data['AgeGroup'] = labelencoder.fit_transform(encode_data['AgeGroup'].map(Age_Group_dic))
        return encode_data
        
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the Categorical data
         
        """
        try:
            data['AgeGroup'] = self.AgeGroup(data['age'])
            ## Ordinal Categorical
            Age_Group_dic = {
                      '0-25': 0, '25-35': 1,
                      '35-45': 2, '45-55': 3,
                      '55-65': 4, '65-75': 5,
                      '75-85':6, '85-100':7
                            }
            data['AgeGroup'] = self.LabelEncoding_AgeGroup(data[['AgeGroup']],Age_Group_dic)
            ## Nominal Categorical
            nomial_categorical = ['job','marital','education','default','housing','loan','contact','poutcome']
            for categorical in nomial_categorical:
                labelencoder = LabelEncoder()
                data[categorical] = labelencoder.fit_transform(data[categorical])
                
            return data 
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
        
class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    
    """ 
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        
        """
        try:
            X = data.drop(['y'],axis=1)
            y = data[['y']]
            y = y['y'].replace({'yes': 1, 'no': 0})
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e 

class BalancingAndScaling(DataStrategy):
    """  
    Strategy for Balancing training data and Scaling the Training and Testing data
    
    """
    def handle_data(self, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
        """
        balance the Train data & Scale the (Train&Test) data
        
        """
        try:
            #Balance
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            #Scaling
            numerical_features = ['age','balance','duration','campaign','pdays','previous']
            scaler = StandardScaler()
            scaler.fit(X_train[numerical_features])
            X_train_numerical = scaler.transform(X_train[numerical_features])
            X_test_numerical = scaler.transform(X_test[numerical_features])
            X_train[numerical_features] = X_train_numerical
            X_test[numerical_features] = X_test_numerical
            
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in Balancing the data {}".format(e))
            raise e
            
            
            
            

class DataCleaning:
    """
    Class for cleaning data Which process data, encode the ordinal category and nominal category and divides it into 
    train and test 
    
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        handle data
        
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e

class DataBalancing:
    """
    Class for cleaning data Which process data, encode the ordinal category and nominal category and divides it into 
    train and test 
    
    """
    def __init__(self, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series, strategy: DataStrategy):
        self.X_train = X_train 
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        handle data
        
        """
        try:
            return self.strategy.handle_data(self.X_train,self.X_test,self.y_train,self.y_test)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e

     
        