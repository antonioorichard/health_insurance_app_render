import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler



class HealthInsurance:
  def __init__( self ):
      self.home_path = ''
      self.annual_premium_scaler                   = pickle.load( open( self.home_path + 'parameter/annual_premium_scaler.pkl', 'rb' ))
      self.age_scaler                              = pickle.load( open( self.home_path + 'parameter/Age_scaler.pkl', 'rb'))
      self.vintage_scaler                          = pickle.load( open( self.home_path + 'parameter/vintage_scaler.pkl', 'rb'))
      self.months_with_us_scaler                   = pickle.load( open( self.home_path + 'parameter/months_with_us_scaler.pkl', 'rb' ))
     # self.region_code_scaler                      = pickle.load( open( self.home_path + 'parameter/region_code_scaler.pkl', 'rb'))
      self.fe_policy_sales_channel_scaler          = pickle.load( open( self.home_path + 'parameter/policy_sales_channel_scaler.pkl', 'rb'))
      self.vehicle_age_1_2_Year_scaler             = pickle.load( open( self.home_path + 'parameter/vehicle_age_1_2_Year_scaler.pkl', 'rb'))
      self.vehicle_age_menor_q_1_Year_scaler       = pickle.load( open( self.home_path + 'parameter/vehicle_age_menor_q_1_Year_scaler.pkl', 'rb'))
      self.vehicle_age_maior_q_2_Years_scaler      = pickle.load( open( self.home_path + 'parameter/vehicle_age_maior_q_2_Years_scaler.pkl', 'rb'))
      self.gender_Female_scaler                    = pickle.load( open( self.home_path + 'parameter/gender_Female_scaler.pkl', 'rb'))
      self.gender_Male_scaler                      = pickle.load( open( self.home_path + 'parameter/gender_Male_scaler.pkl', 'rb'))
      self.vehicle_damage_scaler                   = pickle.load( open( self.home_path + 'parameter/vehicle_damage_scaler.pkl', 'rb'))
      

  def data_cleaning( self, df1):
      # 1.1. Rename Columns
      cols_new = [	
                    'driving_license',	
                    'previously_insured',
                    'annual_premium',
                    'age',
                    'vintage',
                    'months_with_us',
                    'region_code',
                    'policy_sales_channel',
                    'vehicle_damage',
                    'gender_Female',
                    'gender_Male',
                    'vehicle_age_1-2 Year',	
                    'vehicle_age_< 1 Year',
                    'vehicle_age_> 2 Years'
                     ]
      # rename
      df1 = df1[cols_new]

      return df1

  def feature_engineering(self, df2):
    # 2.0 Feature Engineering
    rs = RobustScaler()
    df2['region_code'] = rs.fit_transform( df2[['region_code']].values )
    # Vehicle Damage Number

  #  df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)

    # Vehicle Age

    #df2['vehicle_age'] = df2['vehicle_age'].apply(lambda x: 'over_2_years' if x == '> ')
    return df2

  def data_preparation( self, df5):
        # Rescaling

    #rs = RobustScaler()
    #mms = MinMaxScaler()

    # competition distance
    df5['annual_premium']        = self.annual_premium_scaler.transform( df5[['annual_premium']].values )

    #df5['region_code']           = self.region_code_scaler.transform( df5[['region_code']].values )

    df5['policy_sales_channel']  = self.fe_policy_sales_channel_scaler.transform( df5[['policy_sales_channel']].values )

    df5['vintage']               = self.vintage_scaler.fit_transform( df5[['vintage']].values )

    df5['age']                   = self.age_scaler.transform( df5[['age']].values )

    df5['months_with_us']        = self.months_with_us_scaler.transform( df5[['months_with_us']].values )

    df5['gender_Male']           = self.gender_Male_scaler.transform( df5[['gender_Male']].values )
    
    df5['gender_Female']         = self.gender_Female_scaler.transform( df5[['gender_Female']].values )

    df5['vehicle_age_< 1 Year']  = self.vehicle_age_menor_q_1_Year_scaler.transform( df5[['vehicle_age_< 1 Year']].values )

    df5['vehicle_age_1-2 Year']  = self.vehicle_age_1_2_Year_scaler.transform( df5[['vehicle_age_1-2 Year']].values )

    df5['vehicle_age_> 2 Years'] = self.vehicle_age_maior_q_2_Years_scaler.transform( df5[['vehicle_age_> 2 Years']].values )

    df5['vehicle_damage']        = self.vehicle_damage_scaler.transform( df5[['vehicle_damage']].values )
    
    
    # Encoding
    # gender - One Hot Encoding
    #df5 = pd.get_dummies( df5, prefix=['gender'], columns=['gender'] )

    # vehicle_age - One Hot Encoding
    #df5 = pd.get_dummies( df5, prefix=['vehicle_age'], columns=['vehicle_age'] )


    # vehicle_damage - subtituir 0 1
    #df5['vehicle_damage'] = df5['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)


    # Suponha que você tenha um DataFrame chamado df_test
    # com a coluna 'gender_Famale' contendo os valores {'False': 0, 'True': 1}

    # Crie um dicionário de mapeamento
    #mapeamento = {False: 0, True: 1}

    # Aplique o mapeamento à coluna 'gender_Famale'
    #df5['gender_Female'] = df_train['gender_Female'].map(mapeamento)


    #df5['gender_Male'] = df5['gender_Male'].map(mapeamento)



    #df5['vehicle_age_1-2 Year'] = df5['vehicle_age_1-2 Year'].map(mapeamento)



    #df5['vehicle_age_< 1 Year'] = df5['vehicle_age_< 1 Year'].map(mapeamento)



    #df5['vehicle_age_> 2 Years'] = df5['vehicle_age_> 2 Years'].map(mapeamento)

    return df5



  def get_prediction( self, model, original_data, test_data):
    # model prediction
    pred = model.predict_proba( test_data )

    # join prediction into original data

    original_data['score'] =  pred

    return original_data.to_json (orient = 'records', date_format = 'iso')



