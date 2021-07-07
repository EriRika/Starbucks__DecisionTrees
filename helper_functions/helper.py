import pandas as pd
import numpy as np
import math
import json
from matplotlib import pyplot as plt
import seaborn as sns
import datetime

import sqlalchemy as db
from sqlalchemy import create_engine

def merge_trans_portfolio(transcript, portfolio):
    """
    Creates three new columns from event column: offer_id, amount, reward
    Merges with portfolio on offer id
    """
    transcript_df = transcript.copy()
    portfolio_df = portfolio.copy()
    
    #To align with duration
    transcript_df.time = transcript_df.time/24
    
    #Create column offer_id and amount based on transcript.value
    #This column contains an id only for none-transactional events
    transcript_df['offer_id'] = [x['offer id'] if ('offer id' in list(x.keys())) else x['offer_id'] if ('offer_id' in list(x.keys())) else None for x in transcript.value]
    transcript_df['amount'] = [x['amount'] if 'amount' in list(x.keys()) else None for x in transcript_df.value]
    transcript_df['reward_from_value'] = [x['reward'] if 'reward' in list(x.keys()) else None for x in transcript_df.value]
    transcript_df = transcript_df.merge(portfolio_df, left_on = 'offer_id', right_on ='id', how = 'left')
    return transcript_df

def k_bin_discretizer(col, df, exclude = True, buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] ):
    """
    Input: col to discretizer, data frame and buckets (with edges)
    Output: datframe with additional discretized column
    """
    df_temp = df.copy()
    df_temp[col + '_bucket'] = ''
    if exclude:
        print(exclude)
        df_temp.loc[df_temp[col]!=118,col + '_bucket'] = pd.qcut(df_temp.loc[df_temp[col]!=118,col], buckets, labels = [i for i in range(0, len(buckets)-1)])
        df_temp.loc[df_temp[col]==118, col + '_bucket'] = 118
    else:
        df_temp.loc[:,col + '_bucket'] = pd.qcut(df_temp[col], buckets, duplicates= 'drop')
    return df_temp

def get_dummy(df, col, dummy_na = False, drop_col = True):
    """
    Input: dataframe to create dummies, col to get dummy columns, 
    Output: original dataframe with additional discretized columns
    """
    new = pd.get_dummies(df[col], dummy_na = dummy_na, prefix = col)
    df = pd.concat([df, new], axis = 1)
    df.drop(columns = col, inplace = drop_col)
    return df

def create_channel_dummy(df):
    temp = []
    [temp.extend(x) for x in df.channels]
    channels = set(temp)
    for chnl in channels:
        df[chnl] = [ 1 if (chnl in x) else 0 for x in df.channels]
    return df

def clean_data_per_offer(transcript_df, offer_id):
    """
    id_start_time - shows time of offer received from the moment it was received. Consecutive rows have the same number. Once an offer is completed I set this to 500. If the same offer is sent a second time this shows the new start_time
    id_duration - As above logic but for the duration
    id_age - Age of the offer = time - id_start_time → Important in order to know if an offer is still valid
    id_is_valid - 1 where age <= duration, 0 else
    id_viewed_time - shows time of offer viewed from the moment it was viewed. coonsecutive logic is as in id_start_time. This is important to define, if a transaction and a reward was active or passive
    id_is_active - id_viewed_time - id_start_time <= id_duration
    id_active_transaction - Amount of transaction, if transaction was active
    id_passive_transaction - Amount of transaction, if transaction was passive , but valid!
    social - 1 if offer was active and came through Social Media
    email - see above
    mobile - see above
    web - see above
    reward_0, reward_2, reward_3, reward_5, reward_10 -flag 1 if offer has that reward
    difficulty_0, difficulty_2, difficulty_3, difficulty_5, difficulty_10- flag 1 if offer has that reward
    reward_help_column - reward of completed offer but backfilled to the previous rows
    id_active_reward - reward_help_column * id_is_active
    id_passive_reward - reward_help_column * (~id_is_active)    

    Additional columns (not offer specific)
    active_offers_simul - Sum of active offers at the same time
    valid_offers_simul - Sum of valid offers at the same time
    independent_transaction - value of transaction, which was done during a time, where no offer was running
    
    """
    df = transcript_df.copy()
    #Create id_start_time per offer
    mask_id_start_time = (df.event == 'offer received') & (df.offer_id ==offer_id)
    df.loc[mask_id_start_time,'id_start_time_'+ offer_id] = df.loc[mask_id_start_time,'time']
    #Create stopper when offer is completed
    mask_id_completed_time = (df.event == 'offer completed') & (df.offer_id ==offer_id)
    df.loc[mask_id_completed_time,'id_start_time_'+ offer_id] = 500
    df['id_start_time_'+ offer_id] = df.groupby(['person'])['id_start_time_'+ offer_id].apply(lambda x: x.ffill())
    #I actually want the completed offer to be still valid, but I don't have a better solution than htis
    df.loc[mask_id_completed_time,'id_start_time_'+ offer_id] = None
    df['id_start_time_'+ offer_id] = df.groupby(['person'])['id_start_time_'+ offer_id].apply(lambda x: x.ffill())
    
    #Create id_duration per offer
    mask_id_duration = (df.event == 'offer received') & (df.offer_id ==offer_id)
    df.loc[mask_id_duration,'id_duration_'+ offer_id] = df.loc[mask_id_duration,'duration']
    #Create stopper when offer is completed
    mask_id_duration_completed = (df.event == 'offer completed') & (df.offer_id ==offer_id)
    df.loc[mask_id_duration_completed,'id_duration_'+ offer_id] = -1
    df['id_duration_'+ offer_id] = df.groupby(['person'])['id_duration_'+ offer_id].apply(lambda x: x.ffill())
    #I actually want the completed offer to be still active at the completed status, but I don't have a better solution than htis
    df.loc[(df.event == 'offer completed') & (df.offer_id ==offer_id),'id_duration_'+ offer_id] = None
    df['id_duration_'+ offer_id] = df.groupby(['person'])['id_duration_'+ offer_id].apply(lambda x: x.ffill())
    
    
    #Create id_validation per offer based on duration time
    df['id_age_'+ offer_id] = df['time'] -df['id_start_time_'+ offer_id] 
    df['id_is_valid_'+ offer_id] = np.where((df['id_age_'+ offer_id]<=df['id_duration_'+ offer_id]) & ~((df.event =='offer completed') | (df['id_duration_'+ offer_id] ==-1)),1,0)
    
    #Create received_flag
    mask_id_received = (df.event == 'offer received') & (df.offer_id ==offer_id)
    df.loc[mask_id_received,'id_received_'+ offer_id] = 1
    df.loc[~mask_id_received,'id_received_'+ offer_id] = 0
    
    #Create id_is_active per offer based on view time and complete time
    mask_id_viewed = (df.event == 'offer viewed') & (df.offer_id ==offer_id)
    df.loc[mask_id_viewed,'id_viewed_time_'+ offer_id] = df.loc[mask_id_viewed,'time']
    df.loc[(df.event == 'offer received') & (df.offer_id ==offer_id),'id_viewed_time_'+ offer_id] = 1000
    df.loc[(df.event == 'offer completed') & (df.offer_id ==offer_id),'id_viewed_time_'+ offer_id] = 1000
    df['id_viewed_time_'+ offer_id] = df.groupby(['person','id_start_time_'+offer_id])['id_viewed_time_'+ offer_id].apply(lambda x: x.ffill())
    #I actually want the completed offer to be still active at the completed status, but I don't have a better solution than htis
    df.loc[(df.event == 'offer completed') & (df.offer_id ==offer_id),'id_viewed_time_'+ offer_id] = None
    df['id_viewed_time_'+ offer_id] = df.groupby(['person'])['id_viewed_time_'+ offer_id].apply(lambda x: x.ffill())
    
    df['id_is_active_'+ offer_id] =np.where((df['id_viewed_time_'+ offer_id] -df['id_start_time_'+ offer_id])<=df['id_duration_'+ offer_id],1,0)

    #Create id_active_transactions
    mask_id_active_transaction = (df.event == 'transaction') &  (df['id_is_valid_'+ offer_id] ==1)
    df.loc[mask_id_active_transaction,'id_active_transaction_'+ offer_id] = df.loc[mask_id_active_transaction,'amount'] * df.loc[mask_id_active_transaction,'id_is_active_'+ offer_id] 

    #Create id_passive_transactions
    mask_id_passive_transaction = (df.event == 'transaction') & (df['id_is_valid_'+ offer_id] ==1)
    df.loc[mask_id_passive_transaction,'id_passive_transaction_'+ offer_id] = df.loc[mask_id_passive_transaction,'amount']* np.where(df.loc[mask_id_passive_transaction,'id_is_active_'+ offer_id]==1,0,1)
    
    #Get channel for each offer_id if the offer was valid
    mask_id_offer =  (df.offer_id ==offer_id) & (df.event == 'offer received')
    for chnl in ['social','email', 'mobile', 'web']:
        df.loc[mask_id_offer,chnl+'_'+ offer_id] = df.loc[mask_id_offer,chnl]
        df[chnl + '_'+ offer_id] = df.groupby(['person'])[chnl + '_'+ offer_id].apply(lambda x: x.ffill())
        df[chnl + '_'+ offer_id] = df[chnl +'_'+ offer_id] * df['id_is_valid_'+ offer_id]
    
    #Get offer_type for each offer_id if the offer was valid
    mask_id_offer =  (df.offer_id ==offer_id) & (df.event == 'offer received')
    for offer_type in ['offer_type_bogo', 'offer_type_discount', 'offer_type_informational']:
        df.loc[mask_id_offer,offer_type+'_'+ offer_id] = df.loc[mask_id_offer,offer_type]
        df[offer_type + '_'+ offer_id] = df.groupby(['person'])[offer_type + '_'+ offer_id].apply(lambda x: x.ffill())
        df[offer_type + '_'+ offer_id] = df[offer_type +'_'+ offer_id] * df['id_is_valid_'+ offer_id]

    #Get reward_type for each offer_id if the offer was valid
    mask_id_offer =  (df.offer_id ==offer_id) & (df.event == 'offer received')
    for reward_type in ['reward_0', 'reward_2', 'reward_3', 'reward_5', 'reward_10']:
        df.loc[mask_id_offer,reward_type+'_'+ offer_id] = df.loc[mask_id_offer,reward_type]
        df[reward_type + '_'+ offer_id] = df.groupby(['person'])[reward_type + '_'+ offer_id].apply(lambda x: x.ffill())
        df[reward_type + '_'+ offer_id] = df[reward_type +'_'+ offer_id] * df['id_is_valid_'+ offer_id]

    #Get difficulty_type for each offer_id if the offer was valid
    mask_id_offer =  (df.offer_id ==offer_id) & (df.event == 'offer received')
    for diff_type in ['difficulty_0', 'difficulty_5', 'difficulty_7', 'difficulty_10', 'difficulty_20']:
        df.loc[mask_id_offer,diff_type+'_'+ offer_id] = df.loc[mask_id_offer,diff_type]
        df[diff_type + '_'+ offer_id] = df.groupby(['person'])[diff_type + '_'+ offer_id].apply(lambda x: x.ffill())
        df[diff_type + '_'+ offer_id] = df[diff_type +'_'+ offer_id] * df['id_is_valid_'+ offer_id]
        
    #Get active and passive rewards
    mask_id_offer =  (df.offer_id ==offer_id) & (df.event == 'offer completed')
    df['reward_help_column'] = df.groupby(['person'])['reward'].apply(lambda x: x.bfill())
    df.loc[mask_id_offer,'id_active_reward_'+ offer_id] = df.loc[mask_id_offer,'reward_help_column'] * df.loc[mask_id_offer,'id_is_active_'+ offer_id]
    df.loc[mask_id_offer,'id_passive_reward_'+ offer_id] = df.loc[mask_id_offer,'reward_help_column'] * np.where(df.loc[mask_id_offer,'id_is_active_'+ offer_id]==0,1,0)
    
    return df


def get_clean_data(transcript_df):
    df = transcript_df.copy()
    a = transcript_df.offer_id.unique()
    
    for offer_id in a[a != np.array(None)]:
        print(offer_id)
        df = clean_data_per_offer(df, offer_id)
    
    #Get amount of valid offers, which are running at the same time
    df['active_offers_simul'] = df.loc[:,df.columns.str.contains('id_is_active')].sum(axis = 1)
    df['valid_offers_simul'] = df.loc[:,df.columns.str.contains('id_is_valid')].sum(axis = 1)
    df['no_valid_offer_running'] = np.where(df.loc[:,df.columns.str.contains('id_is_valid')].sum(axis = 1)>=1,0,1)
    df['independent_transaction'] = df['amount'] * df['no_valid_offer_running']
    df['passive_sum'] =df.loc[:,df.columns.str.contains('id_passive_transaction_')].sum(axis = 1)/df.loc[:,df.columns.str.contains('id_is_valid_')].sum(axis = 1)
    df['active_sum'] =df.loc[:,df.columns.str.contains('id_active_transaction_')].sum(axis = 1)/df.loc[:,df.columns.str.contains('id_is_valid_')].sum(axis = 1)
    return df

def merge_trans_profile(transcript, profile):
    transcript_df = transcript.copy()
    profile_df = profile.copy()
    transcript_df = transcript_df.merge(profile_df, left_on = 'person', right_on ='id', how = 'left')
    return transcript_df

def transform_by_person_offer(transcript_df_final):
    """
    Takes temporary/help dataframe and performs grouping by person adn by offer id 
    Input: temporary/help dataframe
    Output: Grouped output contains rewards, offer_type, passive_transactions, active_transactions, reward_types,channle_types 
    on an aggregated level per customer
    Advantage: Algorithm can predict each offer per person separetaly
    Disadvantage: I am loosing information about what combination of offers was sent to the customer overall
    """
    df = transcript_df_final.copy()
    final_df=pd.DataFrame(columns = ['person', 'offer_id', 'valid_offers_simul', 'gender', 'age_bucket',
       'became_member_on', 'income', 'independent_transaction', 'start_time', 'offer_received',
       'active_transaction', 'passive_transaction', 'social', 'email',
       'mobile', 'web', 'offer_type_bogo', 'offer_type_discount',
       'offer_type_informational', 'active_reward', 'passive_reward'])
    offers = ['9b98b8c7a33c4b65b9aebfe6a799e6d9','0b1e1539f2cc45b7b9fa7c272da2e1d7',
              '2906b810c7d4411798c6938adc9daaa5','fafdcd668e3743c1bb461111dcafc2a4',
              '4d5c57ea9a6940dd891ad53e9dbe8da0','f19421c1d4aa40978ebb69ca19b0e20d',
              '2298d6c36e964ae4a3e7e9706d1fb8c2','3f207df678b143eea3cee63160fa8bed',
              'ae264e3637204a6fb9bb56bc8210ddfd','5a8bc65990b245e5a138643cd4eb9837']
    for offer_id in offers:
        grouping_dict ={}
        grouping_temp = {'id_received_' + offer_id:'sum',
                         'id_active_transaction_' + offer_id:'sum', 
                         'id_passive_transaction_' + offer_id:'sum',
                         'social_' + offer_id:'max',
                         'email_' + offer_id:'max',
                         'mobile_' + offer_id:'max',
                         'web_' + offer_id:'max',
                         'reward_0_'+offer_id:'max',
                         'reward_2_'+offer_id:'max',
                         'reward_3_'+offer_id:'max',
                         'reward_5_'+offer_id:'max',
                         'reward_10_'+offer_id:'max',
                         'difficulty_0_'+offer_id:'max',
                         'difficulty_5_'+offer_id:'max',
                         'difficulty_7_'+offer_id:'max',
                         'difficulty_10_'+offer_id:'max',
                         'difficulty_20_'+offer_id:'max',
                         'offer_type_bogo_' + offer_id:'max',
                         'offer_type_discount_' + offer_id:'max',
                         'offer_type_informational_' + offer_id:'max',
                         'id_active_reward_' + offer_id:'sum',
                         'id_passive_reward_' + offer_id:'sum',
                         'id_start_time_' + offer_id:'min',
                         'valid_offers_simul':'max'
                     }
        grouping_dict.update(grouping_temp) 
    
        total = {'gender':'first', 
        'age_bucket': 'first',
        'became_member_on': 'first', 
        'income': 'max',
        'independent_transaction':'sum',}
        grouping_dict.update(total)
        df['active_offers'] =np.where(df.valid_offers_simul>0,transcript_df_final.valid_offers_simul,1)
        df['id_active_transaction_'+offer_id] =df['id_active_transaction_'+offer_id]/df['active_offers']
        df['id_passive_transaction_'+offer_id] =df['id_passive_transaction_'+offer_id]/df['active_offers']
        df['id_active_reward_'+offer_id] =df['id_active_reward_'+offer_id]/df['active_offers']
        df['id_passive_reward_'+offer_id] =df['id_passive_reward_'+offer_id]/df['active_offers']
        grouping_temp = df.groupby(['person','id_start_time_' +offer_id], as_index = False).agg(grouping_dict)
        grouping_temp['offer_id'] = offer_id
        grouping_temp['start_time'] =grouping_temp['id_start_time_'+offer_id]
        grouping_temp['offer_received'] =grouping_temp['id_received_'+offer_id]
        grouping_temp['active_sum'] =grouping_temp['id_active_transaction_'+offer_id]
        grouping_temp['passive_sum'] =grouping_temp['id_passive_transaction_'+offer_id]

        grouping_temp['social'] =grouping_temp['social_'+offer_id]
        grouping_temp['email'] =grouping_temp['email_'+offer_id]
        grouping_temp['mobile'] =grouping_temp['mobile_'+offer_id]
        grouping_temp['web'] =grouping_temp['web_'+offer_id]
        grouping_temp['offer_type_bogo'] =grouping_temp['offer_type_bogo_'+offer_id]
        grouping_temp['offer_type_discount'] =grouping_temp['offer_type_discount_'+offer_id]
        grouping_temp['offer_type_informational'] =grouping_temp['offer_type_informational_'+offer_id]
        grouping_temp['active_reward_amount'] =grouping_temp['id_active_reward_'+offer_id]
        grouping_temp['passive_reward_amount'] =grouping_temp['id_passive_reward_'+offer_id]
        grouping_temp['reward_0_count'] = grouping_temp['reward_0_'+offer_id]
        grouping_temp['reward_2_count'] = grouping_temp['reward_2_'+offer_id]
        grouping_temp['reward_3_count'] = grouping_temp['reward_3_'+offer_id]
        grouping_temp['reward_5_count'] = grouping_temp['reward_5_'+offer_id]
        grouping_temp['reward_10_count'] = grouping_temp['reward_10_'+offer_id]
        grouping_temp['difficulty_0_count'] = grouping_temp['difficulty_0_'+offer_id]
        grouping_temp['difficulty_5_count'] = grouping_temp['difficulty_5_'+offer_id]
        grouping_temp['difficulty_7_count'] = grouping_temp['difficulty_7_'+offer_id]
        grouping_temp['difficulty_10_count'] = grouping_temp['difficulty_10_'+offer_id]
        grouping_temp['difficulty_20_count'] = grouping_temp['difficulty_20_'+offer_id]
        grouping_temp = grouping_temp.drop(columns = ['id_start_time_'+offer_id
                                                  , 'id_received_'+offer_id
                                                  ,'id_active_transaction_'+offer_id
                                                 ,'id_passive_transaction_'+offer_id
                                                 ,'social_'+offer_id
                                                 ,'email_'+offer_id
                                                 ,'mobile_'+offer_id
                                                 ,'web_'+offer_id
                                                 ,'offer_type_bogo_'+offer_id
                                                 ,'offer_type_discount_'+offer_id
                                                 ,'offer_type_informational_'+offer_id
                                                 ,'id_active_reward_'+offer_id
                                                 ,'id_passive_reward_'+offer_id
                                                 ,'reward_0_'+offer_id
                                                ,'reward_2_'+offer_id
                                                ,'reward_3_'+offer_id
                                                ,'reward_5_'+offer_id
                                                ,'reward_10_'+offer_id
                                                ,'difficulty_0_'+offer_id
                                                ,'difficulty_5_'+offer_id
                                                ,'difficulty_7_'+offer_id
                                                ,'difficulty_10_'+offer_id
                                                ,'difficulty_20_'+offer_id])
        final_df = final_df.append(grouping_temp)

        
    return final_df

def transform_by_person(df):
    transcript_df_final = df.copy()
    """
     Takes temprary/help dataframe and performs grouping by person
    Input: temprary/help dtaframe
    Output: Grouped output contains rewards, offer_type, passive_transactions, active_transactions, reward_types,channle_types 
    on an aggregated level per customer. Many columns appear 10 times as they are specific to the offer
    Advantage: This way the algorithm can learn about the impact of the combinationa and amount of orders sent during the 30 day period
    Disadvantage: I am loosing information about when and in which order an offer was sent to a person
    """
    grouping_dict = {}
    offers = ['9b98b8c7a33c4b65b9aebfe6a799e6d9','0b1e1539f2cc45b7b9fa7c272da2e1d7',
              '2906b810c7d4411798c6938adc9daaa5','fafdcd668e3743c1bb461111dcafc2a4',
              '4d5c57ea9a6940dd891ad53e9dbe8da0','f19421c1d4aa40978ebb69ca19b0e20d',
              '2298d6c36e964ae4a3e7e9706d1fb8c2','3f207df678b143eea3cee63160fa8bed',
              'ae264e3637204a6fb9bb56bc8210ddfd','5a8bc65990b245e5a138643cd4eb9837']
    for offer_id in offers:
        grouping_temp = {'id_received_' + offer_id:'sum',
                         'id_active_transaction_' + offer_id:'sum', 
                         'id_passive_transaction_' + offer_id:'sum',
                         'social_' + offer_id:'max',
                         'email_' + offer_id:'max',
                         'mobile_' + offer_id:'max',
                         'web_' + offer_id:'max',
                         'reward_0_'+offer_id:'max',
                         'reward_2_'+offer_id:'max',
                         'reward_3_'+offer_id:'max',
                         'reward_5_'+offer_id:'max',
                         'reward_10_'+offer_id:'max',
                         'difficulty_0_'+offer_id:'max',
                         'difficulty_5_'+offer_id:'max',
                         'difficulty_7_'+offer_id:'max',
                         'difficulty_10_'+offer_id:'max',
                         'difficulty_20_'+offer_id:'max',
                         'offer_type_bogo_' + offer_id:'max',
                         'offer_type_discount_' + offer_id:'max',
                         'offer_type_informational_' + offer_id:'max',
                         'id_active_reward_' + offer_id:'sum',
                         'id_passive_reward_' + offer_id:'sum',
                         'id_start_time_' + offer_id:'min',
                         }
        grouping_dict.update(grouping_temp) 
    
    total = {'gender':'first', 
        'age_bucket': 'first',
        'became_member_on': 'first', 
        'income': 'max',
        'independent_transaction':'sum',
        'passive_sum': 'sum',
        'active_sum': 'sum',}
    grouping_dict.update(total)
    grouping_by_person = transcript_df_final.groupby('person', as_index = False).agg(grouping_dict)

    for feat in ['email_','web_','mobile_','social_','bogo_','discount_','informational_','passive_reward_', 'active_reward_', 'reward_0_', 'reward_2_', 'reward_3_', 'reward_5_', 'reward_10_','difficulty_0_', 'difficulty_5_', 'difficulty_7_', 'difficulty_10_', 'difficulty_20_']:
        grouping_by_person[feat+ 'count'] = grouping_by_person.loc[:,grouping_by_person.columns.str.contains(feat)].sum(axis = 1)

    grouping_by_person =grouping_by_person.rename(columns = {'passive_reward_count':'passive_reward_amount','active_reward_count':'active_reward_amount'})    
    
    grouping_by_person['possible_total_reward'] = grouping_by_person.reward_2_count*2 + grouping_by_person.reward_3_count*3 + grouping_by_person.reward_5_count*5 + grouping_by_person.reward_10_count*10
    grouping_by_person['average_difficulty'] =  (grouping_by_person.difficulty_5_count * 5 + grouping_by_person.difficulty_7_count * 7 + grouping_by_person.difficulty_10_count * 10 )/ (grouping_by_person.difficulty_5_count + grouping_by_person.difficulty_7_count + grouping_by_person.difficulty_10_count)
    grouping_by_person.average_difficulty.fillna(0, inplace = True)
    return grouping_by_person

def get_final_training_df(df, group_by = ['person']):
    """
    Input: takes grouped file
    Output: creates dummy variables where necessary e.g. membership and income bucket
    Adds some additional columns like has_reward, has_active_reward, has_passive_reward
    """
    df_final = df.copy()
    #fill na values where

    fill_columns = 'social_|email_|mobile|web_|bogo_|discount_|informational_|reward_'
    df_final[df_final.columns[df_final.columns.str.contains(fill_columns)]] = df_final[df_final.columns[df_final.columns.str.contains(fill_columns)]].fillna(0)
    fill_columns = 'id_start_time_'
    df_final[df_final.columns[df_final.columns.str.contains(fill_columns)]] = df_final[df_final.columns[df_final.columns.str.contains(fill_columns)]].fillna(-1)

    features_received = list(df_final.columns[df_final.columns.str.contains('id_received_|id_start_time_')])
    features_count = list(df_final.columns[df_final.columns.str.contains('_count')])
    features_person = list(['gender', 'income','age_bucket','became_member_on'])
    target = list(['independent_transaction', 'passive_sum', 'active_sum', 'active_reward_amount', 'passive_reward_amount'])
    features_received.extend(features_count)
    features_received.extend(features_person)
    features_received.extend(target)
    features_received.extend(['person'])

    if 'offer_id' in group_by:
        df_final = get_dummy(df_final, 'offer_id', dummy_na = False, drop_col = False)
        features_received.extend(['offer_id'])
        
  
    #Create dummy variables

    df_final = get_dummy(df_final[features_received], 'gender', dummy_na = True, drop_col = False)
    df_final = get_dummy(df_final, 'age_bucket', drop_col = False)

    df_final['income_bucket'] = k_bin_discretizer('income', df_final, exclude = False,buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    #df_final.drop(columns = 'income', inplace = True)
    df_final = get_dummy(df_final, 'income_bucket', dummy_na = True)
    df_final['membership_start_year'] = [str(x)[0:4] for x in df_final.became_member_on]
    #df_final.drop(columns = 'became_member_on', inplace = True)
    df_final = get_dummy(df_final, 'membership_start_year')

    #Create target variable  
    df_final['has_passive_reward'] = np.where(df_final['passive_reward_amount']>0,1,0)
    df_final['has_active_reward'] = np.where(df_final['active_reward_amount']>0,1,0)
    df_final['has_reward'] = np.where((df_final['passive_reward_amount'] + df_final['active_reward_amount'])>0,1,0)
    

    return df_final


def store_data(filename, df):
    df_store = df.copy()
    engine = create_engine('sqlite:///output/'+filename+'.db')
    df_store.to_sql(filename, engine, index=False, if_exists='replace')

def load_data(filename, train_features, target_features):
    """
    Load Data from database
    Split into Features X and Target Y (Last 36 columns)
    """
    engine = create_engine('sqlite:///output/' + filename+'.db')
    df = pd.read_sql_table(filename, engine)

    Y = df.loc[:,target_features]
    X = df.loc[:,train_features]
   
    return X, Y