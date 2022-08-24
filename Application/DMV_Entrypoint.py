import pandas as pd
import numpy as np
import tensorflow as tf
from DMV_Text_Classification import ClassificationModels
from DMV_Pattern_Denial import Pattern_Denial
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel
from random import randint
import time
import json
import gcsfs
import h5py
import traceback

def ELP_Validation(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
    try:
        request_json = request.get_json()
        start_time = time.time()
        vAR_result_message = ""
        vAR_input_text = request_json['CONFIGURATION']
        vAR_error_message = PreRequestValidation(request_json)

        
        if len(vAR_error_message["Error Message"])==0:
            vAR_profanity_result,vAR_result_message = Profanity_Words_Check(vAR_input_text)
            if not vAR_profanity_result:
                vAR_message_level_1 = "Level 1 Accepted"
            elif vAR_profanity_result:
                vAR_message_level_1 = vAR_result_message

            vAR_regex_result,vAR_pattern = Pattern_Denial(vAR_input_text)
            if not vAR_regex_result:
                vAR_message_level_2 = "Denied - Similar to " +vAR_pattern+ " Pattern"
            elif vAR_regex_result:
                vAR_message_level_2 = "Level 2 Accepted"

            if request_json['MODEL'].upper()=='RNN':
                
                vAR_result,vAR_result_data,vAR_result_target_sum = LSTM_Model_Result(vAR_input_text)
                vAR_result_data = vAR_result_data.to_json(orient='records')
                vAR_random_id = Random_Id_Generator()
                if vAR_result_target_sum>20:
                    vAR_recommendation_level_3 = "Denied"
                    vAR_reason_level_3 = "Since the profanity probability exceeds the threshold(sum of probability >20%)"
                else:
                    vAR_recommendation_level_3 = "Accepted"
                    vAR_reason_level_3 = "Since the profanity probability less than the threshold(sum of probability <20%)"
                vAR_response_time = round(time.time() - start_time,2)

                return {"1st Level(Direct Profanity)":{"Is accepted":not vAR_profanity_result,"Message":vAR_message_level_1},
                "2nd Level(Denied Pattern)":{"Is accepted":vAR_regex_result,"Message":vAR_message_level_2},
                "3rd Level(Model Prediction)":{"Is accepted":vAR_result,"Recommendation":vAR_recommendation_level_3,"Reason":vAR_reason_level_3,"Profanity Classification":json.loads(vAR_result_data),
                'Sum of all Categories':vAR_result_target_sum},
                'Order Id':vAR_random_id,'Configuration':vAR_input_text,
                'Response time':str(vAR_response_time)+" secs","Error Message":vAR_error_message["Error Message"]}

            elif request_json['MODEL'].upper()=='BERT':

                vAR_result,vAR_result_data,vAR_result_target_sum = BERT_Model_Result(vAR_input_text)
                vAR_result_data = vAR_result_data.to_json(orient='records')
                vAR_random_id = Random_Id_Generator()
                if vAR_result_target_sum>20:
                    vAR_recommendation_level_3 = "Denied"
                    vAR_reason_level_3 = "Since the profanity probability exceeds the threshold(sum of probability >20%)"
                else:
                    vAR_recommendation_level_3 = "Accepted"
                    vAR_reason_level_3 = "Since the profanity probability less than the threshold(sum of probability <20%)"
                vAR_response_time = round(time.time() - start_time,2)

                return {"1st Level(Direct Profanity)":{"Is accepted":not vAR_profanity_result,"Message":vAR_message_level_1},
                "2nd Level(Denied Pattern)":{"Is accepted":vAR_regex_result,"Message":vAR_message_level_2},
                "3rd Level(Model Prediction)":{"Is accepted":vAR_result,"Recommendation":vAR_recommendation_level_3,"Reason":vAR_reason_level_3,"Profanity Classification":json.loads(vAR_result_data),
                'Sum of all Categories':vAR_result_target_sum},
                'Order Id':vAR_random_id,'Configuration':vAR_input_text,
                'Response time':str(vAR_response_time)+" secs","Error Message":vAR_error_message["Error Message"]}

        else:
            return vAR_error_message

    except BaseException as e:
        print('In Error Block - '+str(e))
        print('Error Traceback - '+str(traceback.print_exc()))
        return {'Error Message':'### '+str(e)}



def PreRequestValidation(request_json):
    vAR_error_message = ""

    if 'CONFIGURATION' not in request_json or len(request_json['CONFIGURATION'])==0 or request_json['CONFIGURATION']=='nan':
        vAR_error_message =vAR_error_message+ "### Mandatory Parameter CONFIGURATION is missing"
    if 'SG_ID' not in request_json or len(request_json["SG_ID"])==0 or request_json['SG_ID']=='nan':
        vAR_error_message =vAR_error_message+ "### Mandatory Parameter Simply Gov Id is missing"
    if 'ORDER_GROUP_ID' not in request_json or len(request_json["ORDER_GROUP_ID"])==0 or request_json['ORDER_GROUP_ID']=='nan':
        vAR_error_message =vAR_error_message+ "### Mandatory Parameter ORDER_GROUP_ID is missing"
    if 'ORDER_ID' not in request_json or len(request_json["ORDER_ID"])==0 or request_json['ORDER_ID']=='nan':
        vAR_error_message = vAR_error_message+"### Mandatory Parameter ORDER_ID is missing"
    if 'ORDER_DATE' not in request_json or len(request_json["ORDER_DATE"])==0 or request_json['ORDER_DATE']=='nan':
        vAR_error_message = vAR_error_message+"### Mandatory Parameter ORDER_DATE is missing"

    if len(request_json['CONFIGURATION'])>7:
        vAR_error_message = vAR_error_message+"### ELP Configuration can not be more than 7 characters"
    return {'Error Message':vAR_error_message}


def Random_Id_Generator():
    vAR_random_id = randint(10001, 50000)
    return vAR_random_id



def LSTM_Model_Result(vAR_input_text):
    # Input Data Preprocessing
    vAR_data = pd.DataFrame()
    vAR_target_columns = ['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
    vAR_model_obj = ClassificationModels(vAR_data,vAR_target_columns)
    vAR_test_data = pd.DataFrame([vAR_input_text],columns=['comment_text'])
    vAR_test_data['Toxic'] = None
    vAR_test_data['Severe Toxic'] = None
    vAR_test_data['Obscene'] = None
    vAR_test_data['Threat'] = None
    vAR_test_data['Insult'] = None
    vAR_test_data['Identity Hate'] = None
    print('Xtest length - ',len(vAR_test_data))
    vAR_corpus = vAR_model_obj.data_preprocessing(vAR_test_data)
    print('Data Preprocessing Completed')
    vAR_X,vAR_y = vAR_model_obj.word_embedding_vectorization(vAR_corpus,vAR_test_data)
    print('Vectorization Completed Using Word Embedding')
    print('var X - ',vAR_X)
    print('var Y - ',vAR_y)
    
    vAR_load_model = tf.keras.models.load_model('gs://dmv_elp_project/saved_model/LSTM/LSTM_RNN_Model')

    vAR_model_result = vAR_load_model.predict(vAR_X)
    print('LSTM result - ',vAR_model_result)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = (np.sum(vAR_model_result)*100).round(2)
    vAR_result_data.index = pd.Index(['Percentage'],name='category')
    vAR_result_data = vAR_result_data.astype(float).round(5)*100
    

    # Sum of predicted value with 20% as threshold
    if vAR_target_sum>20:
        return False,vAR_result_data,vAR_target_sum
    else:
        return True,vAR_result_data,vAR_target_sum





def BERT_Model_Result(vAR_input_text):
    
    
    
    vAR_test_sentence = vAR_input_text
    vAR_target_columns = ['Toxic','Severe Toxic','Obscene','Threat','Insult','Identity Hate']
    
    # Name of the BERT model to use
    model_name = 'bert-base-uncased'

    # Max length of tokens
    max_length = 128

    # Load transformers config and set output_hidden_states to False
    config = BertConfig.from_pretrained(model_name)
    #config.output_hidden_states = False

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
    
    vAR_test_x = tokenizer(
    text=vAR_test_sentence,
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
    start_time = time.time()
    # print('Copying Model')
    # subprocess.call(["gsutil cp gs://dsai_saved_models/BERT/model.h5 /tmp/"],shell=True)
    # print('Model File successfully copied')  
    MODEL_PATH = 'gs://dmv_elp_project/saved_model/BERT/model.h5'
    # MODEL_PATH = 'gs://dsai_saved_models/BERT/BERT_MODEL_64B_4e5LR_3E'
    FS = gcsfs.GCSFileSystem()
    with FS.open(MODEL_PATH, 'rb') as model_file:
         model_gcs = h5py.File(model_file, 'r')
         vAR_load_model = tf.keras.models.load_model(model_gcs,compile=False)
    # vAR_load_model = tf.keras.models.load_model('gs://dsai_saved_models/BERT/model.h5',compile=False)
    # vAR_load_model = tf.keras.models.load_model(MODEL_PATH,compile=False)
    # vAR_load_model = tf.keras.models.load_model('/tmp/model.h5',compile=False)

    # vAR_load_model = Load_BERT_Model()
    
    print("---Model loading time %s seconds ---" % (time.time() - start_time))
    

    vAR_model_result = vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)
    
    # if "vAR_load_model" not in st.session_state:
    #     st.session_state.vAR_load_model = tf.keras.models.load_model('DSAI_Model_Implementation_Sourcecode/BERT_MODEL_64B_4e5LR_3E')
    # vAR_model_result = st.session_state.vAR_load_model.predict(x={'input_ids': vAR_test_x['input_ids'], 'attention_mask': vAR_test_x['attention_mask']},batch_size=32)
    vAR_result_data = pd.DataFrame(vAR_model_result,columns=vAR_target_columns)
    vAR_target_sum = (np.sum(vAR_model_result)*100).round(2)
    vAR_result_data.index = pd.Index(['Percentage'],name='category')
    vAR_result_data = vAR_result_data.astype(float).round(5)*100
    
    if vAR_target_sum>20:
        return False,vAR_result_data,vAR_target_sum
    else:
        return True,vAR_result_data,vAR_target_sum




def Number_Replacement(vAR_val):
    vAR_output = vAR_val
    if "1" in vAR_val:
        vAR_output = vAR_output.replace("1","I")
    if "2" in vAR_val:
        vAR_output = vAR_output.replace("2","Z")
    if "3" in vAR_val:
        vAR_output = vAR_output.replace("3","E")
    if "4" in vAR_val:
        vAR_output = vAR_output.replace("4","A")
    if "5" in vAR_val:
        vAR_output = vAR_output.replace("5","S")
    if "8" in vAR_val:
        vAR_output = vAR_output.replace("8","B")
        print('8 replaced with B - ',vAR_val)
    if "0" in vAR_val:
        vAR_output = vAR_output.replace("0","O")
    print('number replace - ',vAR_output)
    return vAR_output



def Binary_Search(data, x):
    low = 0
    high = len(data) - 1
    mid = 0
    i =0
    while low <= high:
        i = i+1
        print('No.of iteration - ',i)
        mid = (high + low) // 2
        
        # If x is greater, ignore left half
        if data[mid] < x:
            low = mid + 1
 
        # If x is smaller, ignore right half
        elif data[mid] > x:
            high = mid - 1
 
        # means x is present at mid
        else:
            return mid
 
    # If we reach here, then the element was not present
    return -1

def Profanity_Words_Check(vAR_val):
    vAR_input = vAR_val
    vAR_badwords_df = pd.read_csv('gs://dmv_elp_project/data/badwords_list.csv',header=None)
    print('data - ',vAR_badwords_df.head(20))
    vAR_result_message = ""
    
#---------------Profanity logic implementation with O(log n) time complexity-------------------
    # Direct profanity check
    vAR_badwords_df[1] = vAR_badwords_df[1].str.upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_input)
    if vAR_is_input_in_profanity_list!=-1:
        vAR_result_message = 'Input ' +vAR_val+ ' matches with direct profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list]
        
        return True,vAR_result_message
    
    # Reversal profanity check
    vAR_reverse_input = "".join(reversed(vAR_val)).upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_reverse_input)
    if vAR_is_input_in_profanity_list!=-1:
        vAR_result_message = 'Input ' +vAR_val+ ' matches with reversal profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list]
        return True,vAR_result_message
    
    # Number replacement profanity check
    vAR_number_replaced = Number_Replacement(vAR_val).upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_number_replaced)
    if vAR_is_input_in_profanity_list!=-1: 
       vAR_result_message = 'Input ' +vAR_val+ ' matches with number replacement profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list]
       return True,vAR_result_message
    
    # Reversal Number replacement profanity check(5sa->as5->ass)
    vAR_number_replaced = Number_Replacement(vAR_reverse_input).upper()
    vAR_is_input_in_profanity_list = Binary_Search(vAR_badwords_df[1],vAR_number_replaced)
    if vAR_is_input_in_profanity_list!=-1:  
        vAR_result_message = 'Input ' +vAR_val+ ' matches with reversal number replacement profanity - '+vAR_badwords_df[1][vAR_is_input_in_profanity_list]
        return True,vAR_result_message
    
    print('1st lvl message - ',vAR_result_message)
    return False,vAR_result_message