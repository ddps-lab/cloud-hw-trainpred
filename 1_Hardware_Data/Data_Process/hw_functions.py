import numpy as np 
import pandas as pd 
import io
import pickle
from luminol import anomaly_detector
from luminol.anomaly_detector import AnomalyDetector
import datetime
from datetime import datetime



def find_nearest(array, value): # 제일근처값찾기
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def scoreLuminolALLData(ts):   
    ts_dict = ts.to_dict()
    score_v = []
    my_detector = AnomalyDetector(ts_dict)
    score = my_detector.get_all_scores()
    for timestamp, value in score.iteritems():
        #print(timestamp, value)
        score_v.append(value)
    return score_v

def retFig(x, y): # 결과 pdf로 저장하기용도
    fig = plt.figure()
    a= plt.plot(x, y)
    return fig

def min_max_normalize(lst):
    normalized = []
    
    for value in lst:
        normalized_num = (value - min(lst)) / (max(lst) - min(lst))
        normalized.append(normalized_num)
    return normalized

# -----------------------------------------------------------------------------------------
# gpu wave 전처리
def Datacsv_preprocess(Data_T):  
    # ------ timestamp 찢어서 시/분/초 따로따로 컬럼 만들어서 저장하기 -----------------------------
    Data_T['date'] = pd.to_datetime(Data_T['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    Data_T['hour'] = Data_T['date'].dt.hour
    Data_T['minute'] = Data_T['date'].dt.minute
    Data_T['second'] = Data_T['date'].dt.second
    Data_T['millisecond'] = Data_T['date'].dt.microsecond

    cols = ['hour', 'minute','second']
    Data_T['hms'] =Data_T[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    Data_T['hms'] = Data_T['hms'].astype(str)

    # 전처리 ~~~~---------------------------------------------------------------------
    Data_new_timestamp = pd.DataFrame( Data_T['timestamp'] )
    Data_new_timestamp = Data_new_timestamp.replace(r' ', '', regex=True)
    Data_new_timestamp = Data_new_timestamp.replace(r':', '', regex=True)
    Data_new_timestamp = Data_new_timestamp.replace(r'/', '', regex=True)
    Data_new_timestamp = Data_new_timestamp.astype('float')
    Data_T['new_timestamp_gpu'] = Data_new_timestamp

    Data_T = Data_T.replace(r'W', '', regex=True)
    Data_T = Data_T.replace(r' ', '', regex=True)
    Data_T = Data_T.replace(r'MHz', '', regex=True)
    Data_T = Data_T.replace(r'MiB', '', regex=True)
    Data_T = Data_T.replace(r'%', '', regex=True)
    not_float_feature=[ ' clocks.current.sm [MHz]',' utilization.gpu [%]',' utilization.memory [%]',' power.draw [W]']     
    for i in not_float_feature:
        Data_T[i]=Data_T[i].astype('float')
        
    # 이름 앞에 공백 없애주기 ~~~~------------------------------------------------------------
    new_columns = []
    for i in Data_T.columns:
        new_columns.append(i.replace(" ", "")) 
    Data_T.columns = new_columns
     
    return Data_T , Data_new_timestamp

# -----------------------------------------------------------------------------------------
# cpu  wave 전처리
def dstatlog_preprocess(fn):
    with open(fn) as f:
        data = f.read().replace('|', '  ')
    data = data[data.rfind('system"\n')+8:]
    data = data.replace(',', ' ')
    cols = 'totalcpuusage_usr totalcpuusage_sys totalcpuusage_idl totalcpuusage_wai totalcpuusage_stl ' \
            'dsktotal_read dsktotal_writ ' \
            'nettotal_recv nettotal_send procs_run procs_blk procs_new ' \
            'memoryusage_used memoryusage_free memoryusage_buff memoryusage_cach ' \
            'iototal_read iototal_writ '  \
            'system_day system_time'.split()
    dstat_log=pd.read_csv(io.StringIO(data), delim_whitespace=True, skiprows=1, header=None, names=cols)
    # ------ timestamp 찢어서 시/분/초 따로따로 컬럼 만들어서 저장하기 -----------------------------
    dstat_log['date'] = pd.to_datetime(dstat_log['system_time'], format='%H:%M:%S')
    dstat_log['hour'] = dstat_log['date'].dt.hour
    dstat_log['minute'] = dstat_log['date'].dt.minute
    dstat_log['second'] = dstat_log['date'].dt.second

    cols = ['hour', 'minute','second']
    dstat_log['hms'] =dstat_log[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    dstat_log['hms'] = dstat_log['hms'].astype(str)
    return dstat_log

# -----------------------------------------------------------------------------------------
# 모델별 csv 파일 정리
def per_model_csv(file):
    
    with open(file, 'rb') as fr:
        epoch_start_end_timestamp = pickle.load(fr)

    epoch_start_timestamp_array =[]  # epoch 시작위치
    epoch_end_timestamp_array = []   # epoch 끝위치
    epoch_start_end_timestamp_Dataframe = pd.DataFrame(epoch_start_end_timestamp)

    for i in range(len(epoch_start_end_timestamp.keys())):
        epoch_start_end_timestamp_Dataframe = epoch_start_end_timestamp_Dataframe.replace(r' ', '', regex=True)
        epoch_start_end_timestamp_Dataframe = epoch_start_end_timestamp_Dataframe.replace(r':', '', regex=True)
        epoch_start_end_timestamp_Dataframe = epoch_start_end_timestamp_Dataframe.replace(r'-', '', regex=True)
        epoch_start_end_timestamp_Dataframe = epoch_start_end_timestamp_Dataframe.replace(r'/', '', regex=True)
    for i in range(len(epoch_start_end_timestamp_Dataframe.columns)):
        epoch_start_timestamp_array.append(epoch_start_end_timestamp_Dataframe[i][0])
        epoch_end_timestamp_array.append(epoch_start_end_timestamp_Dataframe[i][1])
        # 시작지점 리스트로 / # 끝지점 리스트로
    
    return epoch_start_timestamp_array , epoch_end_timestamp_array , epoch_start_end_timestamp_Dataframe

# -----------------------------------------------------------------------------------------
#dataframe 이 2/22 2:2:2.000 인경우 뒤의 0을 제외하고 [:-3 으로 계산해서 에러나기때문]
def epoch_latency_total(Data_T3, start_indx , end_index ):
    if(len(str(Data_T3['date_x'][end_index])[:-3]) != 26):
        d2 = datetime.strptime(str(Data_T3['date_x'][end_index-1])[:-3], "%Y-%m-%d %H:%M:%S.%f")
    else:
        d2 = datetime.strptime(str(Data_T3['date_x'][end_index])[:-3], "%Y-%m-%d %H:%M:%S.%f")

    if(len(str(Data_T3['date_x'][start_indx])[:-3]) != 26):
        d1 = datetime.strptime(str(Data_T3['date_x'][start_indx+1])[:-3], "%Y-%m-%d %H:%M:%S.%f")
    else:
        d1 = datetime.strptime(str(Data_T3['date_x'][start_indx])[:-3], "%Y-%m-%d %H:%M:%S.%f")

    return (d2-d1).total_seconds()


#----------------------------------------------------------------------------------------
def epoch_start_end_to_list(Data_GPUCPU, epoch_start_timestamp_array ,epoch_end_timestamp_array):
    # epoch 시작 끝지점이 Data_T 랑 완전히 같지않기 때문에 가장 가까운 지점만 골라내줌 ---------------
    new_timestamp_array = np.array(Data_GPUCPU['new_timestamp_gpu'])

    epoch_start_timestamp_nearest = [] # epoch start timestamp
    epoch_end_timestamp_nearest = []
    epoch_start_list = [] # epoch start index
    epoch_end_list = []

    for i in epoch_start_timestamp_array :  # start_timestamep + start index 어레이에 저장
        start_nearest = find_nearest (new_timestamp_array, float(i)) 
        epoch_start_timestamp_nearest.append(float(start_nearest))  # start 지점이랑 가까운 타임스탬프
        epoch_start_list.append(Data_GPUCPU[Data_GPUCPU['new_timestamp_gpu'] == float(start_nearest)].index[0])  # start 지점이랑 가까운 인덱스
    for i in epoch_end_timestamp_array : 
        end_nearest = find_nearest (new_timestamp_array, float(i)) 
        epoch_end_timestamp_nearest.append(float(end_nearest)) 
        epoch_end_list.append(Data_GPUCPU[Data_GPUCPU['new_timestamp_gpu'] == float(end_nearest)].index[0])
    return epoch_start_timestamp_nearest,epoch_end_timestamp_nearest,epoch_start_list,epoch_end_list





