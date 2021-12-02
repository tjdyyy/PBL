# 필요 함수 모음

import requests
import json
import pandas as pd
from konlpy.tag import Okt
import re
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import os
import librosa
import soundfile as sf
from moviepy.editor import *
import shutil

DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data-out/'

okt = Okt()

# 논의 필요
stop_words = ['은', '는', '이', '가', '하', '아', '것', '들','의', '있', '되', '수', '보', '주', '등', '한']


# 16000 Hz samplerate + 30초 자르기
def trim_audio_data(audio_file, save_file):
    sr = 48000
    sec = 30

    y, sr = librosa.load(audio_file, sr=sr)
    #ny = y[:sr * sec]

    # 길이 제한 x
    ny = y[:]

    # librosa.output.write_wav(save_file + '.wav', ny, sr)
    sf.write(save_file, ny, sr, format='WAV', endian='LITTLE', subtype='PCM_16')

    return save_file


# clova
class ClovaSpeechClient:
    # Clova Speech invoke URL
    invoke_url = ''
    # Clova Speech secret key
    secret = ''

    def req_url(self, url, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                wordAlignment=True, fullText=True, diarization=None):
        request_body = {
            'url': url,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/url',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_object_storage(self, data_key, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                           wordAlignment=True, fullText=True, diarization=None):
        request_body = {
            'dataKey': data_key,
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'Content-Type': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        return requests.post(headers=headers,
                             url=self.invoke_url + '/recognizer/object-storage',
                             data=json.dumps(request_body).encode('UTF-8'))

    def req_upload(self, file, completion, callback=None, userdata=None, forbiddens=None, boostings=None,
                   wordAlignment=True, fullText=True, diarization=None):
        request_body = {
            'language': 'ko-KR',
            'completion': completion,
            'callback': callback,
            'userdata': userdata,
            'wordAlignment': wordAlignment,
            'fullText': fullText,
            'forbiddens': forbiddens,
            'boostings': boostings,
            'diarization': diarization,
        }
        headers = {
            'Accept': 'application/json;UTF-8',
            'X-CLOVASPEECH-API-KEY': self.secret
        }
        print(json.dumps(request_body, ensure_ascii=False).encode('UTF-8'))

        files = {
            'media': open(file, 'rb'),
            'params': (None, json.dumps(request_body, ensure_ascii=False).encode('UTF-8'), 'application/json')
        }
        response = requests.post(headers=headers, url=self.invoke_url + '/recognizer/upload', files=files)
        return response

# clova stt 이용한 결과 json 파일로 저장
def clova(filename):
    jsonname = './clovatest' # 추후 네이밍 룰 정하기

    # res = ClovaSpeechClient().req_url(url='http://example.com/media.mp3', completion='sync')
    # res = ClovaSpeechClient().req_object_storage(data_key='data/media.mp3', completion='sync')
    res = ClovaSpeechClient().req_upload(filename, completion='sync')
    print(res.text)
    json_data = res.text
    json_decode_data = json_data

    with open(jsonname + '.json', 'w', encoding='UTF-8') as f:
        f.write(json_decode_data)

    return jsonname+'.json'


# json 파일에서 필요한 부분만 추출하여 csv 파일로 저장
def json_to_csv(jsonfile):
    with open(jsonfile, "r", encoding="utf8") as f:
        contents = f.read()  # string 타입
        json_d = json.loads(contents)

    sentence = []
    start = []
    end = []

    for i in range(0, len(json_d["segments"])):
        sentence.append(json_d["segments"][i]['text'])
        start.append(json_d["segments"][i]['start'])
        end.append(json_d["segments"][i]['end'])

    df = pd.DataFrame({'sentence': sentence,
                       'start': start,
                       'end': end})
    df.head()

    csvname = './stt_df' # 추후 네이밍 룰 정하기
    df.to_csv(csvname+'.csv', index = False)
    return csvname+'.csv'


# 데이터 전처리 1
def cleanse(text):
    pattern = re.compile(r'\s+')
    text = re.sub(pattern, ' ', str(text))
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ0-9]', ' ', str(text))
    print(text)
    return text

# 데이터 전처리 2
def preprocessing(sentence, okt, remove_stopwords=False, stop_words=[]):
    # 함수의 인자는 다음과 같다.
    # sentence : 전처리할 텍스트
    # okt : okt 객체를 반복적으로 생성하지 않고 미리 생성후 인자로 받는다.
    # remove_stopword : 불용어를 제거할지 선택 기본값은 False
    # stop_word : 불용어 사전은 사용자가 직접 입력해야함 기본값은 비어있는 리스트

    # 1. 중복 제거.
    sentence_text = repeat_normalize(sentence, num_repeats=2)

    # 2. okt 객체를 활용해서 형태소 단위로 나눈다.
    wd_sentence = okt.morphs(sentence_text, stem=True)

    if remove_stopwords:
        # 불용어 제거(선택적)
        wd_sentence = [token for token in wd_sentence if not token in stop_words]

    return wd_sentence

# stt 결과 전처리
def clean_data(csvfile):
    test_data = pd.read_csv(csvfile)
    clean_test_sentence = []

    # 전처리 1
    test_data['sentence'] = test_data['sentence'].apply(cleanse)

    # 전처리 2
    for review in test_data['sentence']:
        # 비어있는 데이터에서 멈추지 않도록 string인 경우만 진행
        if type(review) == str:
            clean_test_sentence.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
        else:
            clean_test_sentence.append([])  # string이 아니면 비어있는 값 추가

    clean_test_onlysentence_df = pd.DataFrame({'clean_word': clean_test_sentence})

    # 토큰화
    clean_train_data = pd.read_csv(DATA_IN_PATH + 'clean_train_38.csv')
    array_text = []
    for arr in clean_train_data['text']:
        array_text.append(eval(arr))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(array_text) # train에 사용된 text
    test_sequences = tokenizer.texts_to_sequences(clean_test_sentence)

    MAX_SEQUENCE_LENGTH = 38 # 문장 평균값

    test_inputs = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')  # test data 벡터화

    # 데이터 저장
    TEST_INPUT_DATA = 'test_input.npy'
    TEST_CLEAN_DATA = 'test_clean.csv'

    # # 저장하는 디렉토리가 존재하지 않으면 생성
    # if not os.path.exists(DATA_IN_PATH):
    #     os.makedirs(DATA_IN_PATH)

    # 전처리 된 테스트 데이터를 csv 형태로 저장
    clean_test_onlysentence_df.to_csv(DATA_IN_PATH + TEST_CLEAN_DATA, index=False)

    # 전처리 된 테스트 데이터를 넘파이 형태로 저장
    np.save(open(DATA_IN_PATH + TEST_INPUT_DATA, 'wb'), test_inputs)

    return TEST_INPUT_DATA, TEST_CLEAN_DATA


# 모델 불러와서 적용
def load_model(npyfile, sttcsv):
    newmodel = tf.keras.models.load_model(DATA_OUT_PATH+'cnn_classifier_kr_model')

    stt_test_input = np.load(open(DATA_IN_PATH + npyfile, 'rb'))
    stt_test_input = pad_sequences(stt_test_input, maxlen=stt_test_input.shape[1])

    predict = newmodel.predict(stt_test_input)
    return_data = pd.read_csv(DATA_IN_PATH + sttcsv)
    return_data['predict'] = predict # 데이터 프레임에 predict 열 추가
    return_data.to_csv(DATA_IN_PATH + 'result' + sttcsv, index=False)
    return 'result' + sttcsv


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# 음성 무음 처리
def make_quiet_wav(wav, result, start, end, output_name):
    quiet, _ = librosa.load(DATA_IN_PATH + 'quiet.wav', sr=44100)  # 무음 파일 로드
    data, _ = librosa.load(wav, sr=44100)  # 변환할 음성 파일 로드
    # data -> samplerate로 나누면 초(s) 단위

    length = len(result)  # 단위 개수
    lenq = len(quiet)  # 694575

    for i in range(0, length):
        tmp = int(end[i] - start[i])
        r = result[i]
        print("for문 : [{}] {} : {}".format(r, start[i] / 44100, end[i] / 44100))
        if (r == 1): # 유해 문장인 경우 무음 처리
            if tmp > lenq:
                data[end[i] - lenq:end[i]] = quiet[:]
                print("quiet : {}:{}".format((end[i] - lenq) / 44100, end[i] / 44100))
            else:
                data[start[i]:end[i]] = quiet[:tmp] * 1.0
                print("quiet : {}:{}".format(start[i] / 44100, end[i] / 44100))
    sf.write(output_name, data, 44100)

# 비디오 복제
def copy_video(video, idd):
    data_output = DATA_IN_PATH + str(idd) + "/"
    print(bcolors.OKGREEN + '\nplease wait! \n' + bcolors.ENDC)

    # 파일 복사하기 (사본 생성)
    shutil.copy2(DATA_IN_PATH+video, data_output+"copied_"+video)
    print(bcolors.OKGREEN + '\nMake copy of video ! \n' + bcolors.ENDC)

# 비디오에서 음성 추출
def wav_from_video(video, data_output):
    videoclip = VideoFileClip(DATA_IN_PATH + video)
    audioclip = videoclip.audio
    audioclip.write_audiofile(data_output + "copy.wav")  # 음성 wav 추출하기
    print(bcolors.OKGREEN + '\nChange mp4 to wav ! \n' + bcolors.ENDC)
    return videoclip

# 비디오 편집
def video_processing(video, resultfile):
    df = pd.read_csv(DATA_IN_PATH + resultfile, header=0)
    start = []
    end = []
    result = []

    # 단위: ms (1000분의 1초)
    # ms -> 데이터 단위화 ( x * 샘플률(44100) / 1000 )
    for i in range(len(df)):
        start.append(int(df['start'][i] * 44.1))
        end.append(int(df['end'][i] * 44.1))

        if df['predict'][i] >= 0.7:
            result.append(1)
        else:
            result.append(0)

    idd = int(input('ID: '))
    data_output = DATA_OUT_PATH + str(idd) + "/"
    try:
        if not os.path.exists(data_output):
            os.makedirs(data_output)
    except OSError:
        print('Error: Creating directory. ' + data_output)
        quit()

    copy_video(video, idd) # 영상 복제
    wav_from_video(video, data_output) # 음성 추출
    # 무음 처리
    make_quiet_wav(data_output + "copy.wav", result, start, end, data_output + "{}_result_{}.wav".format(idd, video))

    videoclip = VideoFileClip(DATA_IN_PATH + video)
    videoclip = videoclip.set_audio(AudioFileClip(data_output + "{}_result_{}.wav".format(idd, video)))
    videoclip.write_videofile(data_output + "{}_result_{}".format(idd, video))

    print(bcolors.OKGREEN + '\nNow you can check!!! ' + bcolors.ENDC)

    print(bcolors.OKGREEN + '\nDone !!! \n' + bcolors.ENDC)

    return data_output + "{}_result_{}.wav".format(idd, video)