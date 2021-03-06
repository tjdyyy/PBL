import func

DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out/'

# 원본 파일 - 실제로는 입력받거나 전달받기
original = 'zip.mp4'
split = original.split('.')

videofile = DATA_IN_PATH + original

# video 파일 넣고 wav 파일 받아오기
audiofile = func.trim_audio_data(videofile, DATA_IN_PATH + split[0] + '.wav')

# 클로바 stt 사용 후 json 파일 받아오기
jsonfile = func.clova(audiofile)
#jsonfile = 'clovatest.json'

# json -> csv
csvfile = func.json_to_csv(jsonfile)

# csv 파일 -> 전처리된 npy 파일, csv 파일
TEST_INPUT_DATA, TEST_CLEAN_DATA = func.clean_data(csvfile)

# 전처리된 npy 파일, stt csv 파일(clean X) -> 모델 적용 후 결과 csv 파일
resultfile = func.load_model(TEST_INPUT_DATA, csvfile)

# 비디오 파일, 결과 csv 파일 -> 비디오 편집
resultvideo = func.video_processing(original, resultfile)
### 근데 wav랑 video랑 안 합쳐짐..  mp3로 바꿨다가 합쳐도 안 합쳐짐..
### 그래서 일단 받아 온 경로는 무음 처리된 wav 파일 경로
###### 나만 안 됨 - 맥 문제?

