import os
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm 
def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    #pixels = mask.flatten()
    pixels = np.concatenate([[0], mask, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    if str(rle) == 'nan':
        return np.zeros( height * width, dtype=np.uint8)#.reshape(2048, 2048)
    s = rle.split()
    flag = 0
    for i in s:
        if i.startswith('[0'):
            print(i)
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img#.reshape(height, width)

# CSV 파일들이 저장된 폴더 경로
# 앙상블할 폴더들을 ensemble_input에 넣어주면 됩니다.
input_path = "/opt/ml/input/level2_cv_semanticsegmentation-cv-09/ensemble_input"
output_path = "/opt/ml/input/level2_cv_semanticsegmentation-cv-09/ensemble_output"
threshold = 0.45
# 폴더 내의 CSV 파일들을 가져와서 읽기
input_files = [file for file in os.listdir(input_path) if file.endswith(".csv")]

# CSV 파일들을 읽어 데이터프레임으로 저장
dfs = []
for file in input_files:
    file_path = os.path.join(input_path, file)
    df = pd.read_csv(file_path)
    dfs.append(df)
    print(file)

with open(os.path.join(output_path, 'output.csv'), 'w', newline='') as output_file:
    writer = csv.writer(output_file)

    col_name = ['image_name','class','rle']
    writer.writerow(col_name)
    col = [0,0,0]
    ensemble_df = pd.DataFrame()
    for i in tqdm(range(len(dfs[0]))):
        masks = [decode_rle_to_mask(str(df['rle'][i]), 2048, 2048) for df in dfs]
        ensemble_mask = np.stack(masks).mean(axis=0)  # Hard voting: 평균값 사용
        ensemble_mask = np.where(ensemble_mask >= threshold, 1, 0)  # Threshold 이상인 값은 1, 그렇지 않은 값은 0으로 변환
        for c in range(2):
            col[c] = dfs[0][col_name[c]][i]
        col[2] = encode_mask_to_rle(ensemble_mask)
        writer.writerow(col)