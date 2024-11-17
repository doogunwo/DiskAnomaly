import subprocess
import pandas as pd
import re
from datetime import datetime

# 데이터 저장을 위한 리스트
data = []

# blktrace와 blkparse를 실행하여 데이터 수집
cmd = "sudo blktrace -d /dev/sda5 -o - | blkparse -i - -o -"
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# 정규 표현식으로 필드 추출
line_pattern = re.compile(
    r"(?P<timestamp>\d+\.\d+)\s+\d+\s+(?P<operation>[A-Z])\s+(?P<rwbs>\S+)\s+(?P<block>\d+)\s+\+\s+(?P<size>\d+)"
)

# 출력 데이터 읽기
try:
    for line in process.stdout:
        match = line_pattern.search(line)
        if match:
            # 각 줄의 이벤트 정보를 딕셔너리로 저장
            event = {
                "timestamp": float(match.group("timestamp")),
                "operation": match.group("operation"),
                "rwbs": match.group("rwbs"),
                "block": int(match.group("block")),
                "size": int(match.group("size"))
            }
            data.append(event)

except KeyboardInterrupt:
    process.terminate()

# 수집된 데이터로 DataFrame 생성
df = pd.DataFrame(data)

# 데이터 전처리
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', origin='unix')
df.set_index('timestamp', inplace=True)

# 지표 계산
window = '1S'  # 1초 단위로 집계
aggregated = df.resample(window).apply({
    'size': 'sum',                   # disk_io: 총 I/O 크기
    'block': lambda x: x.nunique(),  # seek_count: 블록 위치의 고유 값 수
    'operation': 'count',            # iops: 초당 요청 수
})
aggregated.columns = ['disk_io', 'seek_count', 'iops']

# throughput 계산
aggregated['throughput'] = aggregated['disk_io'] / int(window[:-1])

# 결과 출력
print(aggregated.head())

# CSV 파일로 저장
aggregated.to_csv('processed_trace_data.csv')

