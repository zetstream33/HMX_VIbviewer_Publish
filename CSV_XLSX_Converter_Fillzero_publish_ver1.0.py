import streamlit as st
import pandas as pd
import csv
from io import BytesIO
import os

# 월에 대한 딕셔너리
month_dict = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

# 날짜 및 시간을 변환하는 함수
def convert_date_time(arrival_time):
    parts = arrival_time.split()
    month = month_dict[parts[1]]
    date = f"{parts[3]}-{month}-{parts[2]}"
    time = parts[4]
    return f"{date} {time}"

# 파일 변환 작업 수행
def convert_file(input_file):
    # 데이터 저장을 위한 리스트 생성
    data = []

    # CSV 파일 읽기
    input_file.seek(0)
    reader = csv.reader(input_file.read().decode('utf-8').splitlines())

    # 헤더 추출
    headers = next(reader)

    # 첫 번째 열 제목을 'Time'으로 변경
    headers[0] = 'Time'

    # CSV 파일의 모든 데이터를 변환하여 리스트에 추가
    for line in reader:
        if len(line) == 0:  # 빈 행 건너뛰기
            continue
        if len(line) < len(headers):  # 열 수가 맞지 않는 행 건너뛰기
            continue

        arrival_time = line[0]
        time_converted = convert_date_time(arrival_time)

        # 기존의 Date arrival time을 변환된 시간으로 교체
        new_line = [time_converted] + line[1:]
        data.append(new_line)

    # 리스트를 DataFrame으로 변환
    df = pd.DataFrame(data, columns=headers)
    return df

# 결측 시간을 0으로 채우는 함수
def fill_missing_times(df, missing_value=0.00):
    # 'Time' 열을 datetime 형식으로 변환
    df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d %H:%M:%S')

    # 데이터프레임에서 사용된 모든 센서 리스트 추출
    sensors = df['Sensor'].unique()

    # 데이터의 최소 및 최대 시간 추출
    min_time = df['Time'].min()
    max_time = df['Time'].max()

    # 모든 시간에 대해 1초 단위의 시간 생성
    all_times = pd.date_range(start=min_time, end=max_time, freq='S')

    # 결측 시간을 포함한 데이터프레임 생성
    new_data = []

    for current_time in all_times:
        for sensor in sensors:
            row = df[(df['Time'] == current_time) & (df['Sensor'] == sensor)]
            if row.empty:
                if missing_value == 0.00:
                    missing_value = int(0)  # 결측치를 채우는 값이 0.00이라면 정수형식으로 처리
                new_data.append([current_time, sensor, missing_value, missing_value, missing_value, missing_value, missing_value, missing_value])
            else:
                new_data.append([current_time, sensor] + row.iloc[0, 2:].tolist())

    # 새로운 데이터프레임 생성
    columns = ['Time', 'Sensor', 'grms_x', 'grms_y', 'grms_z', 'velocity_x(mm/s)', 'velocity_y(mm/s)', 'velocity_z(mm/s)']
    new_df = pd.DataFrame(new_data, columns=columns)

    # 시간 형식을 '%H:%M:%S'로 변환
    new_df['Time'] = new_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return new_df

# Streamlit 제목 및 설명
st.title("CSV to XLSX File Converter with DateTime Handling and Missing Time Filling")
st.write("CSV 파일을 업로드하여 날짜와 시간을 형식화하고 누락된 시간을 채운 Excel 파일로 변환하세요.")
st.write(f"**:green[Version 2024-11-18]** by QSHE Team Eojin LEE")
st.write(f"이 프로그램은 :blue[자동 업데이트]를 지원합니다. 업데이트를 진행하려면 **게이트웨이를 인터넷에 연결**해주세요.")
st.write(f"- - -")
st.write(f"\n Broadsens 사 **SVT-V** 센서와 **SVT-A** 센서 모두 사용 가능합니다. \n")
st.write(f"SVT-A 센서 데이터 사용시 결측 시간 채움 기능이 **비활성화** 됩니다. \n")

# 결측값 입력받기
missing_value = st.number_input("Enter the value to replace missing data:", value=0.00, step=0.01, min_value=0.00)

# 파일 업로드
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # 원본 파일명에서 확장자 제거
    file_name = os.path.splitext(uploaded_file.name)[0]

    # CSV 파일 변환
    try:
        # CSV 파일 헤더 읽기
        uploaded_file.seek(0)
        header_row = csv.reader(uploaded_file.read().decode('utf-8').splitlines())
        headers = next(header_row)

        # 열 구조 확인
        if 'grms_x' in headers:
            df = convert_file(uploaded_file)
            df = fill_missing_times(df, missing_value=missing_value)
        elif 'X_axis' in headers:
            df = convert_file(uploaded_file)
        else:
            raise ValueError("Unsupported file structure. Please check the input file.")

        # 결과를 새로운 엑셀 파일로 저장
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            writer.close()

        # 변환된 파일 이름 생성
        output_file_name = f"{file_name}_converted_filled.xlsx"

        # 파일 다운로드 링크 제공
        st.download_button(
            label="Download Converted Excel File",
            data=output.getvalue(),
            file_name=output_file_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("Conversion complete! You can download the file now.")
    except Exception as e:
        st.error(f"An error occurred during file conversion: {e}")
