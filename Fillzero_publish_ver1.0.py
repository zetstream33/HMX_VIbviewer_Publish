import streamlit as st
import pandas as pd
from io import BytesIO
import os

# Streamlit 제목 및 설명
st.title("Excel File Processing with Missing Value Handling(결측 시간 대상 0으로 채우기)")
st.write("Upload an Excel file, handle missing values, and download the processed file.")
st.write("현재 SVT 센서 V 시리즈만 대응하도록 개발되었습니다.")

# 결측값 입력받기
missing_value = st.number_input("Enter the value to replace missing data:", value=0.00, step = 0.01, min_value=0.00)

# 파일 업로드
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # 원본 파일명에서 확장자 제거
    file_name = os.path.splitext(uploaded_file.name)[0]

    # 데이터 로드
    df = pd.read_excel(uploaded_file)

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
                    missing_value = int(0) # 만약 결측치를 채우는 값이 0.00(기본값) 이라면 정수형식인 0으로 바꿔서 엑셀에 넣어줌.

                # 결측값이 있는 경우, 사용자가 입력한 값으로 채움
                new_data.append([current_time, sensor, missing_value, missing_value, missing_value, missing_value, missing_value, missing_value])
            else:
                # 결측값이 없는 경우, 해당 값을 사용
                new_data.append([current_time, sensor] + row.iloc[0, 2:].tolist())

    # 새로운 데이터프레임 생성
    columns = ['Time', 'Sensor', 'grms_x', 'grms_y', 'grms_z', 'velocity_x(mm/s)', 'velocity_y(mm/s)', 'velocity_z(mm/s)']
    new_df = pd.DataFrame(new_data, columns=columns)

    # 시간 형식을 '%H:%M:%S'로 변환
    new_df['Time'] = new_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # 결과를 새로운 엑셀 파일로 저장
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        new_df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()

    # 변환된 파일 이름 생성
    output_file_name = f"{file_name}_fillzero.xlsx"

    # 파일 다운로드 링크 제공
    st.download_button(
        label="Download Processed Excel File",
        data=output.getvalue(),
        file_name=output_file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("Processing complete! You can download the file now.")