import csv
import pandas as pd
import tkinter as tk
from tkinter import Tk, Label, Button, Checkbutton, IntVar, OptionMenu, StringVar, filedialog, Scale, HORIZONTAL, Entry, \
    messagebox, Frame, ttk


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
def convert_file(input_file_path, output_file_path):
    # 데이터 저장을 위한 리스트 생성
    data = []

    # CSV 파일 읽기
    with open(input_file_path, 'r') as file:
        reader = csv.reader(file)

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

    # 숫자 데이터로 변환 시도 (첫 번째 열 제외)
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 결과를 XLSX 파일로 저장
    df.to_excel(output_file_path, index=False)


# 파일 처리 함수
def save_files(input_file_paths):
    for input_file_path in input_file_paths:
        # 파일명 추출 및 "_변환.xlsx" 추가
        original_filename = input_file_path.split('/')[-1]  # 파일명만 추출 (윈도우에서는 '\\' 사용)
        base_filename = original_filename.rsplit('.', 1)[0]  # 확장자 제거
        default_output_file = base_filename + "_변환.xlsx"

        # 동일 경로에 저장 경로 설정
        output_file_path = input_file_path.rsplit('/', 1)[0] + '/' + default_output_file  # 윈도우에서는 '\\' 사용

        # 파일 변환
        convert_file(input_file_path, output_file_path)

    messagebox.showinfo("완료", "모든 파일이 변환되어 저장되었습니다.")


# 파일 열기 버튼의 콜백 함수
def open_file():
    input_file_paths = filedialog.askopenfilenames(title="CSV 파일을 선택하세요", filetypes=[("CSV Files", "*.csv")])
    if input_file_paths:
        print(input_file_paths)  # 파일 경로 리스트 출력
        save_files(input_file_paths)

# GUI 생성
root = tk.Tk()
root.title("Broadsens 사 진동 센서(SVT Series) 출력 CSV 파일 to XLSX 변환기")

label_example = Label(root,
                      text='\n Last Updated at 2024-08-19 by QSHE Team Eojin LEE\n 이 프로그램은 현대무벡스(주) 품질환경안전팀의 자산입니다. \n무단 배포를 금지합니다.\n')
label_example.pack()


sp1 = ttk.Separator(root, orient="horizontal")
sp1.pack(fill="both")

sp1 = ttk.Separator(root, orient="horizontal")
sp1.pack(fill="both")

label_example = Label(root,
                      text='\n<사용 방법>\n Broadsens사 무선 진동 모니터링 시스템에서 출력된 원본 파일(예시 : accdata(1).csv) 선택하면 파일을 변환합니다. \n 파일 다중 선택을 지원합니다\n')
label_example.pack()

frame = tk.Frame(root)
frame.pack(pady=20, padx=20)



open_button = tk.Button(frame, text="CSV 파일 열기", command=open_file)
open_button.pack()

root.mainloop()
