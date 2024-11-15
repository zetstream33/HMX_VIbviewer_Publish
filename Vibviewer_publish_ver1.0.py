import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from io import BytesIO

sensor_mapping = {
            166: 'HP Guide roller L',
            167: 'HP Guide roller R',
            192: 'HP Wheel',
            193: 'OP Guide roller L',
            194: 'OP Guide roller R',
            195: 'OP Wheel',
            247: 'Sensor Bracket (주행)',
            248: '기상반 PCB',
            249: 'Carriage',
            250: 'Mast (상단)'
        }

# 엑셀 파일 생성 함수 (모든 센서를 순회하며 평균/최대값을 계산하여 엑셀 파일에 저장)
# 엑셀 파일 생성 함수 (모든 센서를 순회하며 평균/최대값을 계산하여 엑셀 파일에 저장)
def create_excel_output(sensor_mapping, df, start_time, end_time, threshold):
    output = BytesIO()

    # 엑셀 파일로 저장하기 위한 writer 설정
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data = []

        for sensor_no, major_part in sensor_mapping.items():
            # 해당 센서의 데이터를 필터링
            sensor_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time) & (df['Sensor'] == sensor_no)]

            if sensor_data.empty:
                # 센서 데이터가 없을 경우 건너뛰기
                continue

            # X, Y, Z 축에 따라 컬럼 이름 설정
            columns = [f'grms_x', f'grms_y', f'grms_z'] if 'grms_x' in sensor_data.columns else ['X_axis', 'Y_axis',
                                                                                                 'Z_axis']

            # Threshold 적용
            filtered_data = sensor_data[sensor_data[columns].max(axis=1) >= threshold]

            if filtered_data.empty:
                # Threshold를 넘는 데이터가 없으면 건너뛰기
                continue

            # 각 축별 평균 및 최대값 계산
            avg_x = filtered_data[columns[0]].mean()
            avg_y = filtered_data[columns[1]].mean()
            avg_z = filtered_data[columns[2]].mean()
            max_x = filtered_data[columns[0]].max()
            max_y = filtered_data[columns[1]].max()
            max_z = filtered_data[columns[2]].max()

            # 출력 형식에 맞춰 데이터 추가
            data.append([
                major_part,
                f"평균(Avg.) : X축 {avg_x:.2f}G/ Y축 {avg_y:.2f}G/ Z축 {avg_z:.2f}G",
                f"최대(Max.) : X축 {max_x:.2f}G/ Y축 {max_y:.2f}G/ Z축 {max_z:.2f}G"
            ])

        # DataFrame 생성
        df_output = pd.DataFrame(data, columns=['주요 확인 부', '평균값(Avg.)', '최대값(Max.)'])
        # DataFrame을 엑셀 파일로 저장
        df_output.to_excel(writer, index=False, sheet_name='Sensor Data')

    # BytesIO 포인터를 맨 앞으로 이동
    output.seek(0)
    return output
# 데이터 로드 함수
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Time'] = pd.to_datetime(df['Time'])
    return df

# 데이터 형식에 따른 Y축 라벨 결정 함수
def determine_yaxis_label(df):
    if 'grms_x' in df.columns or 'velocity_x(mm/s)' in df.columns:
        return '가속도 RMS | 단위 : g(표준 중력가속도)' if 'grms_x' in df.columns else '속도 RMS | 단위 : mm/s'
    elif 'X_axis' in df.columns and 'Y_axis' in df.columns and 'Z_axis' in df.columns:
        return '가속도 값 (9.81 m/s²) | 단위 : g(표준 중력가속도)'
    else:
        return 'Unknown Data Type'

def plot_data_plotly(df, start_time, end_time, selected_sensors, selected_axes, data_type, threshold, show_text,
                     marker_size, title, start_at_midnight, average, yaxis_major_tick, yaxis_max_value, yaxis_min_value,xaxis_major_tick):
    try:
        if start_time > end_time:
            st.error("Start time must be earlier than end time.")
            return

        if not selected_sensors:
            st.warning("No sensors selected.")
            return

        columns = []
        data_labels = []

        # 컬럼 선택 로직
        if 'grms_x' in df.columns:
            for axis in selected_axes:
                columns.append(f'grms_{axis.lower()}')
                data_labels.append(f'{axis}축')
        elif 'X_axis' in df.columns:
            for axis in selected_axes:
                columns.append(f'{axis}_axis')
                data_labels.append(f'{axis}축')

        # 데이터 필터링
        filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time) &
                         (df['Sensor'].isin(selected_sensors)) &
                         (df[columns].max(axis=1) >= threshold)]

        if start_at_midnight:
            min_time = filtered_df['Time'].min()
            min_date = min_time.normalize()  # 자정 시간으로 설정
            filtered_df.loc[:, 'Time'] = filtered_df['Time'].apply(lambda x: min_date + (x - min_time))
            filtered_df.loc[:, 'Time'] = filtered_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # 날짜와 시간이 같은 데이터에 대해 평균 계산
        if average:
            filtered_df = filtered_df.groupby(['Time', 'Sensor'])[columns].mean().reset_index()

        fig = make_subplots(rows=1, cols=1)

        for sensor in selected_sensors:
            for column, label in zip(columns, data_labels):
                sensor_data = filtered_df[filtered_df['Sensor'] == sensor]
                if sensor_data.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sensor_data['Time'],
                    y=sensor_data[column],
                    mode='lines+markers' + ('+text' if show_text else ''),
                    name=f'Sensor {sensor} - {label}',
                    marker=dict(size=marker_size),
                    text=sensor_data[column].apply(lambda x: f"{x:.2f}") if show_text else None,
                    textposition="top center"
                ))

        fig.update_layout(
            title=title or f"{data_type} Plot",
            title_x=0.5,
            title_xanchor='center',
            title_yanchor='middle',
            title_font_size=25,
            title_font_color="black",
            title_font_family="Arial",
            yaxis=dict(
                title=data_type,
                tick0=0,
                dtick=yaxis_major_tick,  # Major ticks interval을 0.5로 설정 -> Major ticks interval 0.5 로 기본 설정하고, 슬라이더로 바꿀 수 있게끔 설정함.
                range=[yaxis_min_value, yaxis_max_value],  # Y축의 최대값 설정
            ),
            showlegend=True,
            legend=dict(
                # x=0,  # 범례를 화면의 좌측으로 이동
                # y=1,
                # xanchor='left',
                # yanchor='top',
                font=dict(
                    size=10  # 범례의 폰트를 줄이기 (예: 10px)
                )
            )
        )
        # fig.update_traces(mode = "markers+lines")
        fig.update_traces(mode="markers+lines")

        if show_text:
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside',
                              textfont=dict(size=60))  # 폰트 크기를 60로 설정

        fig.update_xaxes(
            tickformat='%H:%M:%S',
            dtick=xaxis_major_tick * 1000,  # 1초 간격으로 틱
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showspikes=True, spikesnap="cursor", spikemode="across"
        )
        fig.update_yaxes(showspikes=True, spikethickness=2)

        fig.update_layout(height=1080, spikedistance=1000, hoverdistance=100) # 그래프 높이 설정 (예: 800px)
        return fig
    except Exception as e:
        st.error(f"Error: {str(e)}")

def plot_bar_data_plotly(df, start_time, end_time, selected_sensors, selected_axes, data_type, threshold, show_text,
                         marker_size, title, start_at_midnight, average, yaxis_major_tick, yaxis_max_value, yaxis_min_value,xaxis_major_tick):
    try:
        if start_time > end_time:
            st.error("Start time must be earlier than end time.")
            return

        if not selected_sensors:
            st.warning("No sensors selected.")
            return

        columns_to_plot = []
        data_labels = []

        # 컬럼 선택 로직
        if 'grms_x' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'grms_{axis.lower()}')
                data_labels.append(f'가속도 RMS {axis}축') #범례 단순화를 위해 주석(Trial and error)
                # data_labels.append(f'{axis}축')

        elif 'X_axis' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'{axis}_axis')
                # data_labels.append(f'가속도 {axis}축')
                data_labels.append(f'{axis}축')

        filtered_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time) &
                           (df['Sensor'].isin(selected_sensors)) &
                           (df[columns_to_plot].max(axis=1) >= threshold)]

        if start_at_midnight:
            min_time = filtered_data['Time'].min()
            min_date = min_time.normalize()  # 자정 시간으로 설정
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].apply(lambda x: min_date + (x - min_time))
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # start_time = filtered_data['Time'].min()
            # end_time = filtered_data['Time'].max()

        if average:
            filtered_data = filtered_data.groupby(['Time', 'Sensor'])[columns_to_plot].mean().reset_index()

        long_format = filtered_data.melt(id_vars=['Time', 'Sensor'], value_vars=columns_to_plot, var_name='Axis',
                                         value_name='Value') #범례 바꾸기 위해 이전꺼 주석처리(Trial and error)
        # long_format = filtered_data.melt(id_vars=['Time', 'Sensor'], value_vars=columns_to_plot, var_name='Axis',
        #                                  value_name='Value')

        # long_format['Color'] = 'Sensor ' + long_format['Sensor'].apply(str) + ' - ' + long_format['Axis'].apply(
        #     lambda x: x.split('_')[1].upper() + '축')

        # 주석 처리 내용 : 범례 확인하니까 A 센서 대상으로 AXIS 축이라고 떠서 임시로 블락처리 해둠(240909)

        long_format['Color'] = 'Sensor ' + long_format['Sensor'].astype(str) + '-' + long_format['Axis']

        fig = px.bar(long_format, x='Time', y='Value', color='Color', barmode='group',
                     labels={'Value': 'Measurement Value'},
                     title=title or 'Measurement Values Over Time by Sensor and Axis')

        if show_text:
            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside',
                              textfont=dict(size=60))  # 폰트 크기를 60로 설정

        fig.update_layout(
            title=title or 'Measurement Values Over Time',
            title_x=0.5,
            title_xanchor='center',
            title_yanchor='middle',
            title_font_size=25,
            title_font_color="black",
            title_font_family="Arial",
            bargap=0.01 * marker_size,
            bargroupgap=0.1,
            xaxis=dict(title='Time', tickformat='%H:%M:%S'
                       #,range = [start_time,end_time]
                        ),
            yaxis=dict(
                title=data_type,
                tick0=0,
                dtick=yaxis_major_tick,  # Major ticks interval
                range=[yaxis_min_value, yaxis_max_value],  # Y축의 최대값 설정
            )
        )
        fig.update_xaxes(
            tickformat='%H:%M:%S',
            dtick=xaxis_major_tick*1000,  # 1초 간격으로 틱
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showspikes = True
        )
        fig.update_yaxes(showspikes=True)
        fig.update_layout(height=1080, spikedistance=1000, hoverdistance=100)  # 그래프 높이 설정 (예: 800px)
        return fig
    except Exception as e:
        st.error(f"Error: {str(e)}")

def plot_range_bar_data(df, start_time, end_time, selected_sensors, selected_axes, threshold, show_text,
                         marker_size, title, start_at_midnight, average, yaxis_major_tick, yaxis_max_value, yaxis_min_value,xaxis_major_tick):
    try:
        if start_time > end_time:
            st.error("Start time must be earlier than end time.")
            return

        if not selected_sensors:
            st.warning("No sensors selected.")
            return

        columns_to_plot = []
        data_labels = []

        # 컬럼 선택 로직
        if 'grms_x' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'grms_{axis.lower()}')
                data_labels.append(f'{axis}축')

        elif 'X_axis' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'{axis}_axis')
                data_labels.append(f'{axis}축')

        # 필터링된 데이터
        filtered_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time) &
                           (df['Sensor'].isin(selected_sensors)) &
                           (df[columns_to_plot].max(axis=1) >= threshold)]

        if start_at_midnight:
            min_time = filtered_data['Time'].min()
            min_date = min_time.normalize()  # 자정 시간으로 설정
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].apply(lambda x: min_date + (x - min_time))
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        if average:
            filtered_data = filtered_data.groupby(['Time', 'Sensor'])[columns_to_plot].mean().reset_index()

        # 시간 단위로 그룹화하고 High(최고값), Low(최저값) 계산
        range_data = []

        for sensor in selected_sensors:
            sensor_data = filtered_data[filtered_data['Sensor'] == sensor]
            if sensor_data.empty:
                continue

            for axis, column in zip(selected_axes, columns_to_plot):
                # 시간 단위로 그룹화하여 각 1초마다 최고값과 최저값 계산
                high_values = sensor_data.groupby(sensor_data['Time'].dt.floor('S'))[column].max()
                low_values = sensor_data.groupby(sensor_data['Time'].dt.floor('S'))[column].min()

                # 범위 막대 차트를 그릴 데이터 추가
                range_data.append(go.Bar(
                    x=high_values.index,
                    y=high_values - low_values,  # 범위의 차이
                    base=low_values,  # 최소값에서 시작
                    name=f'Sensor {sensor} - {axis}축',
                    hovertext=[f'High: {high:.2f}, Low: {low:.2f}' for high, low in zip(high_values, low_values)],
                    hoverinfo='text',
                    marker=dict(color='lightblue', line=dict(width=2, color='blue'))
                ))

        # 그래프 객체 생성
        fig = go.Figure(data=range_data)

        # 레이아웃 설정
        fig.update_layout(
            title=title or 'Range Bar Chart by Sensor and Axis',
            title_x=0.5,
            title_xanchor='center',
            title_yanchor='middle',
            title_font_size=25,
            title_font_color="black",
            title_font_family="Arial",
            xaxis=dict(title='Time', tickformat='%H:%M:%S'),
            yaxis=dict(
                title='범위 막대 그래프 테스트',
                tick0=0,
                dtick=yaxis_major_tick,  # Major ticks interval
                range=[yaxis_min_value, yaxis_max_value],  # Y축의 최소/최대값 설정
            )
        )

        fig.update_xaxes(
            tickformat='%H:%M:%S',
            dtick=xaxis_major_tick*1000,  # 1초 간격으로 틱
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showspikes=True
        )
        fig.update_yaxes(showspikes=True)
        fig.update_layout(height=1080, spikedistance=1000, hoverdistance=100)  # 그래프 높이 설정 (예: 800px)

        return fig
    except Exception as e:
        st.error(f"Error: {str(e)}")


def plot_heatmap_plotly(df, start_time, end_time, selected_sensors, selected_axes, data_type, threshold, title,
                        start_at_midnight, average):
    try:
        if start_time > end_time:
            st.error("Start time must be earlier than end time.")
            return

        if not selected_sensors:
            st.warning("No sensors selected.")
            return

        columns_to_plot = []

        # 컬럼 선택 로직
        if 'grms_x' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'grms_{axis.lower()}')
        elif 'X_axis' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'{axis}_axis')

        # 데이터 필터링
        filtered_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time) &
                           (df['Sensor'].isin(selected_sensors))]

        if start_at_midnight:
            min_time = filtered_data['Time'].min()
            min_date = min_time.normalize()  # 자정 시간으로 설정
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].apply(lambda x: min_date + (x - min_time))
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        if average:
            filtered_data = filtered_data.groupby(['Time', 'Sensor'])[columns_to_plot].mean().reset_index()

        # 데이터를 Long Format으로 변환하여 사용
        long_format = filtered_data.melt(id_vars=['Time', 'Sensor'], value_vars=columns_to_plot, var_name='Axis',
                                         value_name='Value')
        long_format['Sensor_Axis'] = 'Sensor ' + long_format['Sensor'].astype(str) + ' - ' + long_format['Axis']

        # 히트맵 생성 및 값 표시 추가
        fig = go.Figure(data=go.Heatmap(
            z=long_format['Value'],
            x=long_format['Time'],
            y=long_format['Sensor_Axis'],
            colorscale='Viridis',  # 색상 스케일
            colorbar=dict(title='Activation Level'),  # 색상바 제목
            text=long_format['Value'].round(2),  # 히트맵 각 칸에 표시될 값
            hoverinfo="text",  # 호버시 표시되는 정보
            showscale=True
        ))

        # 각 칸에 값 표시
        fig.update_traces(texttemplate='%{text}', textfont={'size': 12})

        # 레이아웃 설정
        fig.update_layout(
            title=title or 'Sensor Activation Heatmap',
            title_x=0.5,
            title_xanchor='center',
            title_yanchor='middle',
            title_font_size=25,
            title_font_color="black",
            title_font_family="Arial",
            xaxis=dict(title='Time', tickformat='%H:%M:%S'),
            yaxis=dict(
                title='Sensor - Axis',
            ),
            showlegend=True
        )

        fig.update_layout(height=1080)  # 그래프 높이 설정

        return fig

    except Exception as e:
        st.error(f"Error: {str(e)}")


def plot_3d_scatter_plotly(df, start_time, end_time, selected_sensors, selected_axes, data_type, threshold, title,
                           start_at_midnight, average):
    try:
        # 시간 범위와 선택된 센서에 대한 기본 검증
        if start_time > end_time:
            st.error("Start time must be earlier than end time.")
            return

        if not selected_sensors:
            st.warning("No sensors selected.")
            return

        columns_to_plot = []

        # 컬럼 선택 로직
        if 'grms_x' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'grms_{axis.lower()}')
        elif 'X_axis' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'{axis}_axis')

        # 데이터 필터링
        filtered_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time) &
                           (df['Sensor'].isin(selected_sensors))]

        # 자정 시간으로 시작하는 옵션 처리
        if start_at_midnight:
            min_time = filtered_data['Time'].min()
            min_date = min_time.normalize()  # 자정 시간으로 설정
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].apply(lambda x: min_date + (x - min_time))
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # 동일 시간 데이터의 평균을 계산하는 옵션 처리
        if average:
            filtered_data = filtered_data.groupby(['Time', 'Sensor'])[columns_to_plot].mean().reset_index()

        # 데이터를 Long Format으로 변환하여 사용
        long_format = filtered_data.melt(id_vars=['Time', 'Sensor'], value_vars=columns_to_plot, var_name='Axis',
                                         value_name='Value')
        long_format['Sensor_Axis'] = 'Sensor ' + long_format['Sensor'].astype(str) + ' - ' + long_format['Axis']

        # 3D 스캐터 플롯 생성
        fig = go.Figure(data=[go.Surface(
            x=long_format['Time'],
            y=long_format['Sensor_Axis'],
            z=long_format['Value'],
            # mode='markers',
            marker=dict(
                size=5,
                color=long_format['Value'],  # Z축 값을 색상으로 사용
                colorscale='Viridis',  # Z축 값에 따른 색상 변화
                colorbar=dict(title='Value'),
                opacity=0.8
            )
        )])

        # 레이아웃 설정
        fig.update_layout(
            title=title or '3D Sensor Data Scatter Plot',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Sensor - Axis',
                zaxis_title='Value'
            ),
            title_x=0.5,
            title_xanchor='center',
            title_yanchor='middle',
            title_font_size=25,
            title_font_color="black",
            title_font_family="Arial",
        )

        return fig

    except Exception as e:
        st.error(f"Error: {str(e)}")

def plot_surface_plotly(df, start_time, end_time, selected_sensors, selected_axes, data_type, threshold, title,
                        start_at_midnight, average):
    try:
        # 시간 범위와 선택된 센서에 대한 기본 검증
        if start_time > end_time:
            st.error("Start time must be earlier than end time.")
            return

        if not selected_sensors:
            st.warning("No sensors selected.")
            return

        columns_to_plot = []

        # 컬럼 선택 로직
        if 'grms_x' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'grms_{axis.lower()}')
        elif 'X_axis' in df.columns:
            for axis in selected_axes:
                columns_to_plot.append(f'{axis}_axis')

        # 데이터 필터링
        filtered_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time) &
                           (df['Sensor'].isin(selected_sensors))]

        # 자정 시간으로 시작하는 옵션 처리
        if start_at_midnight:
            min_time = filtered_data['Time'].min()
            min_date = min_time.normalize()  # 자정 시간으로 설정
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].apply(lambda x: min_date + (x - min_time))
            filtered_data.loc[:, 'Time'] = filtered_data['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # 동일 시간 데이터의 평균을 계산하는 옵션 처리
        if average:
            filtered_data = filtered_data.groupby(['Time', 'Sensor'])[columns_to_plot].mean().reset_index()

        # 데이터를 Long Format으로 변환하여 사용
        long_format = filtered_data.melt(id_vars=['Time', 'Sensor'], value_vars=columns_to_plot, var_name='Axis',
                                         value_name='Value')
        long_format['Sensor_Axis'] = 'Sensor ' + long_format['Sensor'].astype(str) + ' - ' + long_format['Axis']

        # Surface 3D 플롯을 위한 데이터 준비
        x = pd.to_datetime(long_format['Time']).astype(np.int64) // 10**9  # 시간 데이터를 숫자형으로 변환
        y = long_format['Sensor_Axis']
        z = long_format['Value'].values.reshape((len(selected_sensors) * len(selected_axes), -1))

        # Surface 플롯 생성
        fig = go.Figure(data=[go.Surface(
            z=z,
            x=x,
            y=np.array(y),
            colorscale='Viridis',
            colorbar=dict(title='Value')
        )])

        # 레이아웃 설정
        fig.update_layout(
            title=title or '3D Sensor Data Surface Plot',
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Sensor - Axis',
                zaxis_title='Value'
            ),
            title_x=0.5,
            title_xanchor='center',
            title_yanchor='middle',
            title_font_size=25,
            title_font_color="black",
            title_font_family="Arial",
        )

        return fig

    except Exception as e:
        st.error(f"Error: {str(e)}")


st.set_page_config(layout="wide")  # 전체 화면 모드 설정




# Streamlit 인터페이스
st.title("Vibration Sensor Acceleration(RMS, RAW) Viewer (BroadSens 사 SVT V, A Series )")

# 사이드바에 UI 요소 배치
with st.sidebar:
    st.write(f"\n**Last Updated at 2024-11-14 by QSHE Team Eojin LEE**")
    st.write(f"This program is the property of 현대무벡스(주) 품질환경안전팀")
    st.write(f"\n Unauthorized distribution is prohibited. \n")
    # st.write(f"\n 센서 번호 맵핑 (통계 Export 시 SRM 기준 기본값) \n")
    # st.write(f"166: HP Guide roller L / 167: HP Guide roller R / 192: HP Wheel / 193: OP Guide roller L / 194: OP Guide roller R / 195: OP Wheel / 247: Sensor Bracket / 248: 기상반 PCB / 249: Carriage / 250: Mast")
    uploaded_file = st.file_uploader("Step 1. Select File", type=["xlsx", "xls"])
    if uploaded_file:
        df = load_data(uploaded_file)
        yaxis_label = determine_yaxis_label(df)  # Y축 라벨 결정
        # print(yaxis_label)
        st.success(f"File loaded successfully: {uploaded_file.name}")

        # 시간 범위 설정
        min_time = df['Time'].min()
        max_time = df['Time'].max()
        st.write(f"Available Time Range: {min_time} to {max_time}")

        start_time = st.text_input("Start Time:", value=min_time.strftime('%Y-%m-%d %H:%M:%S'))
        end_time = st.text_input("End Time:", value=max_time.strftime('%Y-%m-%d %H:%M:%S'))

        try:
            start_time = pd.to_datetime(start_time)
            end_time = pd.to_datetime(end_time)
        except ValueError:
            st.error("Invalid date format. Please correct the date.")

        sensors = df['Sensor'].unique()
        selected_sensors = st.multiselect("Step 2. Select Sensors:", sensors, default=sensors.tolist())

        axes = ['X', 'Y', 'Z']
        selected_axes = st.multiselect("Step 3. Select Axes:", axes, default=axes)

        threshold = st.slider("Step 5. Set Threshold:", min_value=0.0, max_value=10.0, value=0.0, step=0.01)

        st.write(f"* * *")

        # 선택된 축에 따라 컬럼 이름 설정
        columns = [f'grms_{axis.lower()}' if 'grms_x' in df.columns else f'{axis}_axis' for axis in selected_axes]

        # 필터링된 데이터
        filtered_data = df[(df['Time'] >= start_time) & (df['Time'] <= end_time) &
                           (df['Sensor'].isin(selected_sensors)) &
                           (df[columns].max(axis=1) >= threshold)]

        # 데이터 포인트 수 계산
        filtered_data_count = filtered_data.shape[0]
        st.markdown(
            f"<p style='color: #007ACC; margin-bottom: 20px;'><strong> Threshold 를 넘는 데이터(측정값)의 개수</strong> : {filtered_data_count} 개 </p>",
            unsafe_allow_html=True)

        # 각 축별 평균 계산
        axis_averages = filtered_data[columns].mean()
        st.markdown("<p style='color: #007ACC; margin-bottom: 10px;'>축(Axis) 당 데이터(측정값) <strong>평균</strong> : </p>",
                    unsafe_allow_html=True)
        for axis, avg in zip(selected_axes, axis_averages):
            st.markdown(f"<p style='color: #007ACC; margin-bottom: -10px;'>{axis}축: {avg:.2f} {str(yaxis_label).split(':')[1]}</p>",
                        unsafe_allow_html=True)

        # 각 축별 최대값 계산
        axis_max_values = filtered_data[columns].max()
        st.markdown("<p style='color: #007ACC; margin-bottom: -10px; margin-top: 20px'>축(Axis) 당 데이터(측정값) <strong>최대값</strong> :</p>",
                    unsafe_allow_html=True)
        for axis, max_val in zip(selected_axes, axis_max_values):
            st.markdown(f"<p style='color: #007ACC; margin-bottom: -10px;'>{axis}-axis: {max_val:.2f} {str(yaxis_label).split(':')[1]}</p>",
                        unsafe_allow_html=True)

        # 엑셀 파일 생성
        excel_file = create_excel_output(sensor_mapping, df, start_time, end_time, threshold)

        # 파일 다운로드 버튼 생성 (통합된 기능)
        st.download_button(label="Export and Download Excel file", icon=":material/lab_profile:", data=excel_file,
                               file_name="Sensor_data_report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",help="다음과 같은 순서로 맵핑됩니다 | 166: HP Guide roller L / 167: HP Guide roller R / 192: HP Wheel / 193: OP Guide roller L / 194: OP Guide roller R / 195: OP Wheel / 247: Sensor Bracket / 248: 기상반 PCB / 249: Carriage / 250: Mast")


        st.write(f"* * *")

        # 초기 상태 설정
        if 'plot_type' not in st.session_state:
            st.session_state.plot_type = '선 그래프'
        if 'plot_button_icon' not in st.session_state:
            st.session_state.plot_button_icon = ":material/earthquake:"

        def plot_type_icon_selector():
            plot_type = st.session_state.plot_type
            if plot_type == "선 그래프":
                st.session_state.plot_button_icon = ":material/earthquake:"
            elif plot_type == '막대 그래프':
                st.session_state.plot_button_icon = ":material/finance:"
            elif plot_type == '캔들 막대 그래프(미완성)':
                st.session_state.plot_button_icon = ":material/candlestick_chart:"
            elif plot_type == '히트맵(미완성)':
                st.session_state.plot_button_icon = ":material/key_visualizer:"
            else:
                st.session_state.plot_button_icon = None

        plot_type = st.selectbox("Plot Type:", ['선 그래프', '막대 그래프', '캔들 막대 그래프(미완성)', '히트맵(미완성)', '3D Scatter(개발중)','3D Surface(개발중)'],key='plot_type',on_change=plot_type_icon_selector)


        marker_size = st.slider("Marker Size:", min_value=1, max_value=20, value=4)

        start_at_midnight = st.checkbox("<옵션 1> 타임스탬프 00:00:00 시작")
        average = st.checkbox("<옵션 2> 동일 시간 데이터 평균값 적용(단위 : 초)")
        show_text = st.checkbox('<옵션 3> 데이터 값 모두 표시 (주의 ! Data 수 다량 사용시 문제 발생할 수 있음)')
        custom_title = st.checkbox("<옵션 4> 그래프 Title 수동 입력")


        title = ""
        if custom_title:
            title = st.text_input("Graph Title:", value="")

        # Y축 눈금 설정 슬라이더
        yaxis_max_value = st.slider("Y-axis Max Value:", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        yaxis_min_value = st.slider("Y-axis Min Value:", min_value=-10, max_value=0, value=0, step=1)
        yaxis_major_tick = st.slider("Y-axis Major Tick Interval:", min_value=0.1, max_value=1.5, value=0.5, step=0.1)

        # 시간 눈금 설정 슬라이더
        xaxis_major_tick = st.slider("X-axis Major Tick (Time) Value | unit : Seconds) : ", min_value=1, max_value=60, value=30, step=1)

        # "Plot Data" 버튼도 사이드바에 배치
        plot_button_clicked = st.button('Plot Data',key='plot_data_button',icon=st.session_state.plot_button_icon)


# 중앙에 그래프 표시
if uploaded_file and plot_button_clicked:
    st.write(f"---------------------------------------------")  # 여백 추가를 원하면 사용
    if plot_type == '선 그래프':
        fig = plot_data_plotly(df, start_time, end_time, selected_sensors, selected_axes, yaxis_label, threshold,
                               show_text, marker_size, title, start_at_midnight, average, yaxis_major_tick, yaxis_max_value, yaxis_min_value,xaxis_major_tick
                               )
    elif plot_type == '막대 그래프':
        fig = plot_bar_data_plotly(df, start_time, end_time, selected_sensors, selected_axes, yaxis_label, threshold,
                                   show_text, marker_size, title, start_at_midnight, average, yaxis_major_tick, yaxis_max_value, yaxis_min_value,xaxis_major_tick
                                   )
    elif plot_type == '캔들 막대 그래프(미완성)':
        fig = plot_range_bar_data(df, start_time, end_time, selected_sensors, selected_axes, threshold, show_text,
                         marker_size, title, start_at_midnight, average, yaxis_major_tick, yaxis_max_value, yaxis_min_value,xaxis_major_tick)

    elif plot_type == '히트맵(미완성)':
        fig = plot_heatmap_plotly(df, start_time, end_time, selected_sensors, selected_axes, yaxis_label, threshold, title,
                              start_at_midnight, average)

    elif plot_type == '3D Scatter(개발중)':
        fig = plot_3d_scatter_plotly(df, start_time, end_time, selected_sensors, selected_axes, yaxis_label, threshold,
                                     title,
                                     start_at_midnight, average)
    elif plot_type == '3D Surface(개발중)':
        fig = plot_surface_plotly(df, start_time, end_time, selected_sensors, selected_axes, yaxis_label, threshold, title,
                              start_at_midnight, average)

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Failed to generate the heatmap.")

    # 그래프를 화면에 꽉 차도록 표시
    st.write(f"* * *")
