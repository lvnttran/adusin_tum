import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import cv2
import tensorflow as tf
import plotly.express as px  # interactive charts

import streamlit as st  # 🎈 data web app development
from streamlit_autorefresh import st_autorefresh

from PIL import Image
import time
import sys
import subprocess
import sqlalchemy

import requests
import base64
import io
import glob
from base64 import decodebytes
from io import BytesIO
import matplotlib.pyplot as plt

from io import StringIO
from pathlib import Path

from detect import detect
import os
import argparse
from PIL import Image

import seaborn as sns
from datetime import datetime

# FUNCTION
from process import *
# from control_chart import *
from surface_detection import *

# from predictive_maintainance import *
# from linear_predict import *
# from aggregate import *
# from test import *


# DATABASE
machine_data = sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3325/machine', pool_recycle=3600)
machine_1_data = pd.read_sql_table("machine_1", machine_data)

machine_1_status_data = sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3325/machine', pool_recycle=3600)
m1_status_data = pd.read_sql_table("machine_1_status", machine_data)

sale_data = pd.read_csv('data_train/clean_data.csv')
sale_data_short = sale_data[["Date", "Weekly_Sales", "IsHoliday", "Temperature", "Fuel_Price", "CPI"]]

maintainance_data = pd.read_csv("data_train/predictive_maintenance.csv")
maintainance_data_short = maintainance_data[
    ["Product ID", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]",
     "Tool wear [min]", "Failure Type"]]

aggregate_data = sqlalchemy.create_engine('mysql+pymysql://root:@localhost:3325/aggregate', pool_recycle=3600)
aggregate_data_short = pd.read_sql_table("data1", aggregate_data)


def get_subdirs(b=''):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':
    st.set_page_config(page_title='@adusin',
                       page_icon='random', layout='wide',
                       initial_sidebar_state="expanded")
    st.title("#SmartFactoryApplication")
    option_db = st.sidebar.selectbox(
        'Which module you need?',
        ('', "Database Module", "Quality Control Module", "Business Module", "Testing"))
    placeholder = st.empty()

    # ---------------------------------------------------------------------- #
    # Database Sensor
    if option_db == "Database Module":
        option_machine = st.sidebar.selectbox('Which database you need?',
                                              ('', "Machine 1", "Machine 2"))
        if option_machine == "Machine 1":
            # Dashboard
            st.subheader("Machine 1 Data Summary")
            hide_dataframe_row_index = """
                        <style>
                        .row_heading.level0 {display:none}
                        .column_heading.level0 {display:none}
                        .blank {display:none}
                        </style>
                        """
            # Inject CSS with Markdown
            st_autorefresh(interval=5 * 1000, key="loading")
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            table = st.dataframe(
                machine_1_data.style.format(subset=['humidity', 'temperature', 'pressure', 'torque'],
                                            formatter="{:.1f}"))
            st_autorefresh(interval=10 * 1000, key="loading machine 1 data")
            # CRUD
            option_crud = st.sidebar.selectbox(
                'You want to ...',
                ('', "Create", "Update", "Delete"))
            if option_crud == "Create":
                with st.form(key="Create"):
                    humidity = st.text_input('Enter current Humidity', 'Type here !!!')
                    temperature = st.text_input('Enter current Temperature', 'Type here !!!')
                    pressure = st.text_input('Enter current Pressure', 'Type here !!!')
                    torque = st.text_input('Enter current Torque', 'Type here !!!')
                    create_button = st.form_submit_button("Create")
                # CONFIG
                if create_button:
                    new_data = add_data(humidity, temperature, pressure, torque)
                    table.dataframe(
                        get_all_data().style.format(subset=['humidity', 'temperature', 'pressure', 'torque'],
                                                    formatter="{:.1f}"))

            elif option_crud == "Update":
                date_of_data = [machine_1_data["date"][1]]
                # date_of_data = ["YYYY-MM-DD HH:MM:SS"]
                for i in machine_1_data["date"]:
                    date_of_data.append(i)
                option_date = st.selectbox(
                    'You want to update ...', date_of_data)
                # st.write(option_date)
                index_value = int(machine_1_data.index[machine_1_data["date"] == option_date].values)
                humidity = st.text_input('Enter new value of Humidity', machine_1_data["humidity"][index_value])
                temperature = st.text_input('Enter new value of Temperature',
                                            machine_1_data["temperature"][index_value])
                pressure = st.text_input('Enter new value of Pressure', machine_1_data["pressure"][index_value])
                torque = st.text_input('Enter new value of Torque', machine_1_data["torque"][index_value])
                # CONFIG
                if st.button('Update'):
                    ID = machine_1_data["ID"][index_value]
                    date = machine_1_data["date"][index_value]
                    update_data = update_data(ID, humidity, temperature, pressure, torque, date)
                    table.dataframe(
                        get_all_data().style.format(subset=['humidity', 'temperature', 'pressure', 'torque'],
                                                    formatter="{:.1f}"))

            elif option_crud == "Delete":
                date_of_data = []
                for i in machine_1_data["date"]:
                    date_of_data.append(i)
                # st.write(date_of_data)
                option_date = st.selectbox(
                    'You want to delete ...', date_of_data)
                # st.write(option_date)
                index_value = int(machine_1_data.index[machine_1_data["date"] == option_date].values)
                if st.button('Delete'):
                    ID = machine_1_data["ID"][index_value]
                    delete_data = delete_data(ID)
                    table.dataframe(
                        get_all_data().style.format(subset=['humidity', 'temperature', 'pressure', 'torque'],
                                                    formatter="{:.1f}"))
        elif option_machine == "Machine 2":
            pass

    # ---------------------------------------------------------------------- #
    elif option_db == "Quality Control Module":
        option_qc = st.sidebar.selectbox(
            'Which function you need?',
            ('', "Control Chart", "Surface Detection", "Predictive Maintainance"))
        if option_qc == "Control Chart":
            option_machine = st.sidebar.selectbox('Please choose machine for Control Chart',
                                                  ('', "Machine 1", "Machine 2"))
            if option_machine == "Machine 1":
                # RUN Control Chart Machine 1
                subprocess.run([f"{sys.executable}", "control_chart.py"])

                # LED Machine 1 Status
                m1_current_value = m1_status_data["Stat"][0]
                st_autorefresh(interval=2 * 1000, key="loading machine 1 status")
                if m1_current_value == "1":
                    st.write(f"Machine 1 current status: ON")
                    st.write(m1_current_value)
                else:
                    st.write(f"Machine 1 current status: OFF")
                    st.write(m1_current_value)
                st.write("You want Machine 1 to ..")
                if st.button('ON'):
                    update_m1_led_data(1)
                if st.button('OFF'):
                    update_m1_led_data(2)
                # st.write(m1_status_data["Stat"][0])
            elif option_machine == "Machine 2":
                pass

        elif option_qc == "Surface Detection":
            option_detect = st.sidebar.selectbox(
                'Please choose material to detection',
                ('', "Ceramic GACNN", "Fabric Yolo", "Fabric Ver 1", "Ceramic Ver 1"))
            if option_detect == "Ceramic GACNN":
                source = ("Image", "Video", "Real-time")
                source_index = st.sidebar.selectbox("Input", range(
                    len(source)), format_func=lambda x: source[x])
                if source_index == 0:
                    uploaded_file = st.sidebar.file_uploader(
                        "Upload Image", type=['png', 'jpeg', 'jpg'])
                    if uploaded_file is not None:
                        is_valid = True
                        with st.spinner(text='Loading...'):
                            st.image(uploaded_file)
                            picture = Image.open(uploaded_file)
                            picture.save(f'fabric_yolov7/fabric/stdata/images_gacnn/{uploaded_file.name}')
                            if st.button('Detection'):
                                st.write(f"{uploaded_file.name}: Blob Error")
                elif source_index == 1:
                    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
                    if uploaded_file is not None:
                        is_valid = True
                        with st.spinner(text='Loading...'):
                            st.sidebar.video(uploaded_file)
                            with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            opt.source = f'fabric_yolov7/fabric/stdata/videos/{uploaded_file.name}'
                            if st.button('Detection'):
                                detect(opt)
                                with st.spinner(text='Preparing Video'):
                                    for vid in os.listdir(get_detection_folder()):
                                        st.video(str(Path(f'{get_detection_folder()}') / vid))

                                    st.balloons()

                elif source_index == 2:
                    run = st.checkbox('Real Time Detection')
                    FRAME_WINDOW = st.image([])
                    cap = cv2.VideoCapture(0)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    while run:
                        ret, frame = cap.read()
                        cam_array = np.array(frame)
                        opt.source = "0"
                        camera_surface = detect(opt)
                        frame = cv2.cvtColor(camera_surface, cv2.COLOR_BGR2RGB)
                        FRAME_WINDOW.image(frame)

            elif option_detect == "Fabric Yolo":
                try:
                    parser = argparse.ArgumentParser()
                    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
                    parser.add_argument('--source', type=str, default="fabric_yolov7/fabric/stdata/images",
                                        help='source')  # file/folder, 0 for webcam
                    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
                    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
                    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
                    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
                    parser.add_argument('--view-img', action='store_true', help='display results')
                    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
                    parser.add_argument('--save-conf', action='store_true',
                                        help='save confidences in --save-txt labels')
                    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
                    parser.add_argument('--classes', nargs='+', type=int,
                                        help='filter by class: --class 0, or --class 0 2 3')
                    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
                    parser.add_argument('--augment', action='store_true', help='augmented inference')
                    parser.add_argument('--update', action='store_true', help='update all models')
                    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
                    parser.add_argument('--name', default='exp', help='save results to project/name')
                    parser.add_argument('--exist-ok', action='store_true',
                                        help='existing project/name ok, do not increment')
                    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
                    opt = parser.parse_args()
                    print(opt)

                    source = ("Image", "Video", "Real-time")
                    source_index = st.sidebar.selectbox("Input", range(
                        len(source)), format_func=lambda x: source[x])

                    if source_index == 0:
                        uploaded_file = st.sidebar.file_uploader(
                            "Upload Image", type=['png', 'jpeg', 'jpg'])
                        if uploaded_file is not None:
                            is_valid = True
                            with st.spinner(text='Loading...'):
                                st.sidebar.image(uploaded_file)
                                picture = Image.open(uploaded_file)
                                picture = picture.save(f'fabric_yolov7/fabric/stdata/images/{uploaded_file.name}')
                                opt.source = f'fabric_yolov7/fabric/stdata/images/{uploaded_file.name}'
                                if st.button('Detection'):
                                    detect(opt)
                                    with st.spinner(text='Preparing Images'):
                                        for img in os.listdir(get_detection_folder()):
                                            st.image(str(Path(f'{get_detection_folder()}') / img))

                                        st.balloons()

                    elif source_index == 1:
                        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4'])
                        if uploaded_file is not None:
                            is_valid = True
                            with st.spinner(text='Loading...'):
                                st.sidebar.video(uploaded_file)
                                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                opt.source = f'fabric_yolov7/fabric/stdata/videos/{uploaded_file.name}'
                                if st.button('Detection'):
                                    detect(opt)
                                    with st.spinner(text='Preparing Video'):
                                        for vid in os.listdir(get_detection_folder()):
                                            st.video(str(Path(f'{get_detection_folder()}') / vid))

                                        st.balloons()

                    elif source_index == 2:
                        run = st.checkbox('Real Time Detection')
                        FRAME_WINDOW = st.image([])
                        cap = cv2.VideoCapture(0)
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                        while run:
                            ret, frame = cap.read()
                            cam_array = np.array(frame)
                            opt.source = "0"
                            camera_surface = detect(opt)
                            frame = cv2.cvtColor(camera_surface, cv2.COLOR_BGR2RGB)
                            FRAME_WINDOW.image(frame)

                except PermissionError:
                    pass

            elif option_detect == "Fabric Ver 1":
                # Add in location to select image.

                st.write('#### Select an image to upload.')
                uploaded_file = st.file_uploader('',
                                                 type=['png', 'jpg', 'jpeg'],
                                                 accept_multiple_files=False)

                ## Add in sliders.
                confidence_threshold = st.slider(
                    'Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?',
                    0.0,
                    1.0, 0.5, 0.01)
                overlap_threshold = st.slider(
                    'Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?',
                    0.0,
                    1.0, 0.5, 0.01)

                ##########
                ##### Set up main app.
                ##########

                ## Pull in default image or user-selected image.
                if uploaded_file is None:
                    # Default image.
                    # url = 'https://github.com/matthewbrems/streamlit-bccd/blob/master/BCCD_sample_images/BloodImage_00038_jpg.rf.6551ec67098bc650dd650def4e8a8e98.jpg?raw=true'
                    url = "https://www.cottoninc.com/wp-content/uploads/2017/05/0183_Hole.jpg"
                    image = Image.open(requests.get(url, stream=True).raw)

                else:
                    # User-selected image.
                    image = Image.open(uploaded_file)

                ## Subtitle.
                st.write('### Inferenced Image')

                # Convert to JPEG Buffer.
                buffered = io.BytesIO()
                image.save(buffered, quality=90, format='JPEG')

                # Base 64 encode.
                img_str = base64.b64encode(buffered.getvalue())
                img_str = img_str.decode('ascii')

                ## Construct the URL to retrieve image.
                upload_url = ''.join([
                    'https://detect.roboflow.com/farbic_error/1?api_key=uOzGiXWuhPptr66lLrEp',
                    '&format=image',
                    f'&overlap={overlap_threshold * 100}',
                    f'&confidence={confidence_threshold * 100}',
                    '&stroke=2',
                    '&labels=True'
                ])

                ## POST to the API.
                r = requests.post(upload_url,
                                  data=img_str,
                                  headers={
                                      'Content-Type': 'application/x-www-form-urlencoded'
                                  })

                image = Image.open(BytesIO(r.content))

                # Convert to JPEG Buffer.
                buffered = io.BytesIO()
                image.save(buffered, quality=90, format='JPEG')

                # Display image.
                st.image(image, width=500)

                ## Construct the URL to retrieve JSON.
                upload_url = ''.join([
                    'https://detect.roboflow.com/farbic_error/1?api_key=uOzGiXWuhPptr66lLrEp'
                ])

                ## POST to the API.
                r = requests.post(upload_url,
                                  data=img_str,
                                  headers={
                                      'Content-Type': 'application/x-www-form-urlencoded'
                                  })

                ## Save the JSON.
                output_dict = r.json()

                ## Generate list of confidences.
                confidences = [box['confidence'] for box in output_dict['predictions']]

                ## Display the JSON in main app.
                st.write('### JSON Output')
                st.write(r.json())

                ## Summary statistics section in main app.
                st.write('### Summary Statistics')
                st.write(f'Number of Bounding Boxes (ignoring overlap thresholds): {len(confidences)}')
                st.write(f'Average Confidence Level of Bounding Boxes: {(np.round(np.mean(confidences), 4))}')

                ## Histogram in main app.
                st.write('### Histogram of Confidence Levels')
                fig, ax = plt.subplots()
                ax.hist(confidences, bins=10, range=(0.0, 1.0))
                st.pyplot(fig)

            elif option_detect == "Ceramic Tile":
                img_file_buffer = st.file_uploader("Upload an image")
                if img_file_buffer is not None:
                    image = Image.open(img_file_buffer)

                    img_array = np.array(image)  # if you want to pass it to OpenCV
                    image_surface = detect_surface(img_array, model_surface)
                    st.image(image_surface, caption="Image to detect", use_column_width=True)
                # Real time detect
                run = st.checkbox('Real Time Detection')
                FRAME_WINDOW = st.image([])
                cap = cv2.VideoCapture(0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                while run:
                    ret, frame = cap.read()
                    cam_array = np.array(frame)
                    camera_surface = detect_surface(cam_array, model_surface)
                    frame = cv2.cvtColor(camera_surface, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(frame)

        elif option_qc == "Predictive Maintainance":
            st.subheader("Machine Data")
            hide_dataframe_row_index = """
                                <style>
                                .row_heading.level0 {display:none}
                                .blank {display:none}
                                </style>
                                """
            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            table = st.dataframe(maintainance_data_short.iloc[1:100])
            with st.form(key="Predictive_Maintainance"):
                processtemp_input = st.text_input("Enter the Process temperature in K: ")
                torque_input = st.text_input("Enter the Torque in Nm: ")
                toolwear_input = st.text_input("Enter the Tool wear in mins: ")
                predict_button_2 = st.form_submit_button("Predict")
            # CONFIG
            if predict_button_2:
                maintenance_predict = MPM_model_decision(np.array([[processtemp_input, torque_input, toolwear_input]]))
                st.text(f"{maintenance_predict}")

    # ---------------------------------------------------------------------- #
    elif option_db == "Business Module":
        option_business = st.sidebar.selectbox(
            'Which function you need?',
            ('', "Demand Forecasting", "Aggregate Planning"))
        if option_business == "Demand Forecasting":
            st.subheader("Sale Data")
            hide_dataframe_row_index = """
                        <style>
                        .row_heading.level0 {display:none}
                        .blank {display:none}
                        </style>
                        """
            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            table = st.dataframe(
                sale_data_short.iloc[:100].style.format(subset=["Weekly_Sales", "Temperature", "Fuel_Price"],
                                                        formatter="{:.1f}"))
            with st.form(key="Predict_Demand"):
                fuel_index = st.text_input("Enter Fuel_Price: ")
                predict_button = st.form_submit_button("Predict")
            # CONFIG
            if predict_button:
                sale_predict = predict(float(fuel_index), weight, bias)
                st.text(f"Weekly sale is: {round(sale_predict, 2)}")

        elif option_business == "Aggregate Planning":
            # Dashboard
            st.subheader("Aggregate Data")
            hide_dataframe_row_index = """
                        <style>
                        .row_heading.level0 {display:none}
                        .blank {display:none}
                        </style>
                        """
            # Inject CSS with Markdown
            st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
            table = st.dataframe(
                aggregate_data_short.style.format(subset=["demand", "production_cost", "holding_cost", "labor_cost",
                                                          "overtime_cost", "avai_labor_hour", "avai_over_hour"],
                                                  formatter="{:.1f}"))
            # CRUD
            option_crud = st.selectbox(
                'You want to ...',
                ('', "Create", "Update", "Aggregate Production Planning"))
            if option_crud == "Create":
                with st.form(key="Create"):
                    demand_aggregate = st.text_input('Enter current Demand', 'Type here !!!')
                    production_cost = st.text_input('Enter current Production Cost', 'Type here !!!')
                    holding_cost = st.text_input('Enter current Holding Cost', 'Type here !!!')
                    labor_cost = st.text_input('Enter current Labor Cost', 'Type here !!!')
                    overtime_cost = st.text_input('Enter current Overtime Cost', 'Type here !!!')
                    avai_labor_hour = st.text_input('Enter current Available Labor Hour', 'Type here !!!')
                    avai_over_hour = st.text_input('Enter current Available Overtime Hour', 'Type here !!!')
                    create_button_agg = st.form_submit_button("Create")
                # CONFIG
                if create_button_agg:
                    new_data_agg = add_data_agg(demand_aggregate, production_cost, holding_cost, labor_cost,
                                                overtime_cost,
                                                avai_labor_hour, avai_over_hour)
                    table.dataframe(
                        get_all_data_agg().style.format(
                            subset=["demand", "production_cost", "holding_cost", "labor_cost",
                                    "overtime_cost", "avai_labor_hour", "avai_over_hour"],
                            formatter="{:.1f}"))
            elif option_crud == "Update":
                date_of_data = [data["date"][5]]
                # date_of_data = ["YYYY-MM-DD HH:MM:SS"]
                for i in data["date"]:
                    date_of_data.append(i)
                option_date = st.selectbox(
                    'You want to update ...', date_of_data)
                # st.write(option_date)
                index_value = int(data.index[data["date"] == option_date].values)
                humidity = st.text_input('Enter new value of Humidity', data["humidity"][index_value])
                temperature = st.text_input('Enter new value of Temperature', data["temperature"][index_value])
                # CONFIG
                if st.button('Update'):
                    ID = data["ID"][index_value]
                    date = data["date"][index_value]
                    update_data = update_data(ID, humidity, temperature, date)
                    table.dataframe(get_all_data().style.format(subset=['humidity', 'temperature'], formatter="{:.1f}"))
            elif option_crud == "Aggregate Production Planning":
                with st.form(key="Aggregate_Production_Planning"):
                    predict_button_3 = st.form_submit_button("Planning")
                # CONFIG
                if predict_button_3:
                    st.text(f"Total Production Plan Cost = {value(prob.objective)}")

    # ---------------------------------------------------------------------- #
    elif option_db == "Testing":
        pass
