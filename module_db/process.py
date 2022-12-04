import os
from flask import request, jsonify, send_file
from __init__ import *
import pandas as pd
import streamlit as st
import MySQLdb

data_con = MySQLdb.connect("127.0.0.1", "root", "", "machine", 3325)
my_cursor = data_con.cursor()

data_con_agg = MySQLdb.connect("127.0.0.1", "root", "", "aggregate", 3325)
my_cursor_agg = data_con_agg.cursor()


def add_data(humidity, temperature, pressure, torque):
    if humidity and temperature:
        sqlquery = f"insert into machine_1(humidity, temperature, pressure, torque) values ({humidity}, {temperature}, {pressure}, {torque})"
        my_cursor.execute(sqlquery)
        try:
            data_con.commit()
            return True
        except Exception as ex:
            print(ex)
    return False


def update_data(ID, humidity, temperature, pressure, torque, date):
    print(ID, humidity, temperature, pressure, torque, date)
    if ID and humidity and temperature and pressure and torque and date:
        my_cursor.execute(
            f" UPDATE machine_1 SET humidity = {humidity}, temperature = {temperature}, pressure = {pressure}, torque = {torque} WHERE ID = {ID}")
        try:
            data_con.commit()
            return True
        except Exception as ex:
            print(ex)
    return False


def update_m1_led_data(Stat):
    print(Stat)
    if Stat:
        my_cursor.execute(f" UPDATE machine_1_status SET Stat = {Stat} WHERE machine_1_status.ID = 0")
        try:
            data_con.commit()
            return True
        except Exception as ex:
            print(ex)
    return False


def delete_data(ID):
    if ID:
        my_cursor.execute(f" DELETE FROM machine_1 WHERE ID = {ID}")
        try:
            data_con.commit()
            return True
        except Exception as ex:
            print(ex)
    return False


def get_all_data():
    my_cursor.execute("SELECT * FROM machine_1")
    my_result = my_cursor.fetchall()
    all_data_list = list(my_result)
    all_data_df = pd.DataFrame(
        all_data_list, columns=['ID', 'humidity', 'temperature', "date"])
    # for x in my_result:
    #     print(x)
    return all_data_df


def add_data_agg(demand_aggregate, production_cost, holding_cost, labor_cost, overtime_cost, avai_labor_hour,
                 avia_over_hour):
    if demand_aggregate and production_cost and holding_cost and labor_cost and overtime_cost and avai_labor_hour and avia_over_hour:
        sqlquery_agg = f"insert into data1(demand_aggregate, production_cost, holding_cost, labor_cost, overtime_cost, avai_labor_hour, avia_over_hour) " \
                       f"values ({demand_aggregate}, {production_cost}, {holding_cost}, {labor_cost}, {overtime_cost}, {avai_labor_hour}, {avia_over_hour})"
        my_cursor_agg.execute(sqlquery_agg)
        try:
            data_con_agg.commit()
            return True
        except Exception as ex:
            print(ex)
    return False


def get_all_data_agg():
    my_cursor_agg.execute("SELECT * FROM data1")
    my_result_agg = my_cursor_agg.fetchall()
    all_data_list_agg = list(my_result_agg)
    all_data_df_agg = pd.DataFrame(
        all_data_list_agg, columns=["ID", "demand", "production_cost", "holding_cost", "labor_cost",
                                    "overtime_cost", "avai_labor_hour", "avia_over_hour", "date"])
    # for x in my_result:
    #     print(x)
    return all_data_df_agg
