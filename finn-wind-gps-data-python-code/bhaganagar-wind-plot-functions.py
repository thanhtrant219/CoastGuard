import os
import io
import re
from pathlib import Path
import datetime as dt
import shutil
import math
import cmath
from functools import wraps
from operator import itemgetter

import pandas as pd
import numpy as np
import matplotlib as mtpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.io.img_tiles as img_tiles
import geopy.distance as distance

import geopandas
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from env import MAPBOX_USER, MAPBOX_TOKEN, MAPBOX_MAP_ID

# the argument dictionary for the wind data, mostly for loading the data from
# the CSV files
wind_args = {"x_cols": {"time": {"name": "infer",
                             "unit": "CTZ"}},
             "y_cols": {"wind_x_velocity": {"name": "infer",
                                       "unit": "m/s"},
                        "wind_y_velocity": {"name": "infer",
                                       "unit": "m/s"},
                        "wind_z_velocity": {"name": "infer",
                                       "unit": "m/s"},
                        "wind_speed": {"name": "infer",
                                       "unit": "m/s"}},
         "header_row": None,
         "col_names": ["time", "wind_x_velocity", "wind_y_velocity", "wind_z_velocity",
                       "temperature"],
         # "use_cols": [1, 2, 3, 5],
         "use_cols": [0, 3, 4, 5, 7],
         "output": "show"}

# the argument dictionary for loading the GPS data
gps_args = {"filename_settings": {"base_headline.csv": {"col_names": ["time", "tow", "heading"],
                                                           "use_cols": [1, 5, 6]},
                                  "vel_ned.csv": {"col_names": ["n", "e", "d",
                                                                "h_accuracy", 
                                                                "v_accuracy"],
                                                  "use_cols": [6, 7, 8, 9, 10]},
                                  "pos_llh.csv": {"col_names": ["latitude", "longitude", "height"],
                                                  "use_cols": [6, 7, 8]},
                                  "imu_raw.csv": {"col_names": ["tow_f", "x_acceleration",
                                                                "y_acceleration", "z_acceleration",
                                                                "x_gyr", "y_gyr", "z_gyr"],
                                                  "use_cols": [6, 7, 8, 9, 10, 11, 12]}},
        "header_row": None,
        "skip_rows": [0],
        "x_cols": {"time": {"name": "infer", "unit": "CTZ"},
                   "id": {"name": "infer", "unit": ""}},
        "y_cols": {"x_acceleration": {"name": "infer", "unit": "m/s^2"},
                   "y_acceleration": {"name": "infer", "unit": "m/s^2"},
                   "z_acceleration": {"name": "infer", "unit": "m/s^2"},
                   "longitude": {"name": "infer", "unit": "Degrees"},
                   "latitude": {"name": "infer", "unit": "Degrees"}}}


def main():
    # directories
    base_dir = ENTER YOUR DATA DIR HERE
    
    # these are the folders inside that directory to load data from
    dir_names = [
            "Galveston3_Part1",
            "Galveston3_Part2"
            ]
    project_directories = [os.path.join(base_dir, dir_name) for dir_name in dir_names]

    # drops columns for saving the CSV
    cols_to_drop = [
            "level_0",
            "index",

            "tow",
            "tow_f",
            "height",
            "wind_z_velocity",
            "x_acceleration",
            "y_acceleration",
            "z_acceleration",
            "x_gyr",
            "y_gyr",
            "z_gyr",
            "h_accuracy",
            "v_accuracy",
            "n",
            "e",
            "d",
            "temperature",
            ]

    # loads all of the data
    for base_dir in project_directories:

        # gets the case name
        case_name = os.path.split(base_dir)[1]

        # combines the data
        # combined_df, wind_df, gps_df = main_processor(base_dir)

        # loads the gps data
        gps_data = GPSData(data_locs=base_dir, **gps_args)

        # adds GMT time column
        gps_data.df["GMT_time"] = gps_data.df["time"] + dt.timedelta(hours=12)
        gps_data.df["GMT_time"] = gps_data.df["GMT_time"].dt.time

        # prints the graph
        if (not gps_data.df.empty):

            # plots the graph
            # you can change the plot_every to modify how many points to plot
            # (in this case you are plotting every 100th point)
            gps_data.plot_gps_graph(title=case_name, output="show", plot_every=100, **gps_args)

            # file_name = os.path.split(wind_dir)[1] + "_filtered" + ".csv"

        print(f"\n{case_name}\n")


def main_processor(base_dir):

    # loads the wind file
    wind_file = os.path.join(base_dir, "wind", "wind_data.csv")
    wind_df = fix_one_column_wind(wind_file, **wind_args)
    wind_data = WindData(df=wind_df)

    # loads the gps data
    gps_data = GPSData(data_locs=base_dir, **gps_args)

    # merges the gps and wind data
    merged_data = pd.merge_asof(wind_data.df, gps_data.df, on="time")
    merged_data.dropna(inplace=True)

    # drops columns
    merged_data.drop(cols_to_drop, axis=1, inplace=True)

    # drops headers that have errors
    merged_data = merged_data[merged_data["heading"] != 0]

    # gets the corrected x and y velocities
    merged_data["wind_x_velocity"], merged_data["wind_y_velocity"], \
    merged_data["wind_polar_r"], merged_data["wind_polar_radians"] = \
        GPSData.convert_relative_to_absolute_north(merged_data["wind_x_velocity"],
                                                   merged_data["wind_y_velocity"],
                                                   merged_data["heading"])

    # renames x to u and v to u
    merged_data.rename({"wind_x_velocity": "wind_u_velocity",
                        "wind_y_velocity": "wind_v_velocity",
                        "LO_lat": "latitude",
                        "LO_lon": "longitude",},
                        axis=1, inplace=True)

    # adds GMT time column
    merged_data["GMT_time"] = merged_data["time"] + dt.timedelta(hours=12)
    merged_data["GMT_time"] = merged_data["GMT_time"].dt.time

    # the index range
    # start_time = dt.time(7, 25, 52, microsecond=419 * 1000)
    # end_time = dt.time(8, 27, 57, microsecond=822 * 1000)
    # filtered_by_time = merged_data["GMT_time"].between(start_time, end_time)
    # merged_data = merged_data[filtered_by_time]

    return merged_data, wind_data, gps_data


def fix_one_column_wind(wind_file, **wind_args):
    # reads the base wind file
    df = pd.read_csv(wind_file)

    # gets new column names and values
    col_vals = list(df[df.columns[1]].dropna())

    # creates the string IO object and writes the data with newlines
    col_vals_as_file = io.StringIO()
    col_vals_as_file.write("\n".join(col_vals))
    col_vals_as_file.seek(0)

    # now creates the fixed dataframe
    new_df = pd.read_csv(col_vals_as_file, error_bad_lines=False,
                         usecols=wind_args["use_cols"],
                         names=wind_args["col_names"],
                         # dtype=wind_args["dtype"]
                         )

    numeric_cols = [
            "time",
            "wind_x_velocity",
            "wind_y_velocity",
            "wind_z_velocity",
            "temperature",
            ]
    for numeric_col in numeric_cols:
        new_df[numeric_col] = pd.to_numeric(new_df[numeric_col], errors="coerce")
    new_df.dropna(inplace=True)

    return new_df


# DECORATORS
def _run_inplace_method(func):

    @wraps(func)
    def inner(self, df="infer", inplace=True, *args, **kwargs):
        if not isinstance(df, pd.DataFrame) and df == "infer":
            df = self.df

        if not inplace:
            df = df.copy()

        func(self, df=df, *args, **kwargs)

        if not inplace:
            return df

    return inner


def _run_graphing_method(func):

    @wraps(func)
    def inner(self, df="infer", x_cols="infer", y_cols="infer",
              excluded_pairs=None, output="show", display_lines=True,
              title=str(dt.datetime.now()),
              *args, **kwargs):

        if df == "infer":
            df = self.df

        if x_cols == "infer":
            x_cols = self.x_cols
        if not isinstance(x_cols, dict):
            x_cols = {x_cols[0]: x_cols[1]}

        if y_cols == "infer":
            y_cols = self.y_cols
        if not isinstance(y_cols, dict):
            y_cols = {y_cols[0]: y_cols[1]}

        if excluded_pairs is None:
            excluded_pairs = {}

        # CONVERSION: time column to formattable type
        try:
            df = df.copy()
            df["time"] = pd.to_datetime(df["time"].values.astype("datetime64[D]"))
        except KeyError:
            pass

        func(self, df=df, x_cols=x_cols, y_cols=y_cols,
                    excluded_pairs=excluded_pairs, *args, **kwargs)

        # shows graph
        if output is None:
            pass
        elif output == "show":
            plt.show()
        elif output == "save":
            print("hello! saving")
            plt.savefig(title + ".png")

    return inner


def _run_data_loading_method(func):

    @wraps(func)
    def inner(self, file_path, header_row="infer", col_names="infer", use_cols="infer",
            col_name_transformations="infer",
              *args, **kwargs):


        if header_row == "infer":
            header_row = self.header_row

        if col_names == "infer":
            col_names = self.col_names
        # checks if there are conditional columns based on the filename
        elif isinstance(col_names, dict):
            try:
                col_names = col_names[file_path]
            except KeyError:
                raise FileColumnsNotFoundError

        if use_cols == "infer":
            use_cols = self.use_cols

        if col_name_transformations == "infer":
            col_name_transformations = self.col_name_transformations

        # gets custom filename settings
        try:
            file_name = os.path.split(file_path)[1]
            filename_settings = kwargs["filename_settings"][file_name]

            use_cols = filename_settings.get("use_cols", use_cols)
            col_names = filename_settings.get("col_names", col_names)
            col_name_transformations = filename_settings.get("col_name_transformations",
                                                             col_name_transformations)
        except KeyError:
            pass

        return func(self, file_path, header_row=header_row, col_names=col_names, use_cols=use_cols,
                    col_name_transformations=col_name_transformations, *args,
                    **kwargs)

    return inner

# EXCEPTIONS
class NoTimeColumnError(Exception):
    pass


class NotGPSFileError(Exception):
    pass


class ParametricGraphLengthException(Exception):
    pass


class FileColumnsNotFoundError(Exception):
    pass


# DATA CLASSES
class CustomData():
    def __init__(self, df=None, data_locs=None, x_cols=None, y_cols=None,
                 extension=".csv", use_cols=None, header_row=0, col_names=None,
                 col_name_transformations=None, drop_duplicates=True,
                 load_data=True, **kwargs):

        # basic values
        self.df = df
        self.data_locs = data_locs
        self.x_cols = x_cols
        self.y_cols = y_cols

        # data loading variables
        self.extension = extension
        self.use_cols = use_cols
        self.header_row = header_row
        self.col_names = col_names
        self.col_name_transformations = col_name_transformations

        # turns mutable arg into its correct form
        if self.col_name_transformations is None:
            self.col_name_transformations = {}

        # checks if the data locations var was not a list
        if not isinstance(data_locs, list):
            self.data_locs = [data_locs]

        # checks if the dataframe needs to be loaded
        if isinstance(self.df, pd.DataFrame):
            self.df = self.process_df(df)
        elif self.data_locs and load_data:
            self.df = self.load_data(self.data_locs, **kwargs)

            # drops duplicates
            if drop_duplicates:
                try:
                    self.df.drop_duplicates(subset=["id"], inplace=True)
                except KeyError:
                    pass

    def __dic_name_prettifier(self, dic):
        """
        Internal function that prettifies a dictionary's names

        Parameters
        ----------
        dic (dict)
            Contains a dictionary in the format of {"VAR_NAME": {"name": "PRETTY
            NAME", "unit": "UNIT"}}
            The name key can either be omitted or explicitly labeled as "infer"
            to be automatically determined by this function
        """
        if dic:
            for key, val in dic.items():
                try:
                    if val["name"] != "infer":
                        continue
                except ValueError:
                    pass
                val.update({"name": prettify_var_name(key)})

        return dic

    @property
    def y_cols(self):
        return self.__y_cols

    @y_cols.setter
    def y_cols(self, y_cols):
        self.__y_cols = self.__dic_name_prettifier(y_cols)

    @property
    def x_cols(self):
        return self.__x_cols

    @x_cols.setter
    def x_cols(self, x_cols):
        self.__x_cols = self.__dic_name_prettifier(x_cols)

    def load_data(self, data_locs, **kwargs):

        # loads the data and processes it
        df = self.concatenate_dataframes_from_files(data_locs,
                                                    func=self.process_df,
                                                    inplace=True,
                                                    **kwargs)
        return df

    def process_df(self, df, *args, **kwargs):
        try:
            # tries to convert time from nanoseconds since the epoch to a
            # timedelta
            df["time"] = dt.datetime(1970, 1, 1) + pd.to_timedelta(df["time"])
        except KeyError:
            pass
        return df

    @_run_data_loading_method
    def load_dataframe(self, file_path, header_row="infer", col_names="infer",
                       use_cols="infer", col_name_transformations="infer",
                       skip_rows=None, **kwargs):

        # loads the file
        df = pd.read_csv(
            file_path,
            usecols=use_cols,
            header=header_row,
            names=col_names,
            skiprows=skip_rows
        )

        return df

    def concatenate_dataframes_from_files(self, data_locs, df_func=None,
                                          **kwargs):

        # the list of dataframes
        data_frames = []

        # iterates over data files
        kwargs["extension"] = kwargs.pop("extension", self.extension)
        for df_file in find_files(data_locs, **kwargs):

            # loads the dataframe
            df = self.load_dataframe(df_file, **kwargs)

            # calls any functions that might be specified
            if df_func:
                df = df_func(df=df, file_path=df_file)

            # checks the dataframe is not empty
            if not df.empty:
                data_frames.append(df)

        # concatenates all the dataframes
        concatenated_dfs = pd.concat(data_frames)

        return concatenated_dfs

    def merge_dataframes_from_files(self, data_locs, func=None, **kwargs):

        # turns locations into a list if it isn't already
        if not isinstance(data_locs, list):
            data_locs = [data_locs]

        # the list of dataframes
        df = None

        # iterates over data files
        for df_file in find_files(data_locs, **kwargs):

            # loads the dataframe
            df_part = self.load_dataframe(data_file, **kwargs)

            # calls any functions that might be specified
            if func:
                func(df, df_file)

            df.merge(df_part, **kwargs)

        # concatenates all the dataframes
        return df

    def __get_col_vals(self, col):
        key = col[0]
        name = col[1]["name"]
        unit = col[1]["unit"]
        return key, name, unit

    def __set_title_and_labels(self, axis, x_col, y_col=None, title_note="",y_axis_name=None):
            # variables
            x_key, x_name, x_unit = self.__get_col_vals(x_col)

            # sets the title
            axis.set_xlabel(x_name + " (" + x_unit + ")")

            if y_axis_name:
                axis.set_title((x_name + " versus " + y_axis_name + title_note))
                axis.set_ylabel(y_axis_name)
            else:
                y_key, y_name, y_unit = self.__get_col_vals(y_col)
                axis.set_title((x_name + " versus " + y_key + title_note))
                axis.set_ylabel(y_name + " (" + y_unit + ")")

    def __graphing_x_component(self, num_x_subplots, num_y_subplots):
        fig, axes = plt.subplots(num_x_subplots, num_y_subplots)
        fig.set_figheight(9)
        fig.set_figwidth(16)

        return fig, axes

    def __graphing_y_component(self, df, axis, x_col, y_col, excluded_pairs,
                               fmt=None, combined=False, display_lines=True):
        # sets the x names
        x_key, x_name, x_unit = self.__get_col_vals(x_col)
        # sets the y names
        y_key, y_name, y_unit = self.__get_col_vals(y_col)

        # checks if lines should be removed
        params = []
        if not display_lines:
            params.append(",")

        # checks if this x-y combination shouldn't be graphed
        try:
            if y_key in excluded_pairs[x_key]:
                return
        except KeyError:
            pass

        # plots the data
        if combined:
            y_label = y_name + "(" + y_unit + ")"
            lines = axis.plot(df[x_key], df[y_key], *params, label=y_label)
        else:
            lines = axis.plot(df[x_key], df[y_key], *params)

        # sets the format
        if fmt:
            axis.xaxis.set_major_formatter(fmt)

    def scale_df_by_factor(self, data, scale_factor):
        """
        Helps scale DFs by frequency, meaning if one file was 20 HZ and another
        was 40 HZ it would make them the same length
        """
        if scale_factor != 1:
            # iterates over the new length of the series and creates the new
            # series
            new_length = math.floor(len(data) * scale_factor)
            new_data = pd.DataFrame(columns=data.columns)
            for i in range(new_length):
                old_index = math.floor(i / scale_factor)
                old_data = data.iloc[old_index, :]
                new_data = new_data.append(old_data)
            new_data = new_data.reset_index()
            return new_data
        else:
            return data

    @_run_graphing_method
    def plot_traditional_graphs(self, df="infer", x_cols="infer", y_cols="infer",
                                display_lines=True, title_note="",
                                fmt=mtpl.dates.DateFormatter("%H:%M"),
                                excluded_pairs=None, output="show", **kwargs):

        # gets the number of subplots based on the number of y columns
        num_plots = len(y_cols) * len(x_cols) - sum([excluded for excluded in
                                                     excluded_pairs.values()])
        num_y_values = len(y_cols)
        num_y_subplots = 1 + num_plots // 4
        num_x_subplots = math.ceil(num_plots / num_y_subplots)

        for x_col in x_cols.items():

            fig, axes = self.__graphing_x_component(num_x_subplots, num_y_subplots)

            # creates the subplots
            axes = np.array(axes).flatten()

            # iterates over each graph
            for axis, y_col in zip(axes, y_cols.items()):

                # gets titles
                self.__set_title_and_labels(axis, x_col, y_col=y_col, title_note=title_note)

                # graphs y values
                self.__graphing_y_component(df, axis, x_col, y_col,
                                            excluded_pairs, fmt)

    @_run_graphing_method
    def plot_combined_graph(self, y_axis_names, df="infer", x_cols="infer", y_cols="infer",
                            display_lines=True, title_note="",
                            excluded_pairs=None, fmt=mtpl.dates.DateFormatter("%H:%M"),
                            **kwargs):

        if not isinstance(y_axis_names, list):
            y_axis_names = [y_axis_names]

        if len(y_axis_names) != len(x_cols):
            raise Exception(f"The number of y_axis_names: {len(y_axis_names)} \
                    and graphs to plot: {len(x_cols)} are mistmatched")

        for y_axis_name, x_col, in zip(y_axis_names, x_cols.items()):

            # creates the subplots
            fig, axis = self.__graphing_x_component(1, 1)

            # TODO
            self.__set_title_and_labels(axis, x_col, title_note,
                                        y_axis_name=y_axis_name)

            # iterates over each graph
            for y_col in y_cols.items():

                # TODO
                self.__graphing_y_component(df, axis, x_col, y_col,
                                            excluded_pairs, fmt, combined=True)

            # enables a legend
            axis.legend()

    @_run_graphing_method
    def plot_3d_graph(self, df="infer", y_cols="infer", x_cols=None,
            title_note="", **kwargs):

        # checks the list is of length three
        if len(y_cols) != 3:
           raise ParametricGraphLengthException
       
        # creates the 3d figure
        fig = plt.figure()
        fig.set_figheight(9)
        fig.set_figwidth(16)
        ax = fig.gca(projection="3d")

        # gets the three variables to plot
        x, y, z = [df[var] for var in y_cols.keys()]
        labels = [y_val["name"] for y_val in y_cols.values()]

        # sets the labels
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])

        # plots the data
        lines = ax.plot(x, y, z, ",")

        # sets the title
        title = " versus ".join(labels)
        ax.set_title(title)

        # sets legend
        ax.legend()

        # plt.savefig(title + ".png")
        # plt.show()

    def print_graphs(self, title, y_axis_names, df="infer", x_cols="infer",
                     y_cols="infer", **kwargs):

        # traditional graphs
        traditional_title = "traditional_graph_" + title + ".png"
        self.plot_traditional_graphs(df, x_cols, y_cols, title=traditional_title, **kwargs)

        # combined graph
        combined_title = "combined_graph_" + title + ".png"
        self.plot_combined_graph(df, x_cols, y_cols, y_axis_names=y_axis_names,
                                 title=combined_title, **kwargs)

        # 3d graph
        three_d_title = "3d_graph_" + title + ".png"
        try:
            self.plot_3d_graph(df, x_cols, y_cols, title=three_d_title)
        except Exception:
            pass

    def __str__(self):
        return str(self.df)


class GPSData(CustomData):
    def __init__(self, **kwargs):
        # the subfolder under each case that holds the 
        self.folder_name = "piksi"

        # everything at 5 HZ except for IMU raw at 2 HZ
        self.file_names = {
                "vel_ned.csv": 1,
                "base_headline.csv": 1,
                "pos_llh.csv": 1,
                "imu_raw.csv": 2.5,
                }

        super().__init__(**kwargs)

    def process_df(self, df, freq, *args, **kwargs):
        df = super().process_df(df, *args, **kwargs)
        # scales data
        try:
            df = self.scale_df_by_factor(df, freq)
        except Exception as e:
            print(f"Error when scaling by frequency: {str(e)}")

        # converts heading to degrees
        try:
            # old_heading = df["heading"]
            df["heading"] = df["heading"] / 1000
            # print(old_heading[df["heading"] == 0])
            # print("\n\n\n\n")
        except KeyError:
            pass

        return df

    def post_process_df(self, df):
        # drops rows that aren't fully merged
        df.dropna(inplace=True)

        # sorts by time
        try:
            df = df.sort_values(by="time")
        except KeyError:
            pass

        # resets index
        df = df.reset_index()

        return df

    def load_data(self, data_locs, inplace=True, **kwargs):
        # gets the args for the file types
        filename_settings = kwargs.pop("filename_settings")

        # derives the df value from the first file
        scale_by = None

        all_dataframes = pd.DataFrame()
        for file_name, manual_frequency in self.file_names.items():

            folder_dataframes = pd.DataFrame()

            # gets the file-specific args
            file_args = filename_settings[file_name]

            for file_path in find_files(data_locs,
                                        extension=file_name):

                # gets the data
                data = self.load_dataframe(file_path, **file_args, **kwargs)

                # gets the scale factor
                if scale_by == None:
                    scale_by = len(data)
                frequency = scale_by / len(data)

                # processes the data

                processed_data = self.process_df(data, frequency)

                # adds the processed data to the current folders data frame
                folder_dataframes = pd.concat([folder_dataframes, processed_data], axis=1)

            # adds the folder dataframes to the the list of all of the
            # dataframes
            all_dataframes = pd.concat([all_dataframes, folder_dataframes], axis=1)

        all_dataframes = self.post_process_df(all_dataframes)

        return all_dataframes

    def get_gps_plot_boundaries(self, longitude, latitude):
        lon_min, lon_max = longitude.min(), longitude.max()
        lat_min, lat_max = latitude.min(), latitude.max()
        lon_center, lat_center = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2

        return lon_min, lon_max, lat_min, lat_max, lon_center, lat_center

    @_run_inplace_method
    def plot_gps_graph(self, df="infer", lon_col="longitude",
                       lat_col="latitude", inplace=False, animated=True,
                       title="", points_to_show=100, plot_style="r^",
                       plot_every=1, **kwargs):

        def get_graph_title(df, ax, frame):
            # sets the time for readability
            try:
                time = str(df["time"][frame].time())
                ax.set_title(title + " " + time)
            except KeyError:
                pass

        def init_plot_gps_points():
            line.set_data([], [])
            return line,

        # function that can graph both animated and full data graphs
        # pass -1 for it to graph the entire dataframe
        def plot_gps_points(frame):
            # checks if this frame should be plotted
            get_graph_title(df, ax, frame)

            # gets the min frame (so there is more than 1 point at once)
            min_frame = 0 if frame < points_to_show else frame - points_to_show

            # plots the GPS data
            gps_points = gps_df.iloc[min_frame:frame:plot_every]
            line.set_data(gps_points.longitude,
                          gps_points.latitude)
            return line,

        # gets values and graph boundaries
        lon_min, lon_max, lat_min, lat_max, lon_center, lat_center = self.get_gps_plot_boundaries(df[lon_col], df[lat_col])

        # creates projection
        # projection = ccrs.PlateCarree(central_longitude=lon_center)
        projection = ccrs.PlateCarree()

        # creates axis and adds the min and max lon and lat to display and
        # adds gridlines
        fig, ax = plt.subplots(figsize=(16, 9))
        ax = plt.axes(projection=projection)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max],
                      crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, dms=True,
                     x_inline=False, y_inline=False)

        # adds background image tiles
        map_tiles = img_tiles.MapboxTiles(MAPBOX_TOKEN, "satellite")
        ax.add_image(map_tiles, 15)

        # plots the graph
        line, = ax.plot([], [], plot_style)

        # gets the gps df
        gps_df = df[[lon_col, lat_col]]
        gps_df = geopandas.GeoDataFrame(gps_df,
                                        geometry=geopandas.points_from_xy(gps_df.longitude,
                                                                          gps_df.latitude),
                                        crs=projection.proj4_init)

        print(f"animated: {animated}")
        if animated:
            # only plots every n points
            gps_df = gps_df[::plot_every]
            ani = animation.FuncAnimation(fig, plot_gps_points,
                                          init_func=init_plot_gps_points,
                                          frames=len(gps_df),
                                          interval=500, blit=False, repeat=False)
            writer = animation.FFMpegWriter(fps=60)
            file_name = title if title else str(dt.datetime.now())
            ani.save(os.path.join("animations", file_name) + ".mp4",
                     writer=writer)
        else:
            plot_gps_points(-1)

        # plt.show()

    def old_iterate_over_run_patient_case(self, func, base_dir=None, load_args={}, func_args={}):

        if base_dir is None:
            base_dir = self.data_locs

        run_dirs = [os.path.join(base_dir, run_dir) for run_dir in os.listdir(base_dir) if
                run_dir.endswith("-run")]

        for run_dir in run_dirs:
            patient_dirs = [os.path.join(run_dir, patient_dir) for patient_dir in os.listdir(run_dir)]
            run_name = prettify_var_name(run_dir.split(os.path.sep)[-1])

            for patient_dir in patient_dirs:
                case_dirs = [os.path.join(patient_dir, case_dir) for case_dir in os.listdir(patient_dir)]
                patient_name = prettify_var_name(patient_dir.split(os.path.sep)[-1])

                for case_dir in case_dirs:
                    case_name = prettify_var_name(case_dir.split(os.path.sep)[-1])

                    # loads the data
                    case_df = self.load_data([case_dir], inplace=False, **load_args)

                    # checks if the title should be inferred
                    title = func_args.get("title", " ".join([run_name, patient_name, case_name]))

                    print(f"plotting{case_dir}")
                    # runs the function provided
                    # if not case_df.empty:
                    func(df=case_df, title=title, **func_args)
                    # else:
                    #     print(f"ERROR: ID's on {case_dir} do not match up")
                    print("\n")

            shift_heading = np.vectorize(self._convert_relative_to_absolute_north)

    @staticmethod
    @np.vectorize
    def convert_relative_to_absolute_north(x, y, heading_degrees):
        # converts the x and y vals to polar format
        polar_coord = cmath.polar(complex(x, y))

        # subtracts the heading offset from the polar coordinates (rotates)
        # new_angle = polar_coord[1] + (math.pi * 2 + math.radians(heading_degrees)) % (math.pi * 2)
        new_angle = cmath.pi * 4 + polar_coord[1] - math.radians(heading_degrees)

        # the rectangular coordinates
        rect_coord = cmath.rect(polar_coord[0], new_angle)

        # gets the radians
        radians = (new_angle % (cmath.pi * 2)) / cmath.pi

        # and returns them in a tupled, rounded format
        return rect_coord.real, round(rect_coord.imag, 5), polar_coord[0], radians


class WindData(CustomData):
    def __init__(self, **kwargs):
        # wind vals
        # self.start_times = start_times
        # self.run_day = run_day

        super().__init__(extension=".txt", **kwargs)

    def load_data(self, data_locs, inplace=True, **kwargs):

        # loads the data and processes it
        df = self.concatenate_dataframes_from_files(data_locs,
                                                    df_func=self.process_df,
                                                    **kwargs)
        # sorts dataframe
        df.sort_values(by="time", inplace=True)

        return df

    def process_df(self, df, inplace=True, **kwargs):
        # gets the start time and adds the columns
        # try:
        #     start_time = self.get_start_time(file_path, **kwargs)
        #     self.add_wind_columns(df, start_time, **kwargs)
        # except NoTimeColumnError:
        #     return pd.DataFrame()

        # adds the speed column
        df = self._add_speed_col(df)
        df.dropna(inplace=True)

        df = super().process_df(df)
        return df

    def get_start_time(self, file_path, **kwargs):
        """
        Gets a wind files start time

        Parameters
        ----------
        file_path (str, filepath)
            The files path, uses both the directory and name to get the time
        """
        # splits the file into the path and filename
        directory, file_name = os.path.split(file_path)

        # splits the directory into the bottom level folder
        _, directory = os.path.split(directory)

        # gets what the start time is
        if directory == "galvmorn":
            time_of_day = "morning"
        elif directory == "galvafter":
            time_of_day = "afternoon"
        else:
            raise NoTimeColumnError("There is no associated time of day \
                                         for the folder {file_path}")

        # gets the file name without underscores and an extension
        stripped_file_name = "".join(os.path.splitext(file_name)[0].split("_"))

        # returns the start time
        try:
            start_time = self.start_times[time_of_day][stripped_file_name]
            return start_time
        except KeyError:
            raise NoTimeColumnError(f"No matching time for file: \
                                         {file_name} (time of day: \
                                         {time_of_day})")

    def _add_speed_col(self, df):
        @np.vectorize
        def _speed_col(x, y):
            result = np.sqrt(np.square(x) + np.square(y))
            return result

        # adds the speed col
        speed = _speed_col(df["wind_x_velocity"],
                           df["wind_y_velocity"])

        # adds the speed to the dataframe
        df["wind_speed"] = pd.Series(speed)

        # assigns the new df
        return df

    def DEPRECATED_add_wind_columns(self, df, start_time, frequency_hertz=20,
                         offset=dt.timedelta(hours=6),
                         drop_na=True, inplace=True, **kwargs):
        """
        Adds several useful columns to a wind dataframe, based on existing data

        Parameters:
        -----------
        df (pd.DataFrame)
            The dataframe to operate on
        start_time (dt.datetime)
            The time to start the increments from
        frequency_hertz (int)
            The number of increments per second
        drop_na (bool)
            Whether or not to drop bad rows
        inplace (bool)
            Whether or not to perform the operation in place
        """

        # checks if there should be a return
        if not inplace:
            df = df.copy()

        # adds the speed col
        speed = _speed_col(df["wind_x_velocity"],
                           df["wind_y_velocity"])

        # adds the speed to the dataframe
        df["wind_speed"] = pd.Series(speed)

        # creates the time incremented from start_time
        ms_freq = 1 / frequency_hertz * 1000
        incremented_times = pd.timedelta_range(start=start_time,
                periods=len(df.index), freq=f"{ms_freq}ms")

        # print(incremented_times)
        df["time"] = self.run_day + incremented_times + offset

        # drops nas
        if drop_na:
            df.dropna(inplace=True)

        # assigns the new df
        return df

    def get_wind_statistics(self, df):
        """
        Assigns several statistics internally

        Parameters
        ----------
        df (pd.DataFrame)
            The dataframe to get statistics for
        """

        # the turbulent kinetic energy value
        turbulent_kinetic_energy = 0

        # the overall mean velocity
        velocity_mean = 0

        # gets the turbulence
        for velocity in ["wind_x_velocity", "wind_y_velocity", "wind_z_velocity"]:

            # gets the velocity column
            velocity_col = df[velocity]

            # the mean velocity
            velocity_component_mean = velocity_col.mean()

            # the difference between instantaneous and average velocity mean and
            # variance
            velocity_difference_mean = 0
            velocity_difference_variance = 0

            # gets the mean and variance
            for row in self.df[velocity].iteritems():

                # the difference for the current row
                difference = row[1] - velocity_component_mean

                # adds to the mean and variance
                velocity_difference_mean += difference
                velocity_difference_variance += difference**2

            # divides by the number of observations
            num_observations = len(velocity_col)
            velocity_difference_mean /= num_observations
            velocity_difference_variance /= num_observations

            # adds the difference variance to the turbulent kinetic energy
            turbulent_kinetic_energy += velocity_difference_variance

            # adds the squared velocity component to the mean velocity
            velocity_mean += velocity_component_mean**2

        # takes square root of the mean velocity
        velocity_mean = math.sqrt(velocity_mean)

        # halves the turbulent kinetic energy
        turbulent_kinetic_energy /= 2

        # calculates the root mean square of turbulent velocity fluctuations
        rms_turbulent_velocity_fluctuations = math.sqrt(2 / 3 *
                                                        turbulent_kinetic_energy)

        # calculates the turbulence intensity/level
        turbulence_intensity = rms_turbulent_velocity_fluctuations / velocity_mean

        # assigns the values
        self.velocity_mean = velocity_mean
        self.rms_turbulent_velocity_fluctuations = rms_turbulent_velocity_fluctuations
        self.turbulent_kinetic_energy = turbulent_kinetic_energy
        self.turbulence_intensity = turbulence_intensity
        return velocity_mean, rms_turbulent_velocity_fluctuations,
        turbulent_kinetic_energy, turbulence_intensity


# NON CLASS FUNCTIONS

def combine_folders(base_dir, folder_permutations,
                    strip_from_dest="2021-01-16-", get_before_underscore=True,
                    **kwargs):

    # the list of all folders to copy
    folders_to_copy_from = []

    # finds all the folder permutations (past the top level folders)
    for layer in folder_permutations[1:]:

        # checks if the list is empty
        if not folders_to_copy_from:
            folders_to_copy_from = layer

        # else adds on to the existing folders
        else:

            # the list of replacements for the folder to copy
            new_folders_to_copy_from = []

            # iterates over each folder to copy
            for folder_to_copy in folders_to_copy_from:

                # iterates over each file in the layer
                for subfolder in layer:
                    # adds the new subfolder to the list
                    new_subfolder = os.path.join(folder_to_copy, subfolder)
                    new_folders_to_copy_from.append(new_subfolder)

            # replaces the list
            folders_to_copy_from = new_folders_to_copy_from

    # creates the combined folder
    try:
        output_dir = os.path.join(base_dir, "combined_runs")
        os.mkdir(output_dir)
    # else exits as this program has already been run
    except FileExistsError:
        # return
        pass

    # the top level directories to copy from
    top_level_dirs = [os.path.join(base_dir, top_level_dir) for top_level_dir
                      in folder_permutations[0]]

    # iterates over each subfolder
    for subfolder in folders_to_copy_from:

        # directory to copy from
        copy_to = os.path.join(output_dir, subfolder)

        # tries to make the directory in the combined data runs
        try:
            os.makedirs(copy_to)
        # else it has already been made
        except FileExistsError:
            pass

        # copies all the files from the subfolder path in each top level dir
        for top_level_dir in top_level_dirs:

            # directory to copy from
            copy_from = os.path.join(top_level_dir, subfolder)

            # tries copying folders in that directory
            try:
                for folder_to_copy in os.listdir(copy_from):
                    # gets its absolute path and copies it
                    folder_to_copy_path = os.path.join(copy_from, folder_to_copy)

                    # the name stripped of a user-defined string
                    dest_dir_name = \
                    str.join(*folder_to_copy.split(strip_from_dest))

                    # checks if the directory name should be split before
                    # underscore
                    dest_dir_name = dest_dir_name.split("_")[0]

                    # creates the absolute directory path
                    folder_destination_path = os.path.join(copy_to,
                                                           dest_dir_name)

                    # tries to copy the file
                    try:
                        shutil.copytree(folder_to_copy_path,
                                        folder_destination_path)
                    except FileExistsError:
                        pass
            # passes if that is not a directory
            except FileNotFoundError:
                pass


def prettify_var_name(var_name, chars_to_split="_-"):
    compiled_chars_to_split = re.compile(chars_to_split)
    new_var_name = " ".join(re.split(compiled_chars_to_split, var_name)).title()
    return new_var_name


def find_files(data_locations, extension=".csv", starts_with="", **kwargs):

    # iterates over each directory
    for data_location in data_locations:

        # checks if the location is a file
        if os.path.isfile(data_location):
            yield data_location

        # else if the location is a directory
        elif os.path.isdir(data_location):

            # finds the files in the directory that are of the proper extension
            match_pattern = starts_with + "*" + extension
            for path in Path(data_location).rglob(match_pattern):
                yield path

        # else it doesn'"t exist
        else:
            raise FileNotFoundError(f"The data location {data_location} does not exist")


# DEPRECATED FUNCTIONS
def DEPRECATED_old_data_main():

    # the base directories
    base_dir = os.path.join("/home/larva/Research/bhaganagar/wind-gps-data/data")
    gps_dir = os.path.join(base_dir, "gps-data")
    combined_runs_dir = os.path.join(base_dir, "gps-data", "combined_runs")
    wind_dir = os.path.join(base_dir, "wind-data")

    # a dictionary of start times
    run_day = dt.datetime(2021, 1, 16)
    start_times = {"morning": {
                    "winddata1": dt.timedelta(hours=11, minutes=13),
                    "winddata2": dt.timedelta(hours=11, minutes=25),
                    "winddata3": dt.timedelta(hours=11, minutes=41),
                    "winddata42": dt.timedelta(hours=12, minutes=39),
                    "winddata5": dt.timedelta(hours=12, minutes=49),
                    # "winddata6": dt.timedelta(hours=13, minutes=00),
                    # "winddata7": dt.timedelta(hours=13, minutes=26),
                    "winddata8": dt.timedelta(hours=13, minutes=38),
                    "winddata9": dt.timedelta(hours=13, minutes=46),
                  },
                  "afternoon": {
                    "winddata12": dt.timedelta(hours=16, minutes=2),
                    "winddata2": dt.timedelta(hours=16, minutes=9),
                    "winddata3": dt.timedelta(hours=16, minutes=18),
                    "winddata42": dt.timedelta(hours=16, minutes=47),
                    "winddata5": dt.timedelta(hours=16, minutes=54),
                    "winddata6": dt.timedelta(hours=16, minutes=0),
                    "winddata7": dt.timedelta(hours=17, minutes=9),
                    "winddata8": dt.timedelta(hours=17, minutes=16),
                  }}
    wind_args.update({"run_day": run_day,
                      "start_times": start_times})


    # creates the major component wind graph
    wind_data = WindData(data_locs=wind_dir, **wind_args)
    # wind_data.print_graphs(y_axis_names="Wind Motion")

    # combined_runs_dir = os.path.join(base_dir, "gps-data", "combined_runs", "Oscar_1", "Against_Current")
    lidar_data = pd.read_csv(os.path.join(gps_dir, "Pythonized_LIDAR_Leeway_centroid_traj_2021-01-16-17-10-18_2.csv"))
    lidar_data["tplot"] = pd.to_datetime(lidar_data["tplot"])

    gps_data = GPSData(data_locs=gps_dir, **gps_args)

    # merges the dataframes based on time
    merged_data = pd.merge_asof(wind_data.df, gps_data.df, on="time").dropna()
    merged_data.dropna().drop_duplicates(["x_gyr",
                                          "y_gyr",
                                          "z_gyr",
                                          "wind_x_velocity",
                                          "wind_y_velocity",
                                          "wind_z_velocity",
                                         ], inplace=True)

    merged_lidar_data = pd.merge(lidar_data, merged_data, right_on="time", left_on="tplot").dropna()
    merged_lidar_data["wind_x_velocity"], merged_lidar_data["wind_y_velocity"], merged_lidar_data["wind_polar_r"], merged_lidar_data["wind_polar_radians"] = GPSData.convert_relative_to_absolute_north(merged_lidar_data["wind_x_velocity"],
                                                                                                      merged_lidar_data["wind_y_velocity"],
                                                                                                      merged_lidar_data["heading"])

    merged_lidar_data.drop(["level_0", "n", "e", "d", "h_accuracy",
    "v_accuracy", "x_acceleration", "y_acceleration", "z_acceleration",
    "wind_z_velocity", "temperature", "height", "tow_f", "index", "tow",
    "x_gyr", "y_gyr", "z_gyr", "tplot", "stamp"], axis=1, inplace=True)
    # merged_lidar_data = merged_lidar_data[((merged_lidar_data["time"] >= dt.datetime(2021, 1, 16, 23, 10, 18)) & (merged_lidar_data["time"] <= dt.datetime(2021, 1, 16, 23, 10, 46)))]

    merged_lidar_data["wind_direction"] = merged_lidar_data["wind_polar_radians"] / math.pi * 180
    merged_lidar_data.sort_values(by="time", inplace=True)
    merged_lidar_data.rename(columns={"wind_x_velocity": "wind_u_velocity",
                                      "wind_y_velocity": "wind_v_velocity"},
                             inplace=True)

    merged_lidar_data.to_csv("Lidar_wind_gps_data_23:10:18.4_to_23:10:46.6.csv", index=False)

if __name__ == "__main__":
    mtpl.use('TkAgg')

    main()

