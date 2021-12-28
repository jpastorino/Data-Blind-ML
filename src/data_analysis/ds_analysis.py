import sys
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn import preprocessing
from datetime import date


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)


def addlabel(ax, rects, labels):
    i = 0
    for rect in rects:
        height = rect.get_height()
        ax.annotate(labels[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        i += 1


def plot_metric(labels, metric_data, dataset_name, metric, save_to, out_file, line_value=0.5):
    fig, ax = plt.subplots()
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    rects1 = ax.bar(x - width / 2, metric_data, width, label='Features')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('{}'.format(metric))
    ax.set_title('{} dataset {} analysis'.format(dataset_name, metric))
    plt.hlines(line_value, 0, len(labels), linestyles='dotted')
    plt.hlines(-line_value, 0, len(labels), linestyles='dotted')
    if len(labels) < 25:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)

        autolabel(ax, rects1)
        plt.xticks(rotation=40)

    ax.set_xlabel("Features")
    # else:
    #     ax.set_xticklabels(["F_" + str(a) for a in x])


    fig.tight_layout()
    plt.savefig(save_to)
    out_file.write("Plot was generated and stored at >>>{}\n".format(save_to))
    plt.show()


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    """ RUNs the Dataset Analysis for the given csv file"""

    DATA_PATH = "./data/source"
    OUTPUT_PATH = "./output"

    today = date.today()

    ### DISPLAY FILES
    files = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".csv"):
            files.append(file[:-4])

    files.sort()
    for i, file in enumerate(files):
        print("{} - {}".format(i + 1, file))
    print("-" * 30)
    selection = int(input("Choose file to process [1-{}]:".format(len(files))))

    if not (selection >= 1 and selection <= len(files)):
        print("Invalid Selection. Program Terminated.")
        exit(1)

    ## FILE SELECTED - OPEN
    filename = files[selection - 1]
    csv_file = f"{DATA_PATH}/{filename}.csv"
    image_out_file_placeholder = f"{OUTPUT_PATH}/{filename}.analysis.{'{}'}.png"
    out_file = open(f"{OUTPUT_PATH}/{filename}.analysis.report.txt", "w")

    print("Processing {}".format(filename))
    print()
    out_file.write("Data Analysis Report for {}\n".format(filename))
    out_file.write("{}\n".format(today))

    has_header = input("Does the file has a header? [Y/n]")
    if has_header.lower() == "n":
        df = pd.read_csv(csv_file, header=None, prefix="x")
    else:
        df = pd.read_csv(csv_file)

    # ######### SNS PAIR PLOT
    user_option = input("Do you want to generate PairPlot ? [y/N]")
    if user_option.lower() == "y":
        print("-" * 40)
        print("working....")
        img_out_file = image_out_file_placeholder.format("pair-plot")
        sns_plot = sns.pairplot(df)
        sns_plot.savefig(img_out_file)
        out_file.write("Pair plot was generated and saved at {}\n".format(img_out_file))

    # ######### SKEWNESS
    user_option = input("Compute Skewness? [Y/n]")
    if not user_option.lower() == "n":
        out_file.write("Computing Skewness\n")
        out_file.write("-" * 40 + "\n")
        print("-" * 40)
        labels = []
        skewness = []
        dataTypeDict = dict(df.dtypes)
        print("{:^40} | {:^15}".format("Feature", "Skewness"))
        print("-" * 60)
        out_file.write("{:^40} | {:^15}\n".format("Feature", "Skewness"))
        out_file.write("-" * 60 + "\n")
        for col in df.columns:
            data = df[col].dropna()
            notes = ""
            if not np.issubdtype(dataTypeDict[col], np.number):
                notes = "Encoding {} dType: {}".format(col, dataTypeDict[col])
                le = preprocessing.LabelEncoder()
                le.fit(data)
                data = le.transform(data)
            labels.append(col)
            skewness.append(stats.skew(data))
            if col == df.columns[-1]:
                print("{:^40} | {:10.5f}       {}".format(col, skewness[-1], notes))
            out_file.write("{:^40} | {:10.5f}       {}\n".format(col, skewness[-1], notes))

        plot_metric(labels, skewness, filename, "Skewness", image_out_file_placeholder.format("skewness"), out_file)

    # ######### KURTOSIS
    user_option = input("Compute Kurtosis? [Y/n]")
    if not user_option.lower() == "n":
        out_file.write("\n\nComputing Kurtosis\n")
        out_file.write("-" * 40 + "\n")
        print("-" * 40)
        labels = []
        kurtosis = []
        dataTypeDict = dict(df.dtypes)
        print("{:^40} | {:^15}".format("Feature", "Kurtosis"))
        print("-" * 60)
        out_file.write("{:^40} | {:^15}\n".format("Feature", "Kurtosis"))
        out_file.write("-" * 60 + "\n")

        for col in df.columns:
            data = df[col].dropna()
            notes = ""
            if not np.issubdtype(dataTypeDict[col], np.number):
                notes = "Encoding {} dType: {}".format(col, dataTypeDict[col])
                le = preprocessing.LabelEncoder()
                le.fit(data)
                data = le.transform(data)
            labels.append(col)
            kurtosis.append(stats.kurtosis(data))
            if col == df.columns[-1]:
                print("{:^40} | {:10.5f}       {}".format(col, kurtosis[-1], notes))
            out_file.write("{:^40} | {:10.5f}       {}\n".format(col, kurtosis[-1], notes))

        plot_metric(labels, kurtosis, filename, "Excess Kurtosis", image_out_file_placeholder.format("kurtosis"),
                    out_file, line_value=0)

    # ##### Shapiro-Wilk Test (Data normality)
    user_option = input("Test Data is Normal Distributed? [Y/n]")
    if not user_option.lower() == "n":
        print("-" * 40)
        out_file.write("\n\nTesting If Data Follows Normal Distribution\n")
        out_file.write("-" * 40 + "\n")
        labels = []
        shapiro_p_value = []
        dataTypeDict = dict(df.dtypes)
        print("{:^40} | {:15} | {:^20}".format("Feature", "Shapiro P-Value", "Normally Dist"))
        print("-" * 81)
        out_file.write("{:^40} | {:15} | {:^20}\n".format("Feature", "Shapiro P-Value", "Normally Dist"))
        out_file.write("-" * 81 + "\n")

        for col in df.columns:
            data = df[col].dropna()
            notes = ""
            if not np.issubdtype(dataTypeDict[col], np.number):
                notes = "Encoding {} dType: {}".format(col, dataTypeDict[col])
                le = preprocessing.LabelEncoder()
                le.fit(data)
                data = le.transform(data)
            labels.append(col)
            shapiro_p_value.append(stats.shapiro(data)[1])
            if shapiro_p_value[-1] < 0.05:
                is_normal = "NO"
            else:
                is_normal = "YES"

            print("{:40} | {:3.9E} | {:^20}       {}".format(col, shapiro_p_value[-1], is_normal, notes))
            out_file.write("{:40} | {:3.9E} | {:^20}       {}\n".format(col, shapiro_p_value[-1], is_normal, notes))
