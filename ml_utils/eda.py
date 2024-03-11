import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_null_and_duplicated(data):
    print("------ Null values ------")
    print("Totals: ", data.isnull().sum().sum())
    print("-----------------------")
    print(data.isnull().sum())
    print("")
    print("")
    print("------ Duplicated -----")
    print("Totals: ", data.duplicated().sum())
    print("-----------------------")
    return data[data.duplicated()]


def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to the show density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  # Number of rows of the subplot grid = 2
        sharex=True,  # x-axis will be shared among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  # Creating the 2 subplots
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )  # Boxplot will be created and a star will indicate the mean value of the column
    (
        sns.histplot(
            data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins, palette="winter"
        )
        if bins
        else sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    )  # For histogram
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    )  # Add mean to the histogram
    ax_hist2.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


def sorted_cat_count(df, cat):
    (rows, columns) = df.shape
    r = df[cat].value_counts()
    counts = pd.DataFrame(
        {"name": r.index, "counts": r, "perc": round(r / rows * 100, 2)}
    )
    return counts.sort_values("counts", ascending=False)


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # Length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index,
        hue=feature,
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # Percentage of each class of the category
        else:
            label = p.get_height()  # Count of each level of the category

        x = p.get_x() + p.get_width() / 2  # Width of the plot
        y = p.get_height()  # Height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # Annotate the percentage

    plt.show()  # Show the plot


def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left",
        frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


# outlier detection using boxplot
# Percentage of outliers print
def show_outliers(data, numeric_columns):
    plt.figure(figsize=(15, 12))

    for i, variable in enumerate(numeric_columns):
        plt.subplot(4, 4, i + 1)
        plt.boxplot(data[variable], whis=1.5)
        plt.tight_layout()
        plt.title(variable)

    plt.show()

    # to find the 25th percentile and 75th percentile for the numerical columns.
    Q1 = data[numeric_columns].quantile(0.25)
    Q3 = data[numeric_columns].quantile(0.75)

    IQR = Q3 - Q1  # Inter Quantile Range (75th percentile - 25th percentile)

    lower_whisker = (
        Q1 - 1.5 * IQR
    )  # Finding lower and upper bounds for all values. All values outside these bounds are outliers
    upper_whisker = Q3 + 1.5 * IQR
    print("Percentage of outliers in each column")
    print(
        (
            (data[numeric_columns] < lower_whisker)
            | (data[numeric_columns] > upper_whisker)
        ).sum()
        / data.shape[0]
        * 100
    )


def treat_outliers(df, col):
    """
    treats outliers in a variable
    col: str, name of the numerical variable
    df: dataframe
    col: name of the column
    """
    Q1 = df[col].quantile(0.25)  # 25th quantile
    Q3 = df[col].quantile(0.75)  # 75th quantile
    IQR = Q3 - Q1  # Inter Quantile Range (75th perentile - 25th percentile)
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR

    # all the values smaller than lower_whisker will be assigned the value of lower_whisker
    # all the values greater than upper_whisker will be assigned the value of upper_whisker
    # the assignment will be done by using the clip function of NumPy
    df[col] = np.clip(df[col], lower_whisker, upper_whisker)

    return df


def heatmap(data, color="coolwarm"):
    # Correlation check
    cols_list = data.select_dtypes(include=np.number).columns.tolist()

    plt.figure(figsize=(15, 7))

    sns.heatmap(
        data[cols_list].corr(), annot=True, vmin=-1, vmax=1, fmt=".2f", cmap=color
    )

    plt.show()


def corr_barplot(data, x=None, y=None, figsize=None):
    if figsize:
        plt.figure(figsize=figsize)
    sns.barplot(
        data=data,
        x=x,
        y=y,
        errorbar=("ci", False),
        order=data.groupby([x])[y].mean().sort_values(ascending=False).index,
        hue=x,
    )
    plt.xticks(rotation=90)
    plt.show()
