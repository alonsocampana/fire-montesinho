import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Other
def spatial_counts(df):
    y_max = df["Y"].max()
    y_min = df["Y"].min()
    x_max = df["X"].max()
    x_min = df["X"].min()
    counts = np.zeros([y_max-y_min+1, x_max-x_min+1])
    for i in np.arange(0, len(df)):
        x = (df["X"].iloc[i]-x_min)
        y = (df["Y"].iloc[i]-y_min)
        counts[y][x] += 1
    return counts

def average_area(df):
    y_max = df["Y"].max()
    y_min = df["Y"].min()
    x_max = df["X"].max()
    x_min = df["X"].min()
    xy_counts = spatial_counts(df)
    cumsums = np.zeros([y_max-y_min+1, x_max-x_min+1])
    for i in np.arange(0, len(df)):
        x_start = df["X"].min()
        y_start = df["Y"].min()
        x = (df["X"].iloc[i]-x_start)
        y = (df["Y"].iloc[i]-y_start)
        cumsums[y][x] += df["area"].iloc[i]
        
    return np.divide(cumsums, xy_counts, out=np.zeros_like(cumsums), where=xy_counts!=0)

# Plots
def plot_xy_counts(fires, ax):
    xy_counts = spatial_counts(fires)
    g = sns.heatmap(xy_counts[::-1],cmap = "rainbow", center=1, annot=True, cbar=False, ax=ax)
    yticks = np.arange(fires["Y"].min(), fires["Y"].max()+1)
    xticks = np.arange(fires["X"].min(), fires["X"].max()+1)
    g.set(yticklabels=yticks[::-1])
    g.set(xticklabels=xticks)
    g.axes.set_xlabel("X", fontsize=20)
    g.axes.set_ylabel("Y", fontsize=20)
    g.axes.set_title("Number of fires per coordinates",fontsize=20)
    return g

def plot_xy_averages(fires, ax, title = "Average area burnt in each fire", vmax=30):
    average_areas = average_area(fires)
    g = sns.heatmap(average_areas[::-1], vmax=vmax, cmap = "OrRd", cbar=False,annot=True, ax=ax)
    yticks = np.arange(fires["Y"].min(), fires["Y"].max()+1)
    xticks = np.arange(fires["X"].min(), fires["X"].max()+1)
    g.set(yticklabels=yticks[::-1])
    g.set(xticklabels=xticks)
    g.axes.set_xlabel("X", fontsize=20)
    g.axes.set_ylabel("Y", fontsize=20)
    g.axes.set_title(title,fontsize=20)
    return g

def plot_month_counts(fires, ax):
    fires_month = fires["month"].value_counts()
    months = ['jan', 'feb', 'mar', 'apr','may','jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
    g = fires_month[months].plot(ax=ax)
    g.axes.set_ylabel("Number of fires")
    g.axes.set_title("Number of fires per month",fontsize=20)
    return g

def plot_weekday_counts(fires, ax):
    fires_days = fires["day"].value_counts()
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    g = sns.lineplot(data = fires_days[days], ax=ax)
    g.axes.set_ylabel("Number of fires")
    g.axes.set_title("Number of fires per day of the week",fontsize=20)
    return g

def plot_month_averages(fires, ax=None):
    if ax is None:
        ax = plt.axes()
    months = ['jan', 'feb', 'mar', 'apr','may','jun', 'jul', 'aug','sep', 'oct', 'nov', 'dec']
    areas = [fires[fires.month == m]["area"].mean() for m in months]
    areas_series = pd.Series(areas, index=months)
    g = areas_series.plot(ax=ax)
    g.axes.set_ylabel("Average area burnt")
    g.axes.set_title("Average area burnt by fire each month",fontsize=17)
    return g

def plot_weekday_averages(fires, ax=None):
    if ax is None:
        ax = plt.axes()
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    areas = [fires[fires.day == m]["area"].mean() for m in days]
    areas_series = pd.Series(areas, index=days)
    g = areas_series.plot(ax=ax)
    g.axes.set_ylabel("Average area burnt")
    g.axes.set_title("Average area burnt by fire each weekday",fontsize=17)
    return g

def plot_numerical_counts(fires):
    numerical = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH','wind', 'rain', 'area']
    n = 3
    m = 3
    fig,axs = plt.subplots(n, m, figsize = (15, 10))
    x = 0
    for i in np.arange(0, n):
        for j in np.arange(0, m):
            if (x < len(numerical)):
                sns.histplot(data=fires[numerical[x]], ax=axs[i][j]) 
                x+=1
            else:
                axs[i][j].axis('off')
    fig.suptitle('Counts of fires over the different values of the numerical variables', fontsize=20)
    plt.show()
    
def plot_nonzero_area(fires, ax=None):
    if ax is None:
        ax = plt.axes()
    g = sns.histplot(fires[fires["area"]!=0]["area"], ax=ax)
    g.set_title("Counts of the different non-zero areas burnt", fontsize=15)
    return g
    
def plot_nonzero_area_logtransformed(fires, ax=None):
    if ax is None:
        ax = plt.axes()
    g = sns.histplot(fires[fires["area"]!=0]["area"].apply(np.log), ax=ax)
    g.set_title("Counts of the different non-zero areas burnt, log transformed", fontsize=15)
    return g

def plot_ffmc_wo_outliers(fires, ax=None):
    if ax is None:
        ax = plt.axes()
    ffmc = fires[["FFMC"]]
    ffmc_std = ffmc.std()
    ffmc_mean = ffmc.mean()
    threshold = ffmc_mean - 2*ffmc_std
    ffmc_without_outlyers = ffmc[ffmc>threshold]["FFMC"]
    g = sns.histplot(ffmc_without_outlyers,ax=ax)
    g.set_title("FFMC counts, outliers removed", fontsize=15)
    return g

def plot_ISI_wo_outliers(fires, ax=None):
    if ax is None:
        ax = plt.axes()
    isi = fires[["ISI"]]
    isi_std = isi.std()
    isi_mean = isi.mean()
    threshold = isi_mean + 2*isi_std
    isi_without_outlyers = isi[isi<threshold]["ISI"]
    g = sns.histplot(isi_without_outlyers, ax=ax)
    g.set_title("ISI counts, outlier removed", fontsize=15)
    return g

def plot_DC_split(fires):
    fig, axes = plt.subplots(1, 3, figsize=(28, 10))
    sns.histplot(fires[fires[["DC"]] > 500]["DC"], ax=axes[2])
    sns.histplot((fires[(fires["DC"] < 500).to_numpy() & (fires["DC"] > 150).to_numpy()]["DC"]), ax=axes[1])
    sns.histplot(fires[fires[["DC"]] < 150]["DC"], ax=axes[0])
    fig.suptitle("Counts of the variable DC split into 3 ranges", fontsize="20")
    plt.show()