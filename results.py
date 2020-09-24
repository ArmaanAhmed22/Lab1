import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as lines 
from sklearn.linear_model import LinearRegression
import numpy as np
from math import pi
import matplotlib.ticker

df = pd.read_csv("results.csv",names=["Displacement","Time"])

sns.set_style("ticks")
#sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})


disp = df["Displacement"].values
time = df["Time"].values

reg = LinearRegression().fit(time.reshape(-1,1),disp)
rsquared = LinearRegression().fit(time.reshape(-1,1),disp).score(time.reshape(-1,1),disp)
print(f"unexplained variation: {1 - rsquared}")


time_0 = df.iloc[0]["Time"]
for row in range(df.shape[0]):
	df.at[row,"Time"] = df.iloc[row]["Time"] - time_0


fig, ax = plt.subplots(figsize=(5,2))
ax.grid(b=True, which='major', color='black', linewidth=0.75)
ax.grid(b=True, which='minor', color='black', linewidth=0.5)

sns.regplot(x="Time",y="Displacement",data=df,ax=ax,ci=None,scatter_kws={"color":"black","alpha":0.5,"edgecolor":"none"},line_kws={"color":"black","linewidth":0.5,"linestyle":"--"})

slope = ((ax.lines[0].get_ydata()[-1] - ax.lines[0].get_ydata()[0]) / ax.lines[0].get_xdata()[-1])

print(f"slope: {slope}")

ax.set_xlabel(r"Time ($s$)")
ax.set_ylabel(r"Displacement ($cm$)")
ax.set_ylim((0,55))
ax.set_yticks(range(0,51,10))
ax.set_xlim((0,1))


ax.set_xticks(np.arange(0,1.00000001,0.05),minor=True)
ax.set_yticks(np.arange(0,60,5),minor=True)
plt.savefig("xvst.png",dpi=600,bbox_inches="tight")
plt.clf()

fig, axes = plt.subplots(nrows=2, figsize=(5,4))
fig.tight_layout()

comparison_pd = [
		{"Time" : 0, "Displacement" : 0,"Type":"Obtained Velocity"},
		{"Time" : 1, "Displacement" : reg.coef_[0],"Type":"Obtained Velocity"},
		{"Time" : 0, "Displacement" : 0,"Type":"Lower Velocity"},
		{"Time" : 1, "Displacement" : 30,"Type":"Lower Velocity"},
		{"Time" : 0, "Displacement" : 0,"Type":"Greater Velocity"},
		{"Time" : 1, "Displacement" : 70,"Type":"Greater Velocity"}
	]


changing_list = list()
for i in np.arange(0,1,0.01):
	output = 70*(i)**2
	output2 = -70*(i)**2 + 70
	changing_list.append({"Time" : i, "Displacement" : output,"Type":"Increasing Velocity"})
	changing_list.append({"Time" : i, "Displacement" : output2,"Type":"Decreasing Velocity"})
	
print(f"acc of increasing: 140")
print(f"acc of decreasing: -140")

comparison_pd = pd.DataFrame(comparison_pd)
changing_pd = pd.DataFrame(changing_list)


get_slope = lambda typ,df: (df[df["Type"] == typ].iloc[0]["Displacement"] - df[df["Type"] == typ].iloc[1]["Displacement"]) / (df[df["Type"] == typ].iloc[0]["Time"] - df[df["Type"] == typ].iloc[1]["Time"])

print(f"slope of greater: {get_slope('Greater Velocity',comparison_pd)}")
print(f"slope of lesser: {get_slope('Lower Velocity',comparison_pd)}")


sns.lineplot(x="Time",y="Displacement",data=changing_pd,hue="Type",ax=axes[1])

sns.lineplot(x="Time",y="Displacement",data=comparison_pd,hue="Type",ax=axes[0])
sns.despine(ax=axes[0])
sns.despine(ax=axes[1])
axes[0].set_xlim(left=0)
axes[0].set_ylim(bottom=0)
axes[1].set_xlim(left=0)
axes[1].set_ylim(bottom=0)

axes[0].set_xlabel(r"Time ($s$)")
axes[0].set_ylabel(r"Displacement ($cm$)")
axes[1].set_xlabel(r"Time ($s$)")
axes[1].set_ylabel(r"Displacement ($cm$)")

plt.savefig("avv.png",dpi=600,bbox_inches="tight")





