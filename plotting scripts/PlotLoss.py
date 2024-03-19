from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)
    
    print(df)

    plot = sns.lineplot(data=df.loc[:, ["train loss", "test loss"]], markers=True)
    plot.grid()
    plot.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig("../plots/Loss.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")