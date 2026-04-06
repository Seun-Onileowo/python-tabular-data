import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

"""Run linear regressions and save plots for Iris species."""


def plot_species_regression(dataframe, species_name, output_file):
    """Plot a regression for one Iris species."""
    species_only = dataframe[dataframe.species == species_name]
    x = species_only.petal_length_cm
    y = species_only.sepal_length_cm
    regression = stats.linregress(x, y)
    slope = regression.slope
    intercept = regression.intercept

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x, slope * x + intercept, color="orange", label="Fitted line")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.title(species_name)
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def main():
    """Read the data and make the plots."""
    dataframe = pd.read_csv("iris.csv")
    plot_species_regression(
        dataframe,
        "Iris_setosa",
        "petal_v_sepal_length_regress_setosa.png",
    )
    plot_species_regression(
        dataframe,
        "Iris_versicolor",
        "petal_v_sepal_length_regress_versicolor.png",
    )
    plot_species_regression(
        dataframe,
        "Iris_virginica",
        "petal_v_sepal_length_regress_virginica.png",
    )


if __name__ == "__main__":
    main()
