import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

def process_results(year: int) -> None:

    if year == 2022:
        df_volumes = pd.read_excel('data/Measurements.xlsx', sheet_name='new')
        sheet_name = 'PQ28jul2022'
        measured_trees = {
            1: [2, 3, 5, 7],
            2: [2, 4, 6, 8],
            3: [1, 2, 3, 5],
            4: [2, 3, 5, 7]
        }
    elif year == 2021:
        df_volumes = pd.read_excel('data/Measurements.xlsx', sheet_name='2021')
        sheet_name = 'PQ 29jul21'
        measured_trees = {
            1: [2, 4, 6, 8],
            2: [1, 3, 5, 7],
            3: [2, 4, 6, 8],
            4: [2, 3, 6, 8]
        }

    df_pqa = pd.read_excel('data/PQA_for_plant_21-22.xlsx', sheet_name=sheet_name)
    print(df_pqa)

    rows_mapping = [
        [1, 'F', 'A', 3],
        [1, 'F', 'A', 1],
        [1, 'F', 'B', 3],
        [1, 'F', 'B', 1],

        [1, 'VRA', 'A', 3],
        [1, 'VRA', 'A', 1],
        [1, 'VRA', 'B', 3],
        [1, 'VRA', 'B', 1],

        [2, 'F', 'A', 7],
        [2, 'F', 'A', 5],
        [2, 'F', 'B', 7],
        [2, 'F', 'B', 5],

        [2, 'VRA', 'A', 7],
        [2, 'VRA', 'A', 5],
        [2, 'VRA', 'B', 7],
        [2, 'VRA', 'B', 5]
    ]

    volumes = []
    widths = []
    canopy_gaps_percentage = []
    lln_vine = []
    interior_clusters_percentage = []
    vigors = []

    i = 0
    for row in measured_trees.keys():
        for tree in measured_trees[row]:
            print(df_volumes.loc[(df_volumes['Row'] == row) & (df_volumes['Plant'] == tree)])
            block, treatment, vigor, vine = rows_mapping[i]
            vigors.append(vigor)
            i += 1
            pqa_measurements = df_pqa.loc[(df_pqa['Block'] == block) & (df_pqa['Treatment'] == treatment) & (df_pqa['Vigor'] == vigor) & (df_pqa['Vine'] == vine)]
            print(pqa_measurements)
            print(pqa_measurements['% Canopy Gaps_Vine'].values[0], pqa_measurements['LLN_Vine'].values[0], pqa_measurements['% Interior Clusters_Vine'].values[0])
            canopy_gaps_percentage.append(pqa_measurements['% Canopy Gaps_Vine'].values[0])
            lln_vine.append(pqa_measurements['LLN_Vine'].values[0])
            interior_clusters_percentage.append(pqa_measurements['% Interior Clusters_Vine'].values[0])
            volumes.append(df_volumes.loc[(df_volumes['Row'] == row) & (df_volumes['Plant'] == tree)]['Polynomial SMALL - Baseline RGBD'].values[0])
            widths.append(df_volumes.loc[(df_volumes['Row'] == row) & (df_volumes['Plant'] == tree)]['Widths'].values[0])

    return volumes, widths, canopy_gaps_percentage, lln_vine, interior_clusters_percentage, vigors

def show(target, volumes, widths, vigors, reg=None):
    data_x = np.hstack((np.expand_dims(np.array(volumes), axis=1), np.expand_dims(np.array(widths), axis=1)))
    
    if reg is None:
        reg = LinearRegression().fit(data_x, target)

    xx, yy = np.meshgrid(volumes, widths)
    z = reg.coef_[0] * xx + reg.coef_[1] * yy + reg.intercept_

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.scatter(volumes, widths, target, 'green')
    ax.set_title('MAE: ' + str(np.mean(np.abs(reg.predict(data_x) - target))))
    ax.plot_surface(xx, yy, z, cmap='viridis', edgecolor='green')
    plt.show()

    print(np.corrcoef(reg.predict(data_x), target))
    vigors = np.array(vigors)
    print(vigors=="A")
    print(reg.predict(data_x)[vigors=="A"], np.array(target)[vigors=="A"])
    plt.scatter(reg.predict(data_x)[vigors=="A"], np.array(target)[vigors=="A"])
    plt.scatter(reg.predict(data_x)[vigors=="B"], np.array(target)[vigors=="B"])
    plt.xlabel('Volume')
    plt.ylabel('Canopy gaps %')
    plt.show()

    return reg

volumes22, widths22, canopy_gaps_percentage22, lln_vine22, interior_clusters_percentage22, vigors22 = process_results(2022)
volumes21, widths21, canopy_gaps_percentage21, lln_vine21, interior_clusters_percentage21, vigors21 = process_results(2021)

# plt.hist(canopy_gaps_percentage22, alpha=0.7)
# plt.hist(canopy_gaps_percentage21, alpha=0.7)
# plt.show()

# plt.hist(volumes22, alpha=0.7)
# plt.hist(volumes21, alpha=0.7)
# plt.show()

# reg1 = show(canopy_gaps_percentage22, volumes22, widths22, vigors22)
# reg2 = show(lln_vine22, volumes22, widths22, vigors22)
# reg3 = show(interior_clusters_percentage22, volumes22, widths22, vigors22)

show(canopy_gaps_percentage21, volumes21, widths21, vigors21)
show(lln_vine21, volumes21, widths21, vigors21)
show(interior_clusters_percentage21, volumes21, widths21, vigors21)


plt.scatter(widths21, canopy_gaps_percentage21)
plt.title('Correlation: ' + str(np.corrcoef(widths21, canopy_gaps_percentage21)[0, 1]))
plt.xlabel('Width')
plt.ylabel('Canopy gaps %')
plt.show()

plt.title('Correlation: ' + str(np.corrcoef(widths21, lln_vine21)[0, 1]))
plt.scatter(widths21, lln_vine21)
plt.xlabel('Width')
plt.ylabel('LLN Vine %')
plt.show()

plt.title('Correlation: ' + str(np.corrcoef(widths21, interior_clusters_percentage21)[0, 1]))
plt.scatter(widths21, interior_clusters_percentage21)
plt.xlabel('Width')
plt.ylabel('Interior clusters %')
plt.show()


plt.scatter(volumes21, canopy_gaps_percentage21)
plt.title('Correlation: ' + str(np.corrcoef(volumes21, canopy_gaps_percentage21)[0, 1]))
plt.xlabel('Volume')
plt.ylabel('Canopy gaps %')
plt.show()

plt.title('Correlation: ' + str(np.corrcoef(volumes21, lln_vine21)[0, 1]))
plt.scatter(volumes21, lln_vine21)
plt.xlabel('Volume')
plt.ylabel('LLN Vine %')
plt.show()

plt.title('Correlation: ' + str(np.corrcoef(volumes21, interior_clusters_percentage21)[0, 1]))
plt.scatter(volumes21, interior_clusters_percentage21)
plt.xlabel('Volume')
plt.ylabel('Interior clusters %')
plt.show()

