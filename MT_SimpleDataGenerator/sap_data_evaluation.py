from sap.SapExtractor import SapExtractor

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import database_config
import config as cfg

font = {'size': 24}

matplotlib.rc('font', **font)

sap_extractor = SapExtractor(cfg.STORAGE_BASE_SAP_DATA)
db, min_time, max_time = sap_extractor.extract()


def generate_boxplot_changes_price_distribution():
    mta_changes = db.get_table('MTA_CHANGES').get_data()
    old_values = mta_changes['old_value'].astype(float)
    new_values = mta_changes['new_value'].astype(float)

    old_values = old_values[old_values < 40].tolist()
    new_values = new_values[new_values < 40].tolist()

    print(old_values)
    print(new_values)

    plt.figure(figsize=(24, 24))
    plt.margins(0, 0)
    plt.ylabel('Price')
    plt.title(f'Price Distribution in MTA_CHANGES')

    fig1, ax1 = plt.subplots()
    ax1.boxplot(old_values)

    fig2, ax2 = plt.subplots()
    ax2.boxplot(new_values)

    # plt.box(old_values)
    # plt.box(new_values)

    plt.draw()
    plt.savefig(f'{cfg.STORAGE_BASE_PATH_PLOTS}\\sap\\sap_price_distribution_changes.pdf', format='pdf',
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    generate_boxplot_changes_price_distribution()
