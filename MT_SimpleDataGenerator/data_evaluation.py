import matplotlib.pyplot as plt
import numpy as np
import database_config
import config as cfg


database_config.db.load(cfg.STORAGE_BASE_PATH_SIMULATED_DATA)
changes = database_config.db.get_table('MTA_CHANGES').get_data()
customers = database_config.db.get_table('MST_CUSTOMERS').get_data()
salespersons = database_config.db.get_table('MST_SALESPERSONS').get_data()
sales = database_config.db.get_table('TRC_SALES').get_data()
sale_products = database_config.db.get_table('TRM_SALE_PRODUCTS').get_data()


def _group_to_plot(grouped_df, plot_name, xlabel, title, ylabel):
    print(f'generating plot {plot_name}')
    labels_x = []
    bar_names = []

    # get label/bar names
    for index in grouped_df.index:
        if index[0] not in labels_x:
            labels_x.append(index[0])

        if index[1] not in bar_names:
            bar_names.append(index[1])

    # generate plot
    plt.figure(figsize=(24, 12))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    width = 0.35
    ind = np.arange(len(labels_x))
    for index, bar in enumerate(bar_names):
        series_data = []

        for label in labels_x:
            if (label, bar) in grouped_df:
                series_data.append(grouped_df[(label, bar)])
            else:
                series_data.append(0)

        plt.bar(ind + index * width, series_data, label=bar, width=width)

    plt.xticks(ticks=ind, labels=labels_x)
    # plt.show()
    plt.draw()
    plt.savefig(f'{cfg.STORAGE_BASE_PATH_PLOTS}\\misc\\{plot_name}', bbox_inches='tight')
    plt.close()


def generate_misc_changes_plots():
    data = changes.groupby(by=['change_type', 'is_fraud'])['record_id'].count()
    _group_to_plot(grouped_df=data,
                   plot_name='changes_by_type_and_fraud',
                   xlabel='MTA_CHANGES.change_type',
                   title='Changes per MTA_CHANGES.change_type',
                   ylabel='Number of Changes')

    data = changes.groupby(by=['salesperson_id', 'is_fraud'])['record_id'].count()
    _group_to_plot(grouped_df=data,
                   plot_name='changes_by_salesperson_id_and_fraud',
                   xlabel='MTA_CHANGES.salesperson_id',
                   title='Changes per MTA_CHANGES.salesperson_id',
                   ylabel='Number of Changes')

    data = changes.groupby(by=['table_column_ref', 'is_fraud'])['record_id'].count()
    _group_to_plot(grouped_df=data,
                   plot_name='changes_by_table_column_ref_and_fraud',
                   xlabel='MTA_CHANGES.table_column_ref',
                   title='Changes per MTA_CHANGES.table_column_ref',
                   ylabel='Number of Changes')


def generate_plot_salesperson_changes_over_time():
    events = []

    for index, salesperson in salespersons.iterrows():
        print(f'generating plot for {salesperson["name"]}')

        changes_salesperson = changes[(changes['salesperson_id'] == index)]
        x_times = []
        event_times = []
        y_number_of_changes = []

        for time in range(0, cfg.SIMULATION_END_TIME):
            changes_at_time = changes_salesperson[(changes_salesperson['timestamp'] == time)]
            x_times.append(time)
            y_number_of_changes.append(len(changes_at_time))

            if len(changes_at_time) > 0:
                event_times.append(time)

        events.append(event_times)
        plt.figure(figsize=(24, 12))
        plt.xlabel('Time')
        plt.ylabel('Number of Changes')
        plt.title(f'Changes of "{salesperson["name"]}" over Time')
        plt.xlim(left=0, right=cfg.SIMULATION_END_TIME)
        plt.xticks(np.arange(min(x_times), max(x_times) + 1, 30))
        plt.bar(x_times, y_number_of_changes)
        # plt.show()
        plt.draw()
        plt.savefig(f'{cfg.STORAGE_BASE_PATH_PLOTS}\\changes_per_salesperson\\changes_per_salesperson_{salesperson["name"].lower().replace(" ", "_")}', bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(24, 12))
    plt.margins(0, 0)
    plt.xlabel('Time')
    plt.ylabel('Salesperson')
    plt.title(f'Activity of Salespersons over Time')
    plt.yticks(np.arange(0, cfg.INIT_GEN_SALESPERSON_COUNT))
    plt.xticks(np.arange(0, cfg.SIMULATION_END_TIME + 1, 30))
    plt.eventplot(events, orientation='horizontal', linelengths=0.8, linewidths=0.8)
    plt.vlines([355, 720], 0, cfg.INIT_GEN_SALESPERSON_COUNT, colors=['r', 'r'], linestyles=['dotted', 'dotted'])
    # plt.show()
    plt.draw()
    plt.savefig(f'{cfg.STORAGE_BASE_PATH_PLOTS}\\changes_per_salesperson\\activity_per_salesperson', bbox_inches='tight')
    plt.close()


def generate_plot_customer_purchases_over_time():
    events = []

    for index, customer in customers.iterrows():
        print(f'generating plot for {customer["name"]}')
        customer_events = []
        x_times = []
        y_number_purchased_products = []

        for time in range(0, cfg.SIMULATION_END_TIME):
            purchases = sales[(sales['customer_id'] == index) & (sales['timestamp'] == time)]
            x_times.append(time)

            if len(purchases) > 0:
                number_of_products = 0

                for sale_index, sale in purchases.iterrows():
                    purchased_products = sale_products[sale_products['sale_id'] == sale_index]

                    for _, sale_product in purchased_products.iterrows():
                        number_of_products += int(sale_product['amount'])

                y_number_purchased_products.append(number_of_products)
                customer_events.append(time)
            else:
                y_number_purchased_products.append(0)

        events.append(customer_events)
        plt.figure(figsize=(24, 12))
        plt.xlabel('Time')
        plt.ylabel('Number of Products purchased')
        plt.title(f'Purchases of "{customer["name"]}" over Time')
        plt.xlim(left=0, right=cfg.SIMULATION_END_TIME)
        plt.xticks(np.arange(min(x_times), max(x_times) + 1, 30))
        plt.bar(x_times, y_number_purchased_products)
        # plt.show()
        plt.draw()
        plt.savefig(f'{cfg.STORAGE_BASE_PATH_PLOTS}\\purchases_per_customer\\purchases_per_customer_{customer["name"].lower().replace(" ", "_")}', bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(24, 24))
    plt.margins(0, 0)
    plt.xlabel('Time')
    plt.ylabel('Customer')
    plt.title(f'Activity of Customers over Time')
    plt.yticks(np.arange(0, cfg.INIT_GEN_CUSTOMER_COUNT))
    plt.xticks(np.arange(0, cfg.SIMULATION_END_TIME + 1, 30))
    plt.eventplot(events, orientation='horizontal', linelengths=0.9, linewidths=1)
    plt.vlines([355, 720], 0, cfg.INIT_GEN_CUSTOMER_COUNT, colors=['r', 'r'], linestyles=['dotted', 'dotted'])
    plt.hlines([x - 0.5 for x in range(10, cfg.INIT_GEN_CUSTOMER_COUNT, 10)], 0, cfg.SIMULATION_END_TIME, colors='k')
    # plt.show()
    plt.draw()
    plt.savefig(f'{cfg.STORAGE_BASE_PATH_PLOTS}\\purchases_per_customer\\activity_per_customer', bbox_inches='tight')
    plt.close()


def generate_plot_prices_over_time():
    for index, product in database_config.db.get_table('MST_PRODUCTS').get_data().iterrows():
        print(f'generating plot for {product["name"]}')
        product_changes = changes[(changes['record_id'] == index) & (changes['table_column_ref'] == 'MST_PRODUCTS.price')]
        x_times = []
        y_prices = []
        x_fraud_points = []
        y_fraud_points = []

        for _, change_record in product_changes.iterrows():
            if len(y_prices) == 0:
                x_times.append(0)
                y_prices.append(change_record['old_value'])

            if change_record['timestamp'] - 1 not in x_times:
                x_times.append(change_record['timestamp'] - 1)
                y_prices.append(float(change_record['old_value']))

            x_times.append(change_record['timestamp'])
            y_prices.append(float(change_record['new_value']))

            if change_record['is_fraud']:
                x_fraud_points.append(change_record['timestamp'])
                y_fraud_points.append(float(change_record['new_value']))

        if 0 not in x_times:
            x_times.append(0)
            y_prices.append(product['price'])

        if cfg.SIMULATION_END_TIME not in x_times:
            x_times.append(cfg.SIMULATION_END_TIME)
            y_prices.append(product['price'])

        plt.figure(figsize=(24, 12))
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'Price of "{product["name"]}" over Time')
        plt.xlim(left=0, right=cfg.SIMULATION_END_TIME)
        plt.xticks(np.arange(min(x_times), max(x_times) + 1, 30))
        plt.plot(x_times, y_prices)
        plt.scatter(x_fraud_points, y_fraud_points, marker='o', c='orange')
        plt.hlines(y_prices[0], 0, cfg.SIMULATION_END_TIME, colors=['g'], linestyles=['dashed'])
        plt.vlines([355, 720], 0, max(y_prices), colors=['r', 'r'], linestyles=['dotted', 'dotted'])
        # plt.show()
        plt.draw()
        plt.savefig(f'{cfg.STORAGE_BASE_PATH_PLOTS}\\price_over_time\\price_over_time_{product["name"].lower().replace(" ", "_")}', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    generate_misc_changes_plots()
    generate_plot_customer_purchases_over_time()
    generate_plot_salesperson_changes_over_time()
    generate_plot_prices_over_time()