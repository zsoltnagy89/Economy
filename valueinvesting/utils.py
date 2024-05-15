import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import yfinance as yf
import datetime

def filter_raw_data(input=pd.DataFrame):
    '''
    Keep only the most important columns from the value evaluation point of view.
    '''
    filtered = input[
                [
                'date',
                'real_date',
                'shares',
                'revenue',
                'cogs',
                'gross_profit',
                'net_profit',
                'cash',
                'acc_rec',
                'inventory',
                'curr_assets',
                'goodwill',
                'intangible_assets',
                'total_assets',
                'acc_pay',
                'short_term_debt',
                'current_debt',
                'curr_liab',
                'long_term_debt',
                'total_liab',
                'cash_from_operating_activities',
                'capex'
                ]
            ].copy()
    return filtered


def daily_price(ticker, end, days_earlier=3, columns=['Close']):
    '''
    Returns a DataFrame of prices for a ticker from Yahoo Finance API
    The close date is excluded!!!!
    Minimum 3 days window due to weekends and holidays.
    '''
    result_series = []
    for timestamp in end:
        start = timestamp - datetime.timedelta(days=days_earlier)
        obj = yf.Ticker(ticker)
        data = obj.history(start=start, end=timestamp)[columns]
        result_series.append(data[columns].values.mean())
    return pd.Series(result_series).values

def replace_format_input(input=pd.DataFrame):
    '''
    replace - characters to 0
    add missing 0-s from the end
    drop out , as separators
    set datatype to integer
    '''
    for column in input.columns:
        if isinstance(input[column][0], str):
            # create empty list to add element
            result = []
            # itreate through the columns
            for elem in input[column]:
                # checkt the value contains a ,
                if ',' in elem:
                    # if the last part of string is shorter than 3 characters
                    original_value = elem.split(',')
                    if len(original_value[-1]) < 3:
                        # create new last element of original value
                        original_value[-1] = original_value[-1].ljust(3, '0')    
                        # recreate string
                        new_value = "".join(original_value)
                    else:
                        new_value = "".join(original_value)
                    # add merged element to list
                    result.append(new_value)
                elif elem == '-':
                    # replace - to 0
                    result.append('0')
                else:
                    # add don't modified values
                    result.append(elem)
            # overwrite column values and fix datatype
            input[column] = pd.Series(result).astype(int)
    return input

def convert_national_currency(input_data=pd.DataFrame, currency=pd.DataFrame):
    '''
    Convert columns into national currency, except dates, shares, rates and columns which are in USD.
    '''
    output_data = input_data.copy()
    for column in output_data.columns:
        if column not in ['shares', 'national_div', 'usd_div', 'usd_nat_currency', 'real_date', 'date']:
            output_data[column] = output_data[column].astype(float) * currency['usd_nat_currency']
    return output_data

def calculate_real_date(input):
    '''
    Calculate the estimated date when the quaterly report could be available, called real date.
    '''
    # if "real_date" not in input.columns: --> es hasznalhato amig megy az updateles
    if "real_date" not in input.columns:
        result = []
        for timestamp in input['date']:
            if timestamp.month == 12:
                result.append(timestamp + datetime.timedelta(days=45))
            else:
                result.append(timestamp + datetime.timedelta(days=21))
        input['real_date'] = result
    return input

def calculate_input_value_ratios(input=pd.DataFrame, report='Q'):
    '''
    Calculate EPS, Book Value per share, FCF and FCF per shares.
    '''
    # set conversion between half year and quarterly report
    if report == 'Q':
        multiplier = 4
    else:
        multiplier = 2
    # rcalculate TTM
    input['net_profit_ttm'] = input['net_profit'].rolling(multiplier).sum()
    input['cash_from_operating_activities_ttm'] = input['cash_from_operating_activities'].rolling(multiplier).sum()
    input['capex_ttm'] = input['net_profit'].rolling(multiplier).sum()
    # calculation
    input['eps'] = input['net_profit_ttm'] / input['shares'] # trailing twelve month
    input['bv_per_share'] = (input['total_assets']-input['total_liab']) / input['shares']
    input['fcf'] = input['cash_from_operating_activities_ttm'] - input['capex_ttm'] # trailing twelve month
    input['fcf_per_share'] = input['fcf'] / input['shares']
    # don't drop np.NaN --> those are replaced to 0-s in evaluate_performance()
    return input

def ratios_input_filter(input=pd.DataFrame):
    '''
    Filter out and keep only the Value ratios, revenue and date.
    '''
    ratios = input[
                [
                'date',
                'real_date',
                'revenue',
                'eps',
                'bv_per_share',
                'shares',
                'fcf',
                'fcf_per_share',
                'cash',
                'total_liab'
                ]
            ].copy()
    return ratios

def evaluate_performance(input=pd.DataFrame, output=pd.DataFrame, report='Q'):
    '''
    Calulate Financial ratios. Evaluate short-term, long-term debt, management performance and test economic moat.
    '''
    # set conversion between half year and quarterly report
    if report == 'Q':
        multiplier = 4
    else:
        multiplier = 2
    # calculate TTMs
    input['revenue_ttm'] = input['revenue'].rolling(multiplier).sum()
    input['cogs_ttm'] = input['cogs'].rolling(multiplier).sum()
    input['net_profit_ttm'] = input['net_profit'].rolling(multiplier).sum()
    # evauleat short term debt
    output['current_ratio'] = input['curr_assets'] / input['curr_liab']
    output['quick_ratio'] = (input['curr_assets'] - input['inventory']) / input['curr_liab']
    output['cash_ratio'] = input['cash'] / input['curr_liab']
    #evaluate long term debt
    output['debt_to_equity'] = input['total_liab'] / (input['total_assets'] - input['total_liab'])
    output['equity_ratio'] = (input['total_assets'] - input['total_liab']) / input['total_assets']
    output['debt_ratio'] = input['total_liab'] / input['total_assets']
    # evlauate management --> based on efficiency ratios
    output['acc_rec_ratio'] = input['revenue_ttm'] / input['acc_rec']
    output['acc_pay_ratio'] = (-1 * input['cogs_ttm']) / input['acc_pay']
    output['cash_turnover'] = input['revenue_ttm'] / input['cash']
    output['inventory_turnover'] = (-1 * input['cogs_ttm']) / input['inventory']
    # test economy moat
    output['gross_profit_margin'] = input['gross_profit'] / input['revenue']
    output['net_profit_margin'] = input['net_profit'] / input['revenue']
    output['roa'] = input['net_profit_ttm'] / input['total_assets']
    output['roe'] = input['net_profit_ttm'] / (input['total_assets'] - input['total_liab'])
    # replace possible Nan-s to 0
    for column in output.columns:
        output[column] = output[column].fillna(0)

    return output

def add_share_prices_to_value_ratios(share_name, data, ratios_nat_curr):
    '''
    Pull historocal weekly share prices and merge them with the value ratios dataframe.
    '''
    obj = yf.Ticker(share_name)
    share_price = obj.history(interval="1wk", start=data.date.min(), end=pd.Timestamp.now())
    # add new date column due to pd.merge_asof match
    share_price['real_date'] = share_price.index
    # remove localization (timezone) to let merge the two columns
    share_price['real_date'] = share_price['real_date'].dt.tz_localize(None)
    # add share price column and keep the date and share prices
    share_price['share_price'] = share_price['Close']
    share_price = share_price[['real_date', 'share_price']]
    # merge weekly share prices into the value ratio DataFrame
    merged_nat_curr = pd.merge_asof(left=share_price, right=ratios_nat_curr, on='real_date', direction='backward')
    # drop rows with np.Nan and reset index
    merged_nat_curr = merged_nat_curr.dropna()
    merged_nat_curr = merged_nat_curr.reset_index(drop=True)
    return merged_nat_curr

def price_ratios(input=pd.DataFrame, report='Q'):
    '''
    Calculate Value metrics from quaterly data. The original metrics have been develoed to annual data. I use quaterly data.
    '''
    # set conversion between half year and quarterly report
    if report == 'Q':
        multiplier = 4
    else:
        multiplier = 2
    # calculation
    input['pe_ratio'] = input['share_price'] / input['eps'] # previously multiplied by X (quaterly correction)
    input['pb_ratio'] = input['share_price'] / input['bv_per_share'] # don't need to quaterly correct (Income Statement data)
    input['ps_ratio'] = (input['share_price'] * input['shares']) / (input['revenue'] * multiplier) # quaterly corrected here --> multipled by X
    input['ev_revenue'] = ((input['share_price'] * input['shares']) + input['total_liab'] - input['cash']) / (input['revenue'] * multiplier)
    input['pfcf_ratio'] = (input['share_price'] * input['shares']) / input['fcf']  # previously multiplied by X (quaterly correction)
    return input

def get_historical_currency_rate(currency_pair, merged_nat_curr):
    '''
    Download historical USD-national currency rates from the earliest report date to today.
    '''
    obj = yf.Ticker(currency_pair)
    #get daily USD - national currency rates
    usd_nat_curr = obj.history(interval="1d", start=merged_nat_curr['real_date'].min(), end=pd.Timestamp.now())
    # bring the index as a colum and drop time zone data
    usd_nat_curr['date'] = usd_nat_curr.index
    usd_nat_curr['date'] = usd_nat_curr['date'].dt.tz_localize(None)
    # create column from Close data
    usd_nat_curr['currency_rate'] = usd_nat_curr['Close']
    return usd_nat_curr

def get_historical_share_dividend(share_name, merged_nat_curr):
    '''
    Download historical share prices and dividend yield data.
    '''
    obj2 = yf.Ticker(share_name)
    ticker_all_price = obj2.history(interval="1d", start=merged_nat_curr['real_date'].min(), end=pd.Timestamp.now())
    #use index as date and drop timezone data
    ticker_all_price['date'] = ticker_all_price.index
    ticker_all_price['date'] = ticker_all_price['date'].dt.tz_localize(None)
    # get share prices & caclulate dividend yield
    ticker_all_price['share_price'] = ticker_all_price['Close']
    ticker_all_price['dividend_yield'] = ticker_all_price['Dividends'] * 100 / ticker_all_price['Close']
    # plot historical dividend yields
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(ticker_all_price.index, ticker_all_price['dividend_yield'], color='k', label=share_name)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Dividend Yield (%)', color='k')
    plt.legend()
    plt.show()
    return ticker_all_price

def get_currency_share_price_correlation(share_name, usd_nat_curr, ticker_all_price):
    '''
    Calculate Pearson's correlation coefficient between share price and USD - national currency rate.
    Plot the two variables on time.
    '''
    result = pd.merge_asof(left=usd_nat_curr, right=ticker_all_price, on='date')
    result.index = result['date']
    result = result[['currency_rate', 'share_price']]
    # print correlation coefficient
    print(result.corr()['share_price'])
    # plot the time serieses
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    ax1.plot(usd_nat_curr.index, usd_nat_curr['Close'], color='k', label='USD / national currency')
    ax2.plot(ticker_all_price.index, ticker_all_price['Close'], color='b', label=share_name)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Currency rate (1 USD to X national currency)', color='k')
    ax2.set_ylabel('Share price (national)', color='b')
    plt.legend()
    plt.show()

def get_percentiles(input=pd.Series):
    # get last value of the series
    last_value = input.iloc[-1]
    # order Series element and get reordered index of the current value
    index = input.sort_values().values.searchsorted(last_value)
    # calculate the percentile of the last/current value
    position = (index + 1) / len(input) * 100
    return round(position)

def plot_histogram_value_parameters(input_df=pd.DataFrame, extra_parameters=[], owned_shares=pd.DataFrame):
    # select share's bought list
    bought_date = list(owned_shares['date'])
    # predifined value parameters to plot
    selected_parameters = ['roa', 'roe', 'pe_ratio', 'pb_ratio', 'ps_ratio', 'ev_revenue', 'debt_to_equity', 'current_ratio']
    # add extra user requested value parameters
    selected_parameters = selected_parameters + extra_parameters
    # replace infinite values to avoid error
    input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # plot histograms
    for column in selected_parameters:
        try:
            #plot data
            plt.hist(input_df[column].values, bins=30, edgecolor='black', color='gray')
            # plot values related to buying date 
            for date in bought_date:
                # filter input dataframe and keep the closest row to the timestamp
                input_df_slice = input_df.loc[(input_df['real_date'] >= date - datetime.timedelta(days=4)) & (input_df['real_date'] <= date + datetime.timedelta(days=4))]
                # plot the specific parameter related to the stock buying date
                plt.axvline(input_df_slice[column].values.min(), color='red', linewidth=2, label='Owned Shares')
            # plot percentiles and curren values
            plt.axvline(input_df[column].iloc[-1], color='k', linestyle='dotted', linewidth=2, label='Current Value')
            plt.axvline(input_df[column].quantile(0.1), color='green', linestyle='dashed', linewidth=1, label='P10')
            plt.axvline(input_df[column].quantile(0.3), color='green', linestyle='dashed', linewidth=1, label='P30')
            plt.axvline(input_df[column].quantile(0.5), color='blue', linestyle='dashed', linewidth=1, label='Median')
            plt.axvline(input_df[column].quantile(0.7), color='orange', linestyle='dashed', linewidth=1, label='P70')
            plt.axvline(input_df[column].quantile(0.9), color='red', linestyle='dashed', linewidth=1, label='P90')
            plt.xlabel(column.capitalize())
            plt.ylabel('Frequency')
            plt.suptitle(str(column.capitalize()) + ' percentile currently is ' + str(get_percentiles(input_df[column]))+ '% - ' + str(datetime.date.today()))
            plt.legend()
            plt.show()
        except:
            print(column + " diagram is missing due to error.")

def list_intersect(input1, input2):
    '''
    Create a list of common elements from 2 lists.
    '''
    result = [x for x in input1 if x in input2]
    return result

def get_historical_analouges(input_df, ticker_all_price, share_name, tolerance=0.2, owned_shares=pd.DataFrame):
    # select share's bought list
    bought_date = list(owned_shares['date'])
    # collect dates when the given value variable was between tolearance limit to current one
    roe = input_df[(input_df['roe']<= input_df['roe'].iloc[-1] * (1+tolerance)) & (input_df['roe']>= input_df['roe'].iloc[-1] * (1- tolerance))]['date'].values
    pbr = input_df[(input_df['pb_ratio']<= input_df['pb_ratio'].iloc[-1] * (1+tolerance)) & (input_df['pb_ratio']>= input_df['pb_ratio'].iloc[-1] * (1- tolerance))]['date'].values
    psr = input_df[(input_df['ps_ratio']<= input_df['ps_ratio'].iloc[-1] * (1+tolerance)) & (input_df['ps_ratio']>= input_df['ps_ratio'].iloc[-1] * (1- tolerance))]['date'].values
    de = input_df[(input_df['debt_to_equity']<= input_df['debt_to_equity'].iloc[-1] * (1+tolerance)) & (input_df['debt_to_equity']>= input_df['debt_to_equity'].iloc[-1] * (1- tolerance))]['date'].values
    curr = input_df[(input_df['current_ratio']<= input_df['current_ratio'].iloc[-1] * (1+tolerance)) & (input_df['current_ratio']>= input_df['current_ratio'].iloc[-1] * (1- tolerance))]['date'].values
    
    # create intersect of the above timestamp list
    test1 =  list_intersect(roe, pbr)
    test2 =  list_intersect(test1, psr)
    test3 =  list_intersect(test2, de)
    test4 =  list_intersect(test3, curr)
    print(len(test4), 'timestemos have been found!')

    # plot results
    for column in ['roe', 'pb_ratio', 'ps_ratio', 'ev_revenue', 'debt_to_equity', 'current_ratio']:
        fig, ax1 = plt.subplots(figsize=(15, 6))
        ax2 = ax1.twinx()
        ax1.plot(input_df['real_date'], input_df[column], color='k', label=column)
        ax2.plot(ticker_all_price.index, ticker_all_price['Close'], color='b', label=share_name)
        for date in bought_date:
            plt.axvline(date, color='green', linewidth=2, label='Owned stock')
        for timestamp in test4:     
            plt.axvline(timestamp, color='red', linestyle='dashed', linewidth=1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel(column.capitalize(), color='k')
        ax2.set_ylabel('Share price (national currency)', color='b')
        plt.title(str(column.capitalize()) + ' - ' + str(datetime.date.today()))
        plt.legend()
        plt.show()

def get_country_stocks(country='norway'):
    '''
    Get the basic info related to a specific country's stocks.
    '''
    input_route = f"../data/extras/countries_stocks.csv"
    stock_data = pd.read_csv(input_route)
    # optional filter
    stock_data = stock_data[stock_data['country'] == country]
    return stock_data

def create_summary_value_table(input=pd.DataFrame):
    '''
    It's a mass that calculate the value ratio summary table.
    '''
    # output lists
    countr = []
    ticker = []
    sec = []
    industr = []
    roe_ratio = []
    pb_ratio = []
    ps_ratio = []
    evrv_ratio = []
    de_ratio = []
    current_ratio = []
    roe_perc = []
    pb_perc = []
    ps_perc = []
    evrv_perc = []
    de_perc = []
    current_perc = []
    # calculate value ratios and percentiles
    for index in range(len(input)):
        # set boundary conditions
        evaluate_last_X_years = True
        X=10
        currency_pair = input.currency_pair[index]
        numbers_in_currency = input.numbers_in_currency[index]
        share_name = input.share_name[index]
        country = input.country[index]
        sector = input.sector[index]
        industry = input.industry[index]
        # read accounting data
        try:
            route = f"../data/input/countries/{country}/{share_name}_data.csv"
            data = pd.read_csv(route, sep=';', parse_dates=['date', 'real_date'])
            data = replace_format_input(data)
        except:
            print('Unsuccessfull data load! Relative route hardcoded in fucntion, it could be the issue!')
        # filter out unnecessary old data
        if evaluate_last_X_years:
            data = data[data['date'] > datetime.datetime.today() - datetime.timedelta(days=X*366+93)]
            data = data.reset_index(drop=True)
        # calculate quarterly report availablilty date
        try:
            data = calculate_real_date(data)
        except:
            print('Quarterly availability date calculation error!')
        # drop unnecesarry columns
        try:
            data = filter_raw_data(data)
        except:
            print('Column filter unsuccessful!')
        # pull historical USD/national currency rates and add to dataframe
        try:
            data['usd_nat_currency'] = daily_price(
            ticker=currency_pair,
            end=data['date'],
            days_earlier=90
            )
            # drop rows, when USD rates wasn't available
            data = data[data['usd_nat_currency'].notna()]
        except:
            print('Historical USD pull error!')
        # convert columns into national currency if necessary
        try:
            if numbers_in_currency == 'USD':
                data_nat_curr = convert_national_currency(input_data=data, currency=data)
            else:
                data_nat_curr = data.copy()
        except:
            print('Column USD to national currency conversion error!')
        # filter unnecessary columns
        filtered_nat_curr = calculate_input_value_ratios(data_nat_curr)
        #calculate input to value ratios
        ratios_nat_curr = ratios_input_filter(filtered_nat_curr)
        ratios_nat_curr = evaluate_performance(input=filtered_nat_curr, output=ratios_nat_curr)
        # pull weekly share prices and merge with the value ratios
        merged_nat_curr = add_share_prices_to_value_ratios(share_name, data, ratios_nat_curr)
        # calculate value ratios
        merged_nat_curr = price_ratios(merged_nat_curr)
        # add results to lists
        try:
            countr.append(country)
            ticker.append(share_name)
            sec.append(sector)
            industr.append(industry)
            roe_ratio.append(merged_nat_curr['roe'].iloc[-1])
            pb_ratio.append(merged_nat_curr['pb_ratio'].iloc[-1])
            ps_ratio.append(merged_nat_curr['ps_ratio'].iloc[-1])
            evrv_ratio.append(merged_nat_curr['ev_revenue'].iloc[-1])
            de_ratio.append(merged_nat_curr['debt_to_equity'].iloc[-1])
            current_ratio.append(merged_nat_curr['current_ratio'].iloc[-1])
            roe_perc.append(get_percentiles(merged_nat_curr['roe']))
            pb_perc.append(get_percentiles(merged_nat_curr['pb_ratio']))
            ps_perc.append(get_percentiles(merged_nat_curr['ps_ratio']))
            evrv_perc.append(get_percentiles(merged_nat_curr['ev_revenue']))
            de_perc.append(get_percentiles(merged_nat_curr['debt_to_equity']))
            current_perc.append(get_percentiles(merged_nat_curr['current_ratio']))
        except:
            print('Value adding to list error!')
        print(str(share_name) + ' has been finished successfuly!')
    # calculate result
    result = pd.DataFrame(
            list(zip(countr, ticker, sec, industr, roe_ratio, pb_ratio, ps_ratio, evrv_ratio, de_ratio, current_ratio, roe_perc, pb_perc, ps_perc, evrv_perc, de_perc, current_perc)),
            columns =['country', 'ticker', 'sector', 'industry','roe_ratio', 'pb_ratio', 'ps_ratio', 'evrv_ratio', 'de_ratio', 'current_ratio', 'roe_perc', 'pb_perc', 'ps_perc', 'evrv_perc', 'de_perc', 'current_perc']
        )
    return result

def utility_evaluation(input_df=pd.DataFrame, extra_parameters=[], owned_shares=pd.DataFrame):
    '''
    This function creates 2 plots. The first comes with the price, P/B, P/S and EV/Rev time series,
    the second is the distribution of normaled weekly values with the 200 days moving average.
    It's recommended to use in case of shares with monotonous increasing price.
    '''
    # predifined parameters to plot - extendable
    selected_parameters = ['share_price', 'pb_ratio', 'ps_ratio', 'ev_revenue']
    # add extra user requested value parameters
    selected_parameters = selected_parameters + extra_parameters

    # calculate index list related to owned shares
    index_list = []
    for date in owned_shares['date']:
        index = -1 * len(input_df.loc[input_df['real_date'] > date])
        index_list.append(index)

    for elem in selected_parameters:
        raw = input_df[elem]
        mva_50 = input_df[elem].rolling(50).mean()
        mva_100 = input_df[elem].rolling(100).mean()
        mva_200 = input_df[elem].rolling(200).mean()
        diff_200 = (raw - mva_200) / mva_200

        # time series splited plot
        # fist line:
        plt.figure(figsize=(15,8))
        plt.subplot(211)
        plt.plot(raw,label= elem.capitalize())
        plt.plot(mva_50,label= 'MA 50 days')
        plt.plot(mva_100,label= 'MA 100 days')
        plt.plot(mva_200,label= 'MA 200 days')
        plt.xlim(left=200, right=diff_200.index[-1])
        for i in index_list:
            plt.axvline(len(mva_200) + i, color='green', linewidth=2, label='Owned stock')
        plt.legend(loc='best')
        # second block
        plt.subplot(212)
        plt.plot(diff_200,color='black', linewidth=1, label= '(' + str(elem.capitalize()) + ' - MVA_200) / MVA_200')
        plt.xlim(left=200, right=diff_200.index[-1])
        for i in index_list:
            plt.axvline(len(mva_200) + i, color='green', linewidth=2, label='Owned stock')
        plt.legend(loc='best')
        plt.xlabel('Week number')
        plt.show()

        # histogram
        plt.hist(diff_200.values, bins=30, edgecolor='black', color='gray')
        plt.axvline(diff_200.iloc[-1], color='k', linestyle='dotted', linewidth=2, label='Current Value')
        plt.axvline(diff_200.quantile(0.1), color='green', linestyle='dashed', linewidth=1, label='P10')
        plt.axvline(diff_200.quantile(0.3), color='green', linestyle='dashed', linewidth=1, label='P30')
        plt.axvline(diff_200.quantile(0.5), color='blue', linestyle='dashed', linewidth=1, label='Median')
        plt.axvline(diff_200.quantile(0.7), color='orange', linestyle='dashed', linewidth=1, label='P70')
        plt.axvline(diff_200.quantile(0.9), color='red', linestyle='dashed', linewidth=1, label='P90')
        for i in index_list:
                # plot the specific parameter related to the stock buying date
                plt.axvline(diff_200.iloc[len(diff_200) + i], color='red', linewidth=2, label='Owned Shares')
        plt.xlabel(elem.capitalize())
        plt.ylabel('Frequency')
        plt.suptitle('(' + str(elem.capitalize()) + ' - MVA 200) / MVA 200 percentile currently is ' + str(get_percentiles(diff_200))+ '% - ' + str(datetime.date.today()))
        plt.legend()
        plt.show()