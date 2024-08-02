import pandas as pd
import math
from monte_carlo import generate_monete_carlo_simulation
# from extra import 
import streamlit as st

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def one_contract_eq(df: pd.DataFrame) -> pd.DataFrame:
    cum_profit_points = 0
    cum_profit_precent = 0
    cum_profit_usd = 0

    equity_curve = pd.DataFrame()
    is_long = None
    entry_price = 0

    for i in df.index:
        action = df["Action"][i]
        avg_price = df["Avg. price"][i]
        time = df['Time'][i]
        point_value = df["PointValue"][i]

        if action == "Buy":
            entry_price = avg_price
            is_long = True

        if action == "Sell" and is_long and avg_price != 0:
            # Calculate the profit of each trade:
            trade_points = avg_price - entry_price
            trade_precent = ( (avg_price - entry_price) / entry_price ) * 100  
            trade_usd = trade_points * point_value

            # Cummulative the trades
            cum_profit_points += trade_points
            cum_profit_precent += trade_precent
            cum_profit_usd += trade_usd
        
            new_row = pd.DataFrame({'Time': [time], 
                                    'cum_profit_points': [cum_profit_points], 
                                    'cum_profit_precent': [cum_profit_precent], 
                                    'cum_profit_usd': [cum_profit_usd], 
                                    "single_trade_points_profit": [trade_points],
                                    "single_trade_precent_profit": [trade_precent],
                                    "single_trade_usd_profit": [trade_usd],
                                    "TimeFrame": [df["TimeFrame"][i]],
                                    "Instrument": [df["Instrument"][i]]})
            
            equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

        if action == "SellShort":
            entry_price = avg_price
            is_long = False

        if action == "BuyToCover" and not is_long and avg_price != 0:
            # Calculate the profit of each trade:
            trade_points = entry_price - avg_price
            trade_precent = ( (entry_price - avg_price) / avg_price ) * 100  
            trade_usd = trade_points * point_value
            
            # Cummulative the trades
            cum_profit_points += trade_points
            cum_profit_precent += trade_precent
            cum_profit_usd += trade_usd
            
            cum_profit_points += trade_points
            new_row = pd.DataFrame({'Time': [time], 
                                    'cum_profit_points': [cum_profit_points], 
                                    'cum_profit_precent': [cum_profit_precent], 
                                    'cum_profit_usd': [cum_profit_usd], 
                                    "single_trade_points_profit": [trade_points],
                                    "single_trade_precent_profit": [trade_precent],
                                    "single_trade_usd_profit": [trade_usd],
                                    "TimeFrame": [df["TimeFrame"][i]],
                                    "Instrument": [df["Instrument"][i]]})

            equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

    return equity_curve
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fixed_capital_money_managment(df: pd.DataFrame, fixed_unit, balance, max_contracts = 100):
    """
    Fixed capital is a money managment strategy that:
    - Will trade one contract for a fixed unit of capital 
    - E.g: If the fixed unit of capital is $15,000 and your account balance is $20,000, you would trade one contract.
    - As the account balance increases, the contract's number should increase as well, and vica versa. 
    
    How to get the fixed unit of capital ?
    - fixed_unit = Largest DD / Percentage blowtorch risk

        - The Percentage blowtorch risk refers to how much pain you can bear, 
           or how much of your account balance you are comfortable losing in percentage terms.

    """
    cum_profit_points = 0
    cum_profit_precent = 0
    cum_profit_usd = 0
    

    equity_curve = pd.DataFrame(columns=['Time', 'cum_profit_points', 'cum_profit_usd', 'cum_profit_precent'])
    is_long = None
    entry_price = 0

    entry_price = 0
   

    for i in df.index:
        action = df["Action"][i]
        avg_price = df["Avg. price"][i]
        time = df['Time'][i]
        point_value = df["PointValue"][i]
        order_name = df["Name"][i]

        if order_name == "Stop loss":
            stop_price = df["Stop"][i] 

        if action == "Buy":
            entry_price = avg_price
            is_long = True

        if action == "Sell" and is_long and avg_price != 0:
            trade_diff = avg_price - entry_price    
            # Position size calculation
            position_size = min(math.floor((balance) / ( fixed_unit )),max_contracts)
            if position_size < 1: position_size = 1   
            
            # Calculate the profit of each trade:
            trade_points = trade_diff * position_size
            trade_precent = ( ( trade_diff / entry_price ) * 100 ) * position_size  
            trade_usd = trade_diff  * point_value * position_size 

            balance+=trade_usd

            # Cummulative the trades
            cum_profit_points += trade_points
            cum_profit_precent += trade_precent
            cum_profit_usd += trade_usd
        
            new_row = pd.DataFrame({'Time': [time], 
                                    'cum_profit_points': [cum_profit_points], 
                                    'cum_profit_precent': [cum_profit_precent], 
                                    'cum_profit_usd': [cum_profit_usd], 
                                    "single_trade_points_profit": [trade_points],
                                    "single_trade_precent_profit": [trade_precent],
                                    "single_trade_usd_profit": [trade_usd],
                                    "TimeFrame": [df["TimeFrame"][i]],
                                    "Instrument": [df["Instrument"][i]]})
            equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

        if action == "SellShort":
            entry_price = avg_price
            is_long = False

        if action == "BuyToCover" and not is_long and avg_price != 0:
            trade_diff = entry_price - avg_price

            # Position size calculation
            position_size = min(math.floor((balance) / ( fixed_unit)),max_contracts)
            if position_size < 1: position_size = 1

            # Calculate the profit of each trade:
            trade_points = trade_diff * position_size
            trade_precent = ( ( trade_diff / avg_price ) * 100 ) * position_size  
            trade_usd = trade_diff * point_value * position_size 

            # Update the balance
            balance+=trade_usd
            
            # Cummulative the trades
            cum_profit_points += trade_points
            cum_profit_precent += trade_precent
            cum_profit_usd += trade_usd
            
            cum_profit_points += trade_points
            new_row = pd.DataFrame({'Time': [time], 
                                    'cum_profit_points': [cum_profit_points], 
                                    'cum_profit_precent': [cum_profit_precent], 
                                    'cum_profit_usd': [cum_profit_usd], 
                                    "single_trade_points_profit": [trade_points],
                                    "single_trade_precent_profit": [trade_precent],
                                    "single_trade_usd_profit": [trade_usd],
                                    "TimeFrame": [df["TimeFrame"][i]],
                                    "Instrument": [df["Instrument"][i]]})
            equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

    return equity_curve  
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fixed_ratio_money_managment(df: pd.DataFrame, delta = 50, max_contracts = 100) -> pd.DataFrame:
    """
    The fixed ratio is a money managment strategy that calculate the number of position size
    based on 'delta'. 
    
    Next account level = Current account level + (current contract number * Delta)

    - The idea is simple: each contract need to make profit that equal to delta.
    - The higher the delta, the more convervative the trader. 
    - The Delta should drive from the Drawdown of the system:
        Delta = Drawdown + Initial Margin
    
     Using an $18,000 for  delta value example, 
     means that you should not increase to your next contract level until every 
     current contract you’re trading has been able to contribute $18,000 in profit.

    
    """
    cum_profit_points = 0
    cum_profit_precent = 0
    cum_profit_usd = 0
    

    equity_curve = pd.DataFrame(columns=['Time', 'cum_profit_points', 'cum_profit_usd', 'cum_profit_precent'])
    is_long = None
    entry_price = 0
    entry_price = 0

    for i in df.index:
        action = df["Action"][i]
        avg_price = df["Avg. price"][i]
        time = df['Time'][i]
        point_value = df["PointValue"][i]
        order_name = df["Name"][i]

        if order_name == "Stop loss":
            stop_price = df["Stop"][i] 

        if action == "Buy":
            entry_price = avg_price
            is_long = True

        if action == "Sell" and is_long and avg_price != 0:
            
            trade_diff = avg_price - entry_price 

            # Position size calculation
            position_size = min(math.floor(cum_profit_usd / delta) + 1 if cum_profit_usd >= delta else 1, max_contracts)   
            if position_size < 1: position_size = 1
            

            # Calculate the profit of each trade:
            trade_points = trade_diff * position_size
            trade_precent = ( ( trade_diff / entry_price ) * 100 ) * position_size  
            trade_usd = trade_diff * point_value  * position_size

            # Cummulative the trades
            cum_profit_points += trade_points
            cum_profit_precent += trade_precent
            cum_profit_usd += trade_usd
        
            new_row = pd.DataFrame({'Time': [time], 
                                    'cum_profit_points': [cum_profit_points], 
                                    'cum_profit_precent': [cum_profit_precent], 
                                    'cum_profit_usd': [cum_profit_usd], 
                                    "single_trade_points_profit": [trade_points],
                                    "single_trade_precent_profit": [trade_precent],
                                    "single_trade_usd_profit": [trade_usd],
                                    "TimeFrame": [df["TimeFrame"][i]],
                                    "Instrument": [df["Instrument"][i]]})
            equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)
            

        if action == "SellShort":
            entry_price = avg_price
            is_long = False

        if action == "BuyToCover" and not is_long and avg_price != 0:
            
            trade_diff = entry_price - avg_price
            # Position size calculation
            position_size = min(math.floor(cum_profit_usd / delta) + 1 if cum_profit_usd >= delta else 1, max_contracts)   
            
            if position_size < 1: position_size = 1


            # Calculate the profit of each trade:
            trade_points = trade_diff * position_size
            trade_precent = ( ( trade_diff / avg_price ) * 100 ) * position_size  
            trade_usd = trade_diff * point_value  * position_size


            
            # Cummulative the trades
            cum_profit_points += trade_points
            cum_profit_precent += trade_precent
            cum_profit_usd += trade_usd
            
            cum_profit_points += trade_points
            new_row = pd.DataFrame({'Time': [time], 
                                    'cum_profit_points': [cum_profit_points], 
                                    'cum_profit_precent': [cum_profit_precent], 
                                    'cum_profit_usd': [cum_profit_usd], 
                                    "single_trade_points_profit": [trade_points],
                                    "single_trade_precent_profit": [trade_precent],
                                    "single_trade_usd_profit": [trade_usd],
                                    "TimeFrame": [df["TimeFrame"][i]],
                                    "Instrument": [df["Instrument"][i]]})
            equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)
            


    return equity_curve
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fixed_unit_money_managment(df: pd.DataFrame, fixed_units: int, initial_balance: float, max_contracts: int = 100) -> pd.DataFrame:
    cum_profit_points = 0
    cum_profit_precent = 0
    cum_profit_usd = 0

    equity_curve = pd.DataFrame()
    is_long = None
    entry_price = 0

    balance = initial_balance

    for i in df.index:
        action = df["Action"][i]
        avg_price = df["Avg. price"][i]
        time = df['Time'][i]
        point_value = df["PointValue"][i]
        order_name = df["Name"][i]

        if order_name == "Stop loss":
            stop_price = df["Stop"][i] 

        if action == "Buy":
            entry_price = avg_price
            is_long = True

        if action == "Sell" and is_long and avg_price != 0:
            trade_diff = avg_price - entry_price
            trade_risk = abs(entry_price - stop_price) * point_value

            if trade_risk != 0:
                dollar_risk_per_trade = balance / fixed_units
                position_size = min(math.floor(dollar_risk_per_trade / trade_risk), max_contracts)
                if position_size < 1: position_size = 1

                # Calculate the profit of each trade:
                trade_points = trade_diff * position_size
                trade_precent = (trade_diff / entry_price) * 100 * position_size  
                trade_usd = trade_diff * point_value * position_size

                balance += trade_usd

                # Cumulative the trades
                cum_profit_points += trade_points
                cum_profit_precent += trade_precent
                cum_profit_usd += trade_usd

                new_row = pd.DataFrame({
                    'Time': [time],
                    'cum_profit_points': [cum_profit_points],
                    'cum_profit_precent': [cum_profit_precent],
                    'cum_profit_usd': [cum_profit_usd],
                    "single_trade_points_profit": [trade_points],
                    "single_trade_precent_profit": [trade_precent],
                    "single_trade_usd_profit": [trade_usd],
                    "TimeFrame": [df["TimeFrame"][i]],
                    "Instrument": [df["Instrument"][i]]
                })
                equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

        if action == "SellShort":
            entry_price = avg_price
            is_long = False

        if action == "BuyToCover" and not is_long and avg_price != 0:
            trade_diff = entry_price - avg_price
            trade_risk = abs(entry_price - stop_price) * point_value

            if trade_risk != 0:
                dollar_risk_per_trade = balance / fixed_units
                position_size = min(math.floor(dollar_risk_per_trade / trade_risk), max_contracts)
                if position_size < 1: position_size = 1

                # Calculate the profit of each trade:
                trade_points = trade_diff * position_size
                trade_precent = (trade_diff / avg_price) * 100 * position_size  
                trade_usd = trade_diff * point_value * position_size

                balance += trade_usd

                # Cumulative the trades
                cum_profit_points += trade_points
                cum_profit_precent += trade_precent
                cum_profit_usd += trade_usd

                new_row = pd.DataFrame({
                    'Time': [time],
                    'cum_profit_points': [cum_profit_points],
                    'cum_profit_precent': [cum_profit_precent],
                    'cum_profit_usd': [cum_profit_usd],
                    "single_trade_points_profit": [trade_points],
                    "single_trade_precent_profit": [trade_precent],
                    "single_trade_usd_profit": [trade_usd],
                    "TimeFrame": [df["TimeFrame"][i]],
                    "Instrument": [df["Instrument"][i]]
                })
                equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

    return equity_curve
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def williams_fixed_risk_money_managment(df:pd.DataFrame, precent, balance, largest_loss, max_contracts = 100):
    """
    A money managment strategy that take in account the largest loss of the system.  
    """
    cum_profit_points = 0
    cum_profit_precent = 0
    cum_profit_usd = 0
    

    equity_curve = pd.DataFrame(columns=['Time', 'cum_profit_points', 'cum_profit_usd', 'cum_profit_precent'])
    is_long = None
    entry_price = 0

    entry_price = 0

    for i in df.index:
        action = df["Action"][i]
        avg_price = df["Avg. price"][i]
        time = df['Time'][i]
        point_value = df["PointValue"][i]
        order_name = df["Name"][i]

        if order_name == "Stop loss":
            stop_price = df["Stop"][i] 

        if action == "Buy":
            entry_price = avg_price
            is_long = True

        if action == "Sell" and is_long and avg_price != 0:

            trade_diff = avg_price - entry_price
            
            # Position size calculation
            position_size = min(math.floor((balance * precent) / (largest_loss)),max_contracts)
            if position_size < 1: position_size = 1

            # Calculate the profit of each trade:
            trade_points = trade_diff * position_size
            trade_precent = ( trade_diff / entry_price ) * 100 * position_size  
            trade_usd = trade_diff * point_value * position_size

            balance+=trade_usd

            # Cummulative the trades
            cum_profit_points += trade_points
            cum_profit_precent += trade_precent
            cum_profit_usd += trade_usd
        
            new_row = pd.DataFrame({'Time': [time], 
                                    'cum_profit_points': [cum_profit_points], 
                                    'cum_profit_precent': [cum_profit_precent], 
                                    'cum_profit_usd': [cum_profit_usd], 
                                    "single_trade_points_profit": [trade_points],
                                    "single_trade_precent_profit": [trade_precent],
                                    "single_trade_usd_profit": [trade_usd],
                                    "TimeFrame": [df["TimeFrame"][i]],
                                    "Instrument": [df["Instrument"][i]]})
            equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

        if action == "SellShort":
            entry_price = avg_price
            is_long = False

        if action == "BuyToCover" and not is_long and avg_price != 0:

            trade_diff = entry_price - avg_price

            # Position size calculation
            
            position_size = min(math.floor((balance * precent) / (largest_loss)),max_contracts)
            if position_size < 1: position_size = 1

            # Calculate the profit of each trade:
            trade_points = trade_diff  * position_size
            trade_precent =  ( trade_diff / avg_price ) * 100 * position_size  
            trade_usd = trade_diff * point_value   * position_size

            # Update the balance
            balance+=trade_usd
            
            # Cummulative the trades
            cum_profit_points += trade_points
            cum_profit_precent += trade_precent
            cum_profit_usd += trade_usd
            
            cum_profit_points += trade_points
            new_row = pd.DataFrame({'Time': [time], 
                                    'cum_profit_points': [cum_profit_points], 
                                    'cum_profit_precent': [cum_profit_precent], 
                                    'cum_profit_usd': [cum_profit_usd], 
                                    "single_trade_points_profit": [trade_points],
                                    "single_trade_precent_profit": [trade_precent],
                                    "single_trade_usd_profit": [trade_usd],
                                    "TimeFrame": [df["TimeFrame"][i]],
                                    "Instrument": [df["Instrument"][i]]})
            equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

    return equity_curve
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def fixed_precent_money_managment(df: pd.DataFrame, balance, precent_risk=0.01, max_contracts=100) -> pd.DataFrame:
    """
    A money management strategy that calculates position size based on a fixed %, e.g 1%.
    """

    cum_profit_points, cum_profit_precent, cum_profit_usd, entry_price, is_long, equity_curve = 0, 0, 0, 0, None, pd.DataFrame()

    for i in df.index:
        action, avg_price, time, point_value, order_name = df["Action"][i], df["Avg. price"][i], df['Time'][i], df["PointValue"][i], df["Name"][i]
        
        if order_name == "Stop loss":
            stop_price = df["Stop"][i]

        if action == "Buy":
            entry_price = avg_price
            is_long = True

        if action == "Sell" and is_long and avg_price != 0:
            
            trade_risk = abs(entry_price - stop_price)
            trade_diff = avg_price - entry_price
            

            if trade_risk != 0 and point_value != 0:
                # Position size calculation
                position_size = min(math.floor((balance * precent_risk) / (trade_risk * point_value)), max_contracts)
                if position_size < 1: position_size = 1

                # Calculate the profit of each trade:
                trade_points = trade_diff * position_size
                trade_precent = (trade_diff / entry_price) * 100 * position_size # # Percentage profit
                trade_usd = trade_diff * point_value * position_size

                balance += trade_usd

                # Cumulative the trades
                cum_profit_points += trade_points
                cum_profit_precent += trade_precent
                cum_profit_usd += trade_usd
            
                new_row = pd.DataFrame({'Time': [time], 'cum_profit_points': [cum_profit_points], 'cum_profit_precent': [cum_profit_precent], 'cum_profit_usd': [cum_profit_usd], "single_trade_points_profit": [trade_points], "single_trade_precent_profit": [trade_precent], "single_trade_usd_profit": [trade_usd], "TimeFrame": [df["TimeFrame"][i]], "Instrument": [df["Instrument"][i]]})
                equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

        if action == "SellShort":
            entry_price = avg_price
            is_long = False

        if action == "BuyToCover" and not is_long and avg_price != 0:
            
            trade_risk = abs(entry_price - stop_price)
            trade_diff = entry_price - avg_price
            
            if trade_risk != 0 and point_value != 0:
                # Position size calculation
                position_size = min(math.floor((balance * precent_risk) / (trade_risk * point_value)), max_contracts)
                if position_size < 1: position_size = 1

                # Calculate the profit of each trade:
                trade_points = trade_diff * position_size
                trade_precent = (trade_diff / entry_price) * 100 * position_size # Percentage profit
                trade_usd = trade_diff * point_value * position_size

                # Update the balance
                balance += trade_usd

                # Cumulative the trades
                cum_profit_points += trade_points
                cum_profit_precent += trade_precent
                cum_profit_usd += trade_usd

                new_row = pd.DataFrame({'Time': [time], 'cum_profit_points': [cum_profit_points], 'cum_profit_precent': [cum_profit_precent], 'cum_profit_usd': [cum_profit_usd], "single_trade_points_profit": [trade_points], "single_trade_precent_profit": [trade_precent], "single_trade_usd_profit": [trade_usd], "TimeFrame": [df["TimeFrame"][i]], "Instrument": [df["Instrument"][i]]})
                equity_curve = pd.concat([equity_curve, new_row], ignore_index=True)

    return equity_curve
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def resample_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample the DataFrame based on the given frequency."""
    if freq == "Trades": 
        return df
    
    df_resampled = df.resample(freq).last()  # Use .last() to keep the last observation for each period
    for c in df_resampled.columns:
        df_resampled[c] = df_resampled[c].ffill()  # Forward fill to handle NaNs
    return df_resampled
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def get_equity_curves(preccesed_data: list,freq="Trades"): 

    equity_curves = []

    for df in preccesed_data:

        # equity_curve = None
        equity_curve = one_contract_eq(df)  # Get the equity curve             
        equity_curve['Time'] = pd.to_datetime(equity_curve['Time'], format='%d/%m/%Y %H:%M:%S')  # Set Time as index
        equity_curve.set_index('Time', inplace=True)
        equity_curve = resample_data(equity_curve, freq)  # Resample the data
        equity_curves.append(equity_curve)
        
    return equity_curves
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def clear_equity_curves(equity_curves: list) -> list:
    cleared_equity_curves = []
    for equity_curve in equity_curves:
        indices = equity_curve.index
        for i in range(1, len(indices)):
            if indices[i] <= indices[i-1]:
                equity_curve = equity_curve.iloc[:i] 
                break
        cleared_equity_curves.append(equity_curve)
    return cleared_equity_curves
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def get_equity_curve_after_money_managmant(processed_data, general_precent_risk = 2, method = "One contract", freq = "Trades", initial_balance = 100000, fixed_unit = 100) -> list:

    initial_balance = 100000

    # 1.Generate result from monte carlo simulation for getting the most relatible parameters:
    monte_carlo_results = generate_monete_carlo_simulation(get_equity_curves(processed_data),100,"single_trade_usd_profit")


    # 2. loop on the proccesed data - grap the money managment equity curves
    equity_curves = []
    for i, df in enumerate(processed_data):

        if method == "One contract":
            equity_curve = one_contract_eq(df)
        elif method == "Fixed Capital":
            fixed_unit = abs(monte_carlo_results["avg_dd"][i]) / general_precent_risk
            equity_curve = fixed_capital_money_managment(df,fixed_unit,initial_balance)
        elif method == "Fixed Ratio":
            delta = round(abs(monte_carlo_results["avg_dd"][i]))
            equity_curve = fixed_ratio_money_managment(df,delta=delta)
        elif method == "Fixed Unit":
            equity_curve = fixed_unit_money_managment(df,fixed_unit, initial_balance)
        elif method == "Williams Fixed Risk":
            max_loss = round(abs(monte_carlo_results["max_loss"][i]))
            equity_curve = williams_fixed_risk_money_managment(df,general_precent_risk/100,initial_balance,max_loss)
        elif method == "Fixed Precent":
            equity_curve = fixed_precent_money_managment(df,initial_balance,(general_precent_risk/100))

        equity_curves.append(equity_curve)

    # 3. Add more equity curves - the combines equity curves (e.g of all the 15 Min time frame, etc.)
    equity_curves = get_combined_equity_curves(equity_curves)

    # 4. Change all the equity curves to time series based index 
    equity_curves = change_nultiple_dfs_to_time_series(equity_curves,freq)

    # 5. Clean the equity curves if there is any duplications
    equity_curves = clear_equity_curves(equity_curves)

    return equity_curves
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_combined_equity_curves(equity_curves: list) -> list:
    combined_equity_curves = []

    # Initialize the overall combined equity curve
    combined_all_timeframes = None
    
    # Collect equity curves for each TimeFrame
    equity_curves_by_timeframe = {}

    for eq in equity_curves:
        columns_to_combine = [c for c in eq.columns if c not in ["TimeFrame", "Instrument", "Time"]]
        
        if combined_all_timeframes is None:
            # Initialize the combined equity curve with zeros
            combined_all_timeframes = eq.copy()
            combined_all_timeframes["TimeFrame"] = "All Time frames"
            combined_all_timeframes["Instrument"] = "Combined"
            combined_all_timeframes[columns_to_combine] = 0
        
        # Align the index of the current equity curve with the combined equity curve
        eq_aligned = eq.reindex(combined_all_timeframes.index, method='pad', fill_value=0)
        
        for c in columns_to_combine:
            combined_all_timeframes[c] += eq_aligned[c].values
        
        # Process equity curves by TimeFrame
        for timeframe in eq["TimeFrame"].unique():
            if timeframe not in equity_curves_by_timeframe:
                equity_curves_by_timeframe[timeframe] = None
            
            eq_timeframe = eq[eq["TimeFrame"] == timeframe].copy()
            if equity_curves_by_timeframe[timeframe] is None:
                equity_curves_by_timeframe[timeframe] = eq_timeframe.copy()
                equity_curves_by_timeframe[timeframe]["Instrument"] = "Combined" + f" {timeframe}"
                equity_curves_by_timeframe[timeframe][columns_to_combine] = 0
            
            # Align the index of the current equity curve with the combined equity curve for the specific TimeFrame
            eq_timeframe_aligned = eq_timeframe.reindex(equity_curves_by_timeframe[timeframe].index, fill_value=0, method='pad') #method='pad' )
            
            for c in columns_to_combine:
                equity_curves_by_timeframe[timeframe][c] += eq_timeframe_aligned[c].values
    
    # Append the combined curves for each TimeFrame and the overall combined curve to the list
    for timeframe, eq_timeframe in equity_curves_by_timeframe.items():
        eq_timeframe["TimeFrame"] = timeframe + " All" 
        combined_equity_curves.append(eq_timeframe)
    
    # Append the overall combined curve
    combined_equity_curves.append(combined_all_timeframes)

    equity_curves.extend(combined_equity_curves) 

    return equity_curves
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def get_all_money_managment_methods_at_once(processed_data, general_precent_risk = 20, freq = "Trades",initial_balance =100000, fixed_unit = 20):

    # 1. Get the equity curve of 1 contract: 
    equity_curves = get_equity_curves(processed_data)    
    # 2. Generate result from monte carlo simulation for getting the most relatible parameters:
    monte_carlo_results = generate_monete_carlo_simulation(equity_curves,100,"single_trade_usd_profit")

    equity_curves = []

    for i, df in enumerate(processed_data):
        #                   One Contract
        eq_one_contract_eq = one_contract_eq(df)
    
        #                   Fixed Capital
        fixed_unit = abs(monte_carlo_results["avg_dd"][i]) / general_precent_risk
        eq_fixed_capital = fixed_capital_money_managment(df,fixed_unit,initial_balance)

        #                   Fixed Ratio
        delta = round(abs(monte_carlo_results["avg_dd"][i]))
        eq_fixed_ratio = fixed_ratio_money_managment(df,delta=delta)

        #                   Fixed Unit
        eq_fixed_unit = fixed_unit_money_managment(df,fixed_unit, initial_balance)

        #                   Williams Fixed Risk
        max_loss = round(abs(monte_carlo_results["max_loss"][i]))
        eq_williams_fixed_risk = williams_fixed_risk_money_managment(df,general_precent_risk/100,initial_balance,max_loss)

        #                   Fixed Precent
        eq_fixed_precent = fixed_precent_money_managment(df,initial_balance,(general_precent_risk/100))



        # Combine all the result:
        equity_curves.append({
            "One Contract": change_to_time_series(eq_one_contract_eq,freq),
            "Fixed Capital": change_to_time_series(eq_fixed_capital,freq),
            "Fixed Ratio": change_to_time_series(eq_fixed_ratio,freq),
            "Fixed Unit": change_to_time_series(eq_fixed_unit,freq),
            "Williams Fixed Risk": change_to_time_series(eq_williams_fixed_risk, freq),
            "Fixed Precent": change_to_time_series(eq_fixed_precent, freq),
        })


    equity_curves = clear_equity_curves_from_list_of_dicts(equity_curves)
    
    return equity_curves
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def change_to_time_series(data, freq):
    
    data['Time'] = pd.to_datetime(data['Time'], format='%d/%m/%Y %H:%M:%S')  # Set Time as index
    data.set_index('Time', inplace=True)
    data = resample_data(data, freq)  # Resample the data
    return data
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def change_nultiple_dfs_to_time_series(data, freq):
    dfs = []
    for eq in data:
        eq['Time'] = pd.to_datetime(eq['Time'], format='%d/%m/%Y %H:%M:%S')  # Set Time as index
        eq.set_index('Time', inplace=True)
        eq = resample_data(eq, freq)  # Resample the data
        dfs.append(eq)
    return dfs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def clear_equity_curves_from_list_of_dicts(equity_curves_list: list) -> list:
    cleared_equity_curves_list = []
    for equity_curves_dict in equity_curves_list:
        cleared_equity_curves_dict = {}
        for method, equity_curve in equity_curves_dict.items():
            indices = equity_curve.index
            for i in range(1, len(indices)):
                if indices[i] <= indices[i-1]:
                    equity_curve = equity_curve.iloc[:i]
                    break
            cleared_equity_curves_dict[method] = equity_curve
        cleared_equity_curves_list.append(cleared_equity_curves_dict)
    return cleared_equity_curves_list
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------