import pandas as pd, streamlit as st, numpy as np
import statistics
from money_managment_methods import one_contract_eq




# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def remove_duplications(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop_duplicates()
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def process_data(data: pd.DataFrame) -> list:
    try:
        data = remove_duplications(data)
        grouped = data.groupby(['Instrument', 'TimeFrame'])
        return [group for _, group in grouped]
    except:
        return []
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def resample_data(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample the DataFrame based on the given frequency."""
    if freq == "Trades": 
        return df
    
    df_resampled = df.resample(freq).last()  # Use .last() to keep the last observation for each period
    for c in df_resampled.columns:
        df_resampled[c] = df_resampled[c].ffill()  # Forward fill to handle NaNs
    return df_resampled
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def calculate_drawdown(data: pd.DataFrame) -> pd.DataFrame:
    drawdown_df = pd.DataFrame(index=data.index, columns=data.columns)
    for column in data.columns:
        if column not in ["TimeFrame", "Instrument"]:
            eq = data[column].values
            running_max = pd.Series(eq).cummax().values
            drawdowns = running_max - eq
            drawdown_df[column] = -drawdowns
    return drawdown_df
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def get_equity_curve_end_value(equity_curves: list, column = "cum_profit_usd") -> pd.DataFrame:
    end_values ,time_frames, instruments = [], [], []
    for eq in equity_curves:
        # Get the end value for the specified column
        end_value = eq[column].iloc[-1]
        # Append the end value, time frame, and instrument
        end_values.append(end_value)
        time_frames.append(eq["TimeFrame"][0])
        instruments.append(eq["Instrument"][0])
    # Create a DataFrame for analysis
    end_value_data = pd.DataFrame({ 'TimeFrame': time_frames, 'EndValue': end_values, 'Instrument': instruments})

    return end_value_data
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def get_equity_curves(preccesed_data: list,freq="Trades"): 
    equity_curves = []
    for df in preccesed_data:
        equity_curve = one_contract_eq(df)  # Get the equity curve
        equity_curve['Time'] = pd.to_datetime(equity_curve['Time'], format='%d/%m/%Y %H:%M:%S')  # Set Time as index
        equity_curve.set_index('Time', inplace=True)
        equity_curve = resample_data(equity_curve, freq)  # Resample the data
        equity_curves.append(equity_curve)
    return equity_curves
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_max_avg_consecutive_loss_wins(data):
    results = []

    for equity_curve in data:
        
        try:
            cum_profit = equity_curve['cum_profit_usd']
            max_con_loss, max_con_wins, current_con_loss, current_con_wins = 0,0,0,0
            con_loss_lengths, con_win_lengths = [], []

            # Loop over the equity curve, take the concecutive wins and losses
            for i in range(1, len(cum_profit)):
                if cum_profit[i] < cum_profit[i - 1]: # if this trade a loss
                    current_con_loss += 1
                    if current_con_wins > 0:
                        con_win_lengths.append(current_con_wins)
                        max_con_wins = max(max_con_wins, current_con_wins)
                        current_con_wins = 0
                else:                                # if this trade a win
                    current_con_wins += 1
                    if current_con_loss > 0:
                        con_loss_lengths.append(current_con_loss)
                        max_con_loss = max(max_con_loss, current_con_loss)
                        current_con_loss = 0

            # Add the last streak if it ends with a win or loss
            if current_con_loss > 0:
                con_loss_lengths.append(current_con_loss)
                max_con_loss = max(max_con_loss, current_con_loss)
            if current_con_wins > 0:
                con_win_lengths.append(current_con_wins)
                max_con_wins = max(max_con_wins, current_con_wins)

            # take the avarage of those 
            avg_con_loss = sum(con_loss_lengths) / len(con_loss_lengths) if con_loss_lengths else 0
            avg_con_wins = sum(con_win_lengths) / len(con_win_lengths) if con_win_lengths else 0
        except:
            max_con_loss, max_con_wins, avg_con_wins, avg_con_loss = 0,0,0,0

        results.append({
            "Instrument":equity_curve["Instrument"][0],
            "TimeFrame": equity_curve["TimeFrame"][0],
            "MaxConLoss": max_con_loss,
            "MaxConWins": max_con_wins,
            "AvgConWin": avg_con_wins,
            "AvgConLoss": avg_con_loss
        })

    result_df = pd.DataFrame(results)

    return result_df
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_win_loss_metrics(equity_curves: list, column='cum_profit_usd'):
    results = []

    for eq in equity_curves:
        try:
            cum_profit = eq[column]
            losses, wins = [], []
            win_count, loss_count = 0,0

            for i in range(1, len(cum_profit)):
                equity_diff = cum_profit[i] - cum_profit[i - 1]
                if equity_diff < 0:
                    losses.append(abs(equity_diff))
                    loss_count += 1
                else:
                    wins.append(abs(equity_diff))
                    win_count += 1

            if losses: avg_loss = statistics.mean(losses) 
            if wins: avg_win = statistics.mean(wins) 
            
            if wins and not losses: win_probability = 1
            elif losses and not wins: win_probability = 0
            elif wins and losses: win_probability = win_count / (loss_count + win_count)

            rr = avg_win / avg_loss
            max_loss =  max(losses) if losses else 0
            max_win = max(wins) if wins else 0
        
        except:
            avg_loss, avg_win, rr, win_probability, max_loss, max_win = 0,0,0,0,0,0
        
        results.append({ "Instrument": eq["Instrument"][0],
                         "TimeFrame": eq["TimeFrame"][0], 
                         "avg_loss": avg_loss, 
                         "avg_win": avg_win, 
                         "RR": rr,
                         "win_probability": win_probability,
                         "max_loss": max_loss,
                         "max_win": max_win})

   
    return pd.DataFrame(results)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_avg_trades_per_day(data) -> pd.DataFrame:
    results = []
    for equity_curve in data:
        try:
            equity_curve.index = pd.to_datetime(equity_curve.index, format='%d/%m/%Y %H:%M:%S') # Ensure the index is in datetime format
            start_date = equity_curve.index.min().normalize()                                   # Define the start and end date for the complete date range
            end_date = equity_curve.index.max().normalize()
            all_dates = pd.date_range(start=start_date, end=end_date, freq='D')                 # Generate a complete date range covering the period of the index
            trades_per_day = equity_curve.groupby(equity_curve.index.date).size()               # Count the number of trades per day
            trades_per_day = trades_per_day.reindex(all_dates.date, fill_value=0)               # Reindex trades_per_day to include all dates, filling missing dates with zero
            avg_trades_per_day = trades_per_day.mean()                                          # Calculate the average number of trades per day
        except:
            avg_trades_per_day = 0
        
        results.append({ 'instrument': equity_curve["Instrument"][0], 'time_frame': equity_curve["TimeFrame"][0], 'avg_trades_per_day': avg_trades_per_day}) # Append results to the list

    return pd.DataFrame(results)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_expectency(win_loss_metrics, avg_trades_per_day):
    result = []
    
    for i in range(len(win_loss_metrics)):
        try:
            # Risk-Reward Ratio
            rr = win_loss_metrics["avg_win"][i] / win_loss_metrics["avg_loss"][i]
            
            # Calculate expectancy per trade
            winning = win_loss_metrics["win_probability"][i] * win_loss_metrics["avg_win"][i]
            losing = (1 - win_loss_metrics["win_probability"][i]) * win_loss_metrics["avg_loss"][i]
            expectency_per_trade = winning - losing

            # Calculate expectancy per USD
            expectency_per_usd = win_loss_metrics["win_probability"][i] * rr - (1 - win_loss_metrics["win_probability"][i])
            
            # Get average trades per day
            avg_trades = avg_trades_per_day["avg_trades_per_day"][i]
            
            # Calculate expectancy per period
            expectency_per_day = avg_trades * expectency_per_trade
            expectency_per_week = 6 * avg_trades * expectency_per_trade
            expectency_per_month = 22 * avg_trades * expectency_per_trade
            expectency_per_quarter = 65 * avg_trades * expectency_per_trade
            expectency_per_year = 252 * avg_trades * expectency_per_trade
        except:
            expectency_per_usd,expectency_per_trade,expectency_per_day,expectency_per_week,expectency_per_month,expectency_per_quarter,expectency_per_year =0,0,0,0,0,0,0 
        # Append results
        result.append({
            "Instrument": win_loss_metrics["Instrument"].iloc[i],
            "TimeFrame": win_loss_metrics["TimeFrame"].iloc[i],
            "expectency_per_usd": expectency_per_usd,
            "expectency_per_trade": expectency_per_trade,
            "expectency_per_day": expectency_per_day,
            "expectency_per_week": expectency_per_week,
            "expectency_per_month": expectency_per_month,
            "expectency_per_quarter": expectency_per_quarter,
            "expectency_per_year": expectency_per_year,
        })
    
    return pd.DataFrame(result)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_ulcer_index(equity_curves: pd.Series) -> float:
    
    # NOTE: the calculation of the ulcer index should be only with the precent column. 
    ulcer_indexs = []
    column = "cum_profit_precent"
    for eq in equity_curves:
        try:
            max_returns = eq[column].cummax()      # 1. Calc the running max. eg: [1,5,4,3,2,1] -> [1,5,5,5,5,5]                        
            drawdowns = eq[column] - max_returns   # 2. Calc the DD for each data point 
            squared_drawdowns = drawdowns ** 2     # 3. Squared the DDs
            ui = np.sqrt(squared_drawdowns.mean()) # 4. Calc the ucler index 
        except:
            ui = 0
        ulcer_indexs.append({"Instrument": eq["Instrument"].iloc[0], "TimeFrame": eq["TimeFrame"].iloc[0],"ulcer_index": ui})

    return pd.DataFrame(ulcer_indexs)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_ulcer_index_for_each_tf(ulcer_index_df: pd.DataFrame):
    grouped_df = ulcer_index_df.groupby("TimeFrame")["ulcer_index"].mean().reset_index()
    overall_mean = pd.DataFrame({"TimeFrame": ["All"], "ulcer_index": [ulcer_index_df["ulcer_index"].mean()]})
    result_df = pd.concat([grouped_df, overall_mean], ignore_index=True)
    return result_df
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


@st.cache_data
def calculate_cagr(equity_curve: pd.Series) -> pd.DataFrame:
    cgars = []
    column = "cum_profit_precent"
    for eq in equity_curve:
        try:
            equity_series = eq[column]
            ending_value = equity_series.iloc[-1]
            num_years = (equity_series.index[-1] - equity_series.index[0]).days / 365.25
            growth_factor = 1 + ending_value / 100 
            cgar = ((growth_factor) ** (1 / num_years) - 1) * 100 
        except:
            cgar = 0
        cgars.append({"Instrument": eq["Instrument"][0], "TimeFrame": eq["TimeFrame"][0], "CAGR": cgar})

    return pd.DataFrame(cgars)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_cagr_for_each_tf(cagr_df:pd.DataFrame) -> pd.DataFrame:
    cagr_tf_mean_df = cagr_df.groupby("TimeFrame")["CAGR"].mean().reset_index()
    cagr_all_mean = cagr_df["CAGR"].mean()
    cagr_all_mean_df = pd.DataFrame({"TimeFrame": ["All"], "CAGR": [cagr_all_mean]})
    return pd.concat([cagr_tf_mean_df,cagr_all_mean_df], ignore_index=True)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_ulcer_performance_index(cgr_df:pd.DataFrame, ulcer_index_df: pd.DataFrame) -> pd.DataFrame:

    ucler_performance_indexes = []
    for i in range(len(cgr_df)):
        try:
            cagr = cgr_df["CAGR"][i]
            ui = ulcer_index_df["ulcer_index"][i]
            upi = cagr / ui
        except:
            upi = 0

        ucler_performance_indexes.append({"TimeFrame":cgr_df["TimeFrame"][i], "Instrument": cgr_df["Instrument"][i], "ulcer_performance_index": upi})

    return pd.DataFrame(ucler_performance_indexes) 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_upi_for_each_tf(ucler_performance_index_df:pd.DataFrame) -> pd.DataFrame:
    upi_tf_mean_df = ucler_performance_index_df.groupby("TimeFrame")["ulcer_performance_index"].mean().reset_index()
    upi_all_mean = ucler_performance_index_df["ulcer_performance_index"].mean()
    upi_all_mean_df = pd.DataFrame({"TimeFrame": ["All"], "ulcer_performance_index": [upi_all_mean]})
    return pd.concat([upi_tf_mean_df,upi_all_mean_df], ignore_index=True)
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 