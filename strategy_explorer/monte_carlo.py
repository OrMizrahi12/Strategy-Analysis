import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def calculate_drawdowns(cumm_return):
    drawdowns = np.maximum.accumulate(cumm_return) - cumm_return
    max_dd = np.max(drawdowns, axis=1)
    avg_dd = np.mean(drawdowns, axis=1)
    best_dd = np.min(drawdowns, axis=1)
    return -max_dd, -avg_dd, -best_dd
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def generate_monete_carlo_simulation(equity_curves: list, num_of_simulations=1000, column="single_trade_usd_profit"):
    results = []
    
    for equity_curve in equity_curves:
        num_of_trades = len(equity_curve)
        
        # 1. Generate random samples
        samples = np.random.choice(equity_curve[column].values, size=(num_of_simulations, num_of_trades), replace=True)
        
        # 2. Calculate cumulative returns
        cumm_return = np.cumsum(samples, axis=1)
        
        # 3. Extract final cumulative profits
        final_profits = cumm_return[:, -1]
        
        # 4. Calculate drawdowns
        max_dd, avg_dd, best_dd = calculate_drawdowns(cumm_return)
        
        # 5. Calculate the best, average, and worst scenario for the final cumulative profit
        best_scenario   = np.max(final_profits)
        avg_scanraio    = np.mean(final_profits)
        worst_scanraio  = np.min(final_profits)

        # 6. Calculate maximum and average loss per trade
        losses = samples[samples < 0]  # Filter only losses
        max_loss_per_simulation = np.min(samples, axis=1)
        avg_loss_per_simulation = np.mean(samples, axis=1)
        
        # 6. Add the simulation result of the instrument
        results.append({
            "Instrument": equity_curve["Instrument"][0], 
            "TimeFrame": equity_curve["TimeFrame"][0], 
            "best_scenario": best_scenario, 
            "avg_scanraio": avg_scanraio, 
            "worst_scanraio": worst_scanraio,
            "max_dd": np.mean(max_dd),    # average max drawdown across simulations
            "avg_dd": np.mean(avg_dd),    # average drawdown across simulations
            "best_dd": np.mean(best_dd),  # average best drawdown across simulations
            "max_loss": np.mean(max_loss_per_simulation), # average max loss across simulations
            "avg_loss": np.mean(avg_loss_per_simulation)  # average loss across simulations
        })
    
    return pd.DataFrame(results)
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------