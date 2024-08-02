import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st

# TODO: For realistic RoR, we make take the realistic initial capital to trade, avg loss for evg risk and so on.
# For initial capital: Max DD * 2
# For risk per trade: Avg loss
# Its a good idea!!!!!!

# ========================================== Fixed precentag =========================================================
@st.cache_data
def simulate_trading_fixed_precentag(initial_capital, risk_per_trade, rrr, win_probability, ruin_threshold, max_trades):
    capital = initial_capital
    capital_trajectory = [capital]
    for _ in range(max_trades):

        if np.random.rand() < win_probability:
            gain = capital * risk_per_trade * rrr
            commision = 0.1 * gain 
            capital += gain 
            capital -= commision
        else:
            loss = capital * risk_per_trade
            commision = 0.1 * loss
            capital -= loss 
            capital -= commision

        capital_trajectory.append(capital)

        if capital < ruin_threshold:
            break
    return capital_trajectory


@st.cache_data
def risk_of_ruin_fixed_precentage(data:pd.DataFrame, initial_capital = 100, risk_per_trade = 0.01, ruin_threshold = 0.5, num_simulations=100, max_trades=1000) -> pd.DataFrame:
        
    rors = []
    for i in range(len(data)):
        trajectories = [
            simulate_trading_fixed_precentag(
                initial_capital,
                risk_per_trade, data["RR"][i], 
                data["win_probability"][i], 
                ruin_threshold, 
                max_trades
            ) for _ in range(num_simulations)
        ]

        ruin_count = sum(1 for trajectory in trajectories if trajectory[-1] < ruin_threshold)
        risk_of_ruin = ruin_count / num_simulations
        rors.append({"Instrument": data["Instrument"][i],"TimeFrame": data["TimeFrame"][i],"RoR": risk_of_ruin*100})

    return pd.DataFrame(rors)

# ====================================================================================================================

# =================================================== fixed_capital ==================================================
 
@st.cache_data 
def simulate_trading_fixed_capital(initial_capital, fixed_risk_amount, rrr, win_probability, ruin_threshold, max_trades):
    capital = initial_capital
    capital_trajectory = [capital]
    
    for _ in range(max_trades):
        if np.random.rand() < win_probability:
            gain = fixed_risk_amount * rrr
            commision = 0.1 * gain 
            capital += gain 
            capital -= commision
        else:
            loss = fixed_risk_amount
            commision = 0.1 * loss
            capital -= loss 
            capital -= commision

        capital_trajectory.append(capital)

        if capital < ruin_threshold:
            break
    
    return capital_trajectory

@st.cache_data
def risk_of_ruin_fixed_capital(data, initial_capital=100, fixed_risk_amount=1, ruin_threshold=0.5, num_simulations=100, max_trades=1000):
    rors = []
    
    for i in range(len(data)):
        trajectories = [
            simulate_trading_fixed_capital(
                initial_capital,
                fixed_risk_amount, 
                data["RR"][i], 
                data["win_probability"][i], 
                ruin_threshold, 
                max_trades
            ) for _ in range(num_simulations)
        ]

        ruin_count = sum(1 for trajectory in trajectories if trajectory[-1] < ruin_threshold)
        risk_of_ruin = ruin_count / num_simulations
        rors.append({
            "Instrument": data["Instrument"][i],
            "TimeFrame": data["TimeFrame"][i],
            "RoR": risk_of_ruin * 100
        })

    return pd.DataFrame(rors)

# ====================================================================================================================

# ======================================================= Fixed risk =================================================
def simulate_trading_fixed_risk(initial_capital, num_units, trade_risk, rrr, win_probability, ruin_threshold, max_trades):
    capital = initial_capital
    capital_trajectory = [capital]
    
    for _ in range(max_trades):
        fixed_dollar_risk = capital / num_units
        contracts = fixed_dollar_risk / trade_risk
        
        if np.random.rand() < win_probability:
            gain = contracts * trade_risk * rrr
            commission = 0.1 * gain 
            capital += gain 
            capital -= commission
        else:
            loss = contracts * trade_risk
            commission = 0.1 * loss
            capital -= loss 
            capital -= commission

        capital_trajectory.append(capital)

        if capital < ruin_threshold:
            break
    
    return capital_trajectory

def risk_of_ruin_fixed_risk(data, initial_capital=100, num_units=100, trade_risk=1, ruin_threshold=0.5, num_simulations=100, max_trades=1000):
    rors = []
    
    for i in range(len(data)):
        trajectories = [
            simulate_trading_fixed_risk(
                initial_capital,
                num_units, 
                trade_risk, 
                data["RR"][i], 
                data["win_probability"][i], 
                ruin_threshold, 
                max_trades
            ) for _ in range(num_simulations)
        ]

        ruin_count = sum(1 for trajectory in trajectories if trajectory[-1] < ruin_threshold)
        risk_of_ruin = ruin_count / num_simulations
        rors.append({
            "Instrument": data["Instrument"][i],
            "TimeFrame": data["TimeFrame"][i],
            "RoR": risk_of_ruin * 100
        })

    return pd.DataFrame(rors)

# ========================================================================================================================

def plot_risk_of_ruin(ror_df):
    fig = make_subplots(rows=len(ror_df['TimeFrame'].unique()), cols=1, subplot_titles=[f"Time Frame: {tf}" for tf in ror_df['TimeFrame'].unique()])

    for i, time_frame in enumerate(ror_df['TimeFrame'].unique()):
        filtered_df = ror_df[ror_df['TimeFrame'] == time_frame]

        fig.add_trace(
            go.Bar(
                x=filtered_df['Instrument'],
                y=filtered_df['RoR'],
                name=f"RoR {time_frame}"
            ),
            row=i + 1, col=1
        )

    fig.update_layout(
        height=200 * len(ror_df['TimeFrame'].unique()),
        title_text="Risk of Ruin by Time Frame",
        showlegend=False
    )

    for i in range(len(ror_df['TimeFrame'].unique())):
        fig.update_xaxes(title_text="Instrument", row=i + 1, col=1)
        fig.update_yaxes(title_text="RoR", row=i + 1, col=1)

    return fig