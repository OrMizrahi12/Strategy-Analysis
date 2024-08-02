from extra import *
from money_managment_methods import *
from monte_carlo import * 
from risk_of_ruin import * 
import plotly.express as px

def get_final_conclusion(processed_data: list,money_managment_methods_for_report, risk_precentage,initial_capital,fixed_units, simulation_num = 100, max_trades = 5000, maximum_loss = 50):
    
    
    base_names = ["$","%"]
    base_columns = ["cum_profit_usd","cum_profit_precent"] 
    final_df = pd.DataFrame()
    for money_managmant_method in money_managment_methods_for_report: 
        
        # Get the equity curves
        equity_curves = get_equity_curve_after_money_managmant(processed_data=processed_data, 
                                                               general_precent_risk=risk_precentage, 
                                                               method=money_managmant_method,
                                                               initial_balance=initial_capital,
                                                               fixed_unit=fixed_units)
        
        
        # for $ and %
        for name, column in zip(base_names,base_columns):
            
            # Get the end value 
            curve_end_value =  get_equity_curve_end_value(equity_curves,column)
            final_df["Instrument"] = curve_end_value["Instrument"] 
            final_df["TimeFrame"] = curve_end_value["TimeFrame"]
            final_df[f"End Value ({name}) - {money_managmant_method}"] = curve_end_value["EndValue"] 

            # Calculate expectency
            win_loss_metrics_df = calculate_win_loss_metrics(equity_curves,column)                   # Get a df of some win loss metrics (like RR, win %, and so on)
            avg_trades_per_day = calculate_avg_trades_per_day(equity_curves)                          # Get a df of the avg trades per day 
            expectency_df = calculate_expectency(win_loss_metrics_df, avg_trades_per_day) 
            final_df[f"expectency_per_usd ({name}) - {money_managmant_method}"] = expectency_df["expectency_per_usd"]
            final_df[f"expectency_per_day ({name}) - {money_managmant_method}"] = expectency_df["expectency_per_day"]
            final_df[f"expectency_per_week ({name}) - {money_managmant_method}"] = expectency_df["expectency_per_week"]
            final_df[f"expectency_per_month ({name}) - {money_managmant_method}"] = expectency_df["expectency_per_month"]
            final_df[f"expectency_per_quarter ({name}) - {money_managmant_method}"] = expectency_df["expectency_per_quarter"]
            final_df[f"expectency_per_year ({name}) - {money_managmant_method}"] = expectency_df["expectency_per_year"]

            # Monte Carlo
            monte_carlo_equity_curve_simulations = generate_monete_carlo_simulation(equity_curves, simulation_num, column)
            final_df[f"Monte Carlo Worst DD ({name}) - {money_managmant_method}"] = monte_carlo_equity_curve_simulations["max_dd"]
            final_df[f"Monte Carlo Avg DD ({name}) - {money_managmant_method}"] = monte_carlo_equity_curve_simulations["avg_dd"]
            final_df[f"Monte Carlo Worst Net profit ({name}) - {money_managmant_method}"] = monte_carlo_equity_curve_simulations["worst_scanraio"]
            final_df[f"Monte Carlo Avg Net profit ({name}) - {money_managmant_method}"] = monte_carlo_equity_curve_simulations["avg_scanraio"]

        # Calculate the UI, UPI, CAGR
        ulcer_index_df = calculate_ulcer_index(equity_curves)
        cagr_df = calculate_cagr(equity_curves)                                                  
        ucler_performance_index_df = calculate_ulcer_performance_index(cagr_df,ulcer_index_df)  
        final_df[f"UI - ({money_managmant_method})"] =  ulcer_index_df["ulcer_index"]                                     
        final_df[f"CAGR (%) - ({money_managmant_method})"] =  cagr_df["CAGR"]                                     
        final_df[f"UPI - ({money_managmant_method})"] =  ucler_performance_index_df["ulcer_performance_index"]


    # Calculate risk of ruin 
    win_loss_metrics_df = calculate_win_loss_metrics(equity_curves,column)
    risk_of_ruin_fixed_precentage_df = risk_of_ruin_fixed_precentage(win_loss_metrics_df,
                                                                        risk_per_trade=risk_precentage / 100,
                                                                        num_simulations=simulation_num,
                                                                        max_trades=max_trades,
                                                                        ruin_threshold=maximum_loss)  
    final_df["RoR (%)"] = risk_of_ruin_fixed_precentage_df["RoR"]
    

    return final_df.round(2)

        


def plot_conclusion(final_df):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    color_index = 0
    
    # Combine Instrument and TimeFrame for x-axis
    final_df['Instrument_TimeFrame'] = final_df['Instrument'] + " - " + final_df['TimeFrame']

    for column in final_df.columns:
        if column not in ["Instrument", "TimeFrame", "Instrument_TimeFrame"]:
            fig.add_trace(go.Scatter(
                x=final_df["Instrument_TimeFrame"],
                y=final_df[column],
                mode='lines+markers',
                name=column,
                line=dict(color=colors[color_index])
            ))
            
            # Update the color index to ensure no color repeats
            color_index = (color_index + 1) % len(colors)

    fig.update_layout(
        title="Comprehensive Analysis of Money Management Methods",
        xaxis_title="Instrument - TimeFrame",
        yaxis_title="Values",
        legend_title="Metrics",
        template="plotly_dark"
    )

    return fig


    

