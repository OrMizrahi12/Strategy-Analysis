import pandas as pd
import streamlit as st
from extra import (process_data,get_equity_curve_end_value, calculate_max_avg_consecutive_loss_wins,
                   calculate_win_loss_metrics,calculate_avg_trades_per_day, calculate_expectency,calculate_ulcer_index,
                   calculate_ulcer_index_for_each_tf,calculate_cagr, calculate_cagr_for_each_tf,calculate_ulcer_performance_index,calculate_upi_for_each_tf)

from plots import (plot_equity_curve_by_timeframe, plot_drawdown, plot_end_value_equity_curve, plot_end_equity_value_percent_pass, plot_mean_end_equity_value
                   ,plot_max_avg_consecutive_loss_wins,plot_win_loss_metrics, plot_avg_trades_per_period, plot_expectency, plot_ulcer_index,
                   plot_ulcer_index_for_each_tf,plot_cagr,plot_cagr_for_each_tf, plot_ulcer_performance_index, plot_upi_for_each_tf,plot_monete_carlo_simulation, plot_all_money_management_methods)
from monte_carlo import generate_monete_carlo_simulation

from risk_of_ruin import  risk_of_ruin_fixed_precentage, plot_risk_of_ruin
from money_managment_methods import get_equity_curve_after_money_managmant, get_all_money_managment_methods_at_once
from conclusion import  get_final_conclusion, plot_conclusion
import io


 

predefined_columns = [
    "Time", "OCO", "TimeFrame", "Instrument", "Action", "Type", "Quantity", 
    "Limit", "Stop", "State", "Filled", "Avg. price", "Name", "PointValue"
]

def main():
    st.set_page_config(page_title="Equity Curve Analysis", layout="wide")
    st.title("Equity Curve Analysis")

    # ----------------------------------------------- Files issue ----------------------------------------

    if 'dataframes' not in st.session_state:
        st.session_state['dataframes'] = {}

    chosen_file = None
    st.sidebar.header("Navigation")
    with st.sidebar:
        btn_submit_row_data_uploading,btn_submit_csv_data_uploading=None,None

        use_extra_analysis = st.selectbox("You want extra analysis ? (Not recomended)", options=[False, True])
        
        with st.expander("Upload Data"):
        
            upload_option = st.selectbox("Select the way you want to upload the data:", options=["Past raw data", "Upload CSV file"])

            if upload_option == "Past raw data": 
                with st.form(key='raw_data_form'):
                    raw_data = st.text_area("Paste your TSV data here")
                    strategy_name = st.text_input("Enter a name for this data (strategy name)", "My strategy")
                    btn_submit_row_data_uploading = st.form_submit_button(label='Submit')
            
            elif upload_option == "Upload CSV file":
                with st.form(key='csv_data_form'):
                    csv_file = st.file_uploader("Upload your CSV files", type=["csv"])
                    strategy_name = st.text_input("Enter a name for this data (strategy name)", "My strategy")
                    btn_submit_csv_data_uploading = st.form_submit_button(label='Submit')



            if  btn_submit_row_data_uploading and raw_data and strategy_name:
                df = pd.read_csv(io.StringIO(raw_data), sep='\t', header=None, names=predefined_columns)
                df._name = strategy_name
                st.session_state['dataframes'][strategy_name] = df
            
            elif btn_submit_csv_data_uploading and csv_file and strategy_name:
                df = pd.read_csv(csv_file)
                df._name = strategy_name
                st.session_state['dataframes'][strategy_name] = df

        with st.expander("Choose The Data"):

            if st.session_state['dataframes']:
                chosen_file = st.selectbox("Select the strategy you want to analyze", options=[name for name in st.session_state['dataframes'].keys()])


    # ---------------------------------------------- Settings --------------------------------------------
    
    with st.expander("Settings"):
        with st.form(key='settings_form'):

            st.subheader("Money managmant")
            col1, col2, col3, col4 = st.columns(4)
            with col1: money_managment_method = st.selectbox("Select Money Manggmant Method", options=["One contract", "Fixed Capital", "Fixed Ratio", "Fixed Unit", "Williams Fixed Risk", "Fixed Precent"])
            with col2: risk_precentage = st.number_input(value=1,label="Precent To risk") 
            with col3: initial_capital = st.number_input(value=100000,label="Initial capital")
            with col4: fixed_units = st.number_input(value=100,label="Fixed Units")
            
            st.subheader("Data format")
            col1, col2 = st.columns(2)
            with col1: freq = st.selectbox("Select the resampling frequency:", options=["Trades", "D", "W", "M"])
            with col2: eq_base = st.selectbox("Select equity curve value base:", options=["cum_profit_usd", "cum_profit_points", "cum_profit_precent"])

            st.subheader("Monte Carlo")
            col1,col2 = st.columns(2)
            with col1: simulation_num = st.number_input(value=100,label="Number of simulations")
            with col2:
                option_mapping = {"$": "single_trade_usd_profit", "P": "single_trade_points_profit", "%": "single_trade_precent_profit"} 
                options = list(option_mapping.keys()) 
                monte_carlo_column_base = st.selectbox(label="Type", options=options)

            st.subheader("Risk of Ruin")
            col1,col2,col3 = st.columns(3)
            with col1: num_of_sim = st.number_input(value=100, label="Number Of Simulations")
            with col2: max_trades = st.number_input(value=5000, label="Max trades") 
            with col3: maximum_loss = st.number_input(value=50, label="Maximum Drawdown you can saffer (%)") / 100

            st.subheader("Conclusion Report")
            money_managment_methods_for_report = st.multiselect("Choose / Cancel Money managment methods in the report", 
                                                                ["One contract", "Fixed Capital", "Fixed Ratio", "Fixed Unit", "Williams Fixed Risk", "Fixed Precent"], 
                                                                ["One contract"])
            show_conclusion_plot = st.selectbox("Show conclusion Plots", options=[False, True])



            btn_submit_settings = st.form_submit_button(label='Submit')

    # ----------------------------------------------- Tabs -------------------------------------------------

    tab_1, tab_2, tab_3, tab_4, tab_5 = st.tabs(["Equity curve analysis", "Stats", "Simulations", "Money Managmant Comperation", "Conclusion"])

    # --------------------------------------------- Loop over the user files ---------------------------------------------- 
    if chosen_file and btn_submit_settings:
        # ------------------------------------ Data proccesing -----------------------------------

        big_data_frame = st.session_state['dataframes'][chosen_file]            # Make the data csv
        processed_data = process_data(big_data_frame) # Seperate the big df to structure of: [ df,df ...]

#         # ----------------------------------- Data Calculations ------------------------------------
        
        if use_extra_analysis:
            equity_curves = get_equity_curve_after_money_managmant(processed_data, risk_precentage, money_managment_method, freq,initial_capital,fixed_units)
            
    #         # !--------------------------------------------------------!
            
            equity_curves_end_value_df =  get_equity_curve_end_value(equity_curves,eq_base)           # Get a df of the end values of equity curves
            max_avg_consecutive_loss_wins_df = calculate_max_avg_consecutive_loss_wins(equity_curves) # get a df of max / avg cons. loss / wins 
            win_loss_metrics_df = calculate_win_loss_metrics(equity_curves,eq_base)                   # Get a df of some win loss metrics (like RR, win %, and so on)
            avg_trades_per_day = calculate_avg_trades_per_day(equity_curves)                          # Get a df of the avg trades per day 
            expectency_df = calculate_expectency(win_loss_metrics_df, avg_trades_per_day)             # Get a df of the expectency.
            ulcer_index_df = calculate_ulcer_index(equity_curves)                                     # Get a df of ucler index. 
            ulcer_index_df_for_each_tf_df = calculate_ulcer_index_for_each_tf(ulcer_index_df)         # Get a df of mean ucler index for each tf and all. 
            cagr_df = calculate_cagr(equity_curves)                                                   # Get a df of CAGR
            cagr_for_each_tf_df = calculate_cagr_for_each_tf(cagr_df)                                 # Get a df of CARG of each tf and all. 
            ucler_performance_index_df = calculate_ulcer_performance_index(cagr_df,ulcer_index_df)    # Get a df of UPI
            upi_for_each_tf_df = calculate_upi_for_each_tf(ucler_performance_index_df)                # Get a df of UPI of each tf and all.

            # ------------------------------------ Generate plots ---------------------------------------

            general_fig, figs_by_tf = plot_equity_curve_by_timeframe(equity_curves, eq_base) 
            figs_by_tf_dd_fig = plot_drawdown(equity_curves, eq_base) 
            end_value_fig = plot_end_value_equity_curve(equity_curves_end_value_df) 
            end_value_fig_precent_pass_fig = plot_end_equity_value_percent_pass(equity_curves_end_value_df) 
            mean_end_value_fig = plot_mean_end_equity_value(equity_curves_end_value_df) 
            max_avg_consecutive_loss_wins_fig = plot_max_avg_consecutive_loss_wins(max_avg_consecutive_loss_wins_df)  
            win_loss_fig = plot_win_loss_metrics(win_loss_metrics_df) 
            avg_trades_per_period_fig = plot_avg_trades_per_period(avg_trades_per_day)  
            expectency_fig = plot_expectency(expectency_df)  
            ucler_index_fig = plot_ulcer_index(ulcer_index_df)
            ulcer_index_for_each_tf_fig = plot_ulcer_index_for_each_tf(ulcer_index_df_for_each_tf_df)
            cagr_fig = plot_cagr(cagr_df)
            cagr_for_each_tf_fig = plot_cagr_for_each_tf(cagr_for_each_tf_df)
            ulcer_performance_index_fig = plot_ulcer_performance_index(ucler_performance_index_df)
            upi_for_each_tf_fig = plot_upi_for_each_tf(upi_for_each_tf_df)

            

            # ---------------------------------------- TAB 1: Equity curves ----------------------------------------
            with tab_1:
                st.header("General Equity Curves 📈")
                with st.expander("Show / Hide"):
                    st.plotly_chart(general_fig, use_container_width=True)
                    
                    st.header("Equity Curves by Time Frame")
                    cols = st.columns(2)
                    for i, (_, fig) in enumerate(figs_by_tf.items()):
                        with cols[i % 2]:
                            st.plotly_chart(fig, use_container_width=True)

                st.header("Drawdowns 📉")
                with st.expander("Show / Hide"):
                    cols = st.columns(2)
                    for i, (_, fig) in enumerate(figs_by_tf_dd_fig.items()):
                        with cols[i % 2]:
                            st.plotly_chart(fig, use_container_width=True)

                    
            # ------------------------------------------- TAB 2: Stats --------------------------------------------
            with tab_2:
                st.header("Equity curve End values 💰")
                with st.expander("Show / Hide"):
                    col1, col2,col3 = st.columns(3)
                    with col1:                 
                        st.plotly_chart(end_value_fig, use_container_width=True)
                    with col2:
                        st.plotly_chart(end_value_fig_precent_pass_fig, use_container_width=True)
                    with col3:
                        st.plotly_chart(mean_end_value_fig, use_container_width=True)
                
                st.header("Max / Avg cons. Loss / Wins 🏅")
                with st.expander("Show / Hide"):
                    st.plotly_chart(max_avg_consecutive_loss_wins_fig, use_container_width=True)
                
                st.header("Win Loss metrics ⚖️")
                with st.expander("Show / Hide"):
                    st.plotly_chart(win_loss_fig,use_container_width=True)
                
                
                st.header("Avg trades per period 📊")
                with st.expander("Show / Hide"):
                    st.plotly_chart(avg_trades_per_period_fig,use_container_width=True)
                
                st.header("Expectency 🎯")
                with st.expander("Show / Hide"):
                    st.plotly_chart(expectency_fig,use_container_width=True)
                
                st.header("CAGR 🚀")
                with st.expander("Show / Hide"):
                    st.plotly_chart(cagr_fig, use_container_width = True)
                    st.plotly_chart(cagr_for_each_tf_fig, use_container_width = True)

                st.header("Ucler 🩹")
                with st.expander("Show / Hide"):
                    st.plotly_chart(ucler_index_fig, use_container_width=True)
                    st.plotly_chart(ulcer_index_for_each_tf_fig, use_container_width=True)    
                    st.plotly_chart(ulcer_performance_index_fig, use_container_width = True)
                    st.plotly_chart(upi_for_each_tf_fig, use_container_width = True)


            # ------------------------------------------- Simulations --------------------------------------------    
            with tab_3:
                st.header("Monte Carlo Simulation 🎲")
                with st.expander("Show / Hide"):
                    monte_carlo_equity_curve_simulations = generate_monete_carlo_simulation(equity_curves, simulation_num, option_mapping[monte_carlo_column_base])
                    monte_carlo_equity_curve_simulations_fig = plot_monete_carlo_simulation(monte_carlo_equity_curve_simulations)
                    st.plotly_chart(monte_carlo_equity_curve_simulations_fig, use_container_width=True)
                        
                st.header("Risk of Ruin🌪️")
                with st.expander("Show / Hide"):

                    risk_of_ruin_fixed_precentage_df = risk_of_ruin_fixed_precentage(win_loss_metrics_df,
                                                                                        risk_per_trade=risk_precentage / 100,
                                                                                        num_simulations=num_of_sim,
                                                                                        max_trades=max_trades,
                                                                                        ruin_threshold=maximum_loss)
                        
                    risk_of_ruin_fixed_percentage_fig = plot_risk_of_ruin(risk_of_ruin_fixed_precentage_df)
                    st.plotly_chart(risk_of_ruin_fixed_percentage_fig, use_container_width=True)
                
            

            with tab_4: 
                equity_curves_all = get_all_money_managment_methods_at_once(processed_data, risk_precentage,freq, initial_capital,fixed_units)
                equity_curves_all_figs = plot_all_money_management_methods(equity_curves_all, eq_base)

                for fig in equity_curves_all_figs:
                    st.plotly_chart(fig)

        with tab_5:
            st.subheader("Here's the final conclusion and the result of the strategy")
            final_conclusion_df =  get_final_conclusion(processed_data,money_managment_methods_for_report,risk_precentage,initial_capital,fixed_units,simulation_num,max_trades,maximum_loss)
            
            if show_conclusion_plot:
                st.plotly_chart(plot_conclusion(final_conclusion_df))
            
            final_conclusion_df["ID"] = final_conclusion_df["Instrument"] + " " + final_conclusion_df["TimeFrame"]
            del final_conclusion_df["Instrument"]
            del final_conclusion_df["TimeFrame"]
            st.write(final_conclusion_df.T) 
            




if __name__ == "__main__":
    main()
                    

        