import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from extra import calculate_drawdown


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_equity_curve_by_timeframe(data, eq_base="cum_profit_points"):

    # Initialize a single figure for all equity curves
    figs_by_tf = {df["TimeFrame"][0]: go.Figure() for df in data}
    general_fig = go.Figure()

    # Prepare and plot each equity curve
    for equity_curve in data:
        tf = equity_curve["TimeFrame"][0]
        ins = equity_curve["Instrument"][0]

        figs_by_tf[tf].add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve[eq_base], mode='lines', name=f"{ins} ({tf})", )
        )
        general_fig.add_trace(
            go.Scatter(x=equity_curve.index, y=equity_curve[eq_base], mode='lines', name=f"{ins} ({tf})")
        )

        figs_by_tf[tf].update_layout(title=f"Equity curves ({tf})")
        general_fig.update_layout(title=f"Feneral Equity curves")

    # Return the general figure and the dictionary of figures by time frame
    return general_fig, figs_by_tf
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_drawdown(data, eq_base="cum_profit_points"):

    # Initialize a single figure for all equity curves
    # figs_by_tf = {tf: go.Figure() for (_, tf), _ in data}
    figs_by_tf = {df["TimeFrame"][0]: go.Figure() for df in data}

    # Prepare and plot each equity curve
    for equity_curve in data:

        dd = calculate_drawdown(equity_curve)
        
        # Add red cloud style for drawdown area
        tf = equity_curve["TimeFrame"][0]
        ins = equity_curve["Instrument"][0]
        figs_by_tf[tf].add_trace( go.Scatter( x=dd.index, y=dd[eq_base], fill='tozeroy', name=f"{ins} ({tf})", line=dict(color='red', width=1), showlegend=True ))

        # Update layout for better visualization
        figs_by_tf[tf].update_layout( title=f"Drawdown Plot ({tf})", xaxis_title="Time", yaxis_title="Drawdown", xaxis_rangeslider_visible=False, template="plotly_white")

    # Return the dictionary of figures by time frame
    return figs_by_tf
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_end_value_equity_curve(equity_curves_end_value_df):


    # Define color based on whether EndValue is positive or negative
    equity_curves_end_value_df['Color'] = equity_curves_end_value_df['EndValue'].apply(lambda value: 'green' if value >= 0 else 'red')

    # Create the scatter plot
    fig = px.scatter(
        data_frame=equity_curves_end_value_df,
        x='TimeFrame',  # Use TimeFrame for X-axis
        y='EndValue',
        color='Color',
        color_discrete_map={'green': 'green', 'red': 'red'},
        title='End Value of Equity Curves by Time Frame',
        labels={'EndValue': 'End Value', 'TimeFrame': 'Time Frame'},
        hover_name='Instrument'  # Show instrument names on hover
    )

    # Update layout to hide the legend and format hover information
    fig.update_layout(
        xaxis_title="Time Frame",
        yaxis_title="End Value",
        template="plotly_white",
        showlegend=False  # Hide the legend
    )
    
    # Update hover template to only show instrument name
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>End Value: %{y}<br>Time Frame: %{x}<extra></extra>'
    )

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_end_equity_value_percent_pass(equity_curves_end_value_df) -> px.histogram:

    # Calculate the percentage of end values > 0 for each time frame
    percentage_pass = equity_curves_end_value_df.groupby('TimeFrame')['EndValue'].apply(lambda x: (x > 0).mean() * 100).reset_index()
    percentage_pass.columns = ['TimeFrame', 'PercentPass']

    # Calculate the overall percentage pass for all time frames
    overall_pass_rate = (equity_curves_end_value_df['EndValue'] > 0).mean() * 100
    overall_pass_df = pd.DataFrame([{'TimeFrame': 'ALL', 'PercentPass': overall_pass_rate}])

    # Combine the percentage pass data with the overall pass rate
    percentage_pass = pd.concat([percentage_pass, overall_pass_df], ignore_index=True)

    # Create a bar chart to visualize the percentage pass
    fig = px.bar(
        percentage_pass,
        x='TimeFrame',
        y='PercentPass',
        title='Percentage of Positive End Values by Time Frame',
        labels={'PercentPass': 'Percentage of Positive End Values', 'TimeFrame': 'Time Frame'},
        color_discrete_sequence=['green']
    )

    return fig


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_mean_end_equity_value(equity_curves_end_value_df) -> px.histogram:

    # Calculate the mean end value for each time frame
    mean_end_value = equity_curves_end_value_df.groupby('TimeFrame')['EndValue'].mean().reset_index()
    mean_end_value.columns = ['TimeFrame', 'MeanEndValue']

    # Calculate the overall mean end value for all time frames
    overall_mean_end_value = equity_curves_end_value_df['EndValue'].mean()
    overall_mean_end_value_df = pd.DataFrame([{'TimeFrame': 'ALL', 'MeanEndValue': overall_mean_end_value}])

    # Combine the mean end value data with the overall mean end value
    mean_end_value = pd.concat([mean_end_value, overall_mean_end_value_df], ignore_index=True)

    # Add a color column based on the MeanEndValue
    mean_end_value['Color'] = mean_end_value['MeanEndValue'].apply(lambda x: 'red' if x < 0 else 'green')

    # Create a bar chart to visualize the mean end values
    fig = px.bar(
        mean_end_value,
        x='TimeFrame',
        y='MeanEndValue',
        title='Mean End Value of Equity Curve for Each Time Frame',
        labels={'MeanEndValue': 'Mean End Value', 'TimeFrame': 'Time Frame'},
        color='Color',
        color_discrete_map={'red': 'red', 'green': 'green'}
    )

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_max_avg_consecutive_loss_wins(max_avg_consecutive_loss_wins_df: pd.DataFrame):
    
    
    # Create a new column combining Instrument and TimeFrame
    max_avg_consecutive_loss_wins_df['Instrument_TimeFrame'] = max_avg_consecutive_loss_wins_df['Instrument'] + ' (' + max_avg_consecutive_loss_wins_df['TimeFrame'] + ')'
    
    # Melting the data to make it suitable for facet grid plotting
    melted_data = max_avg_consecutive_loss_wins_df.melt(id_vars=['Instrument', 'TimeFrame', 'Instrument_TimeFrame'], 
                                                                value_vars=['MaxConLoss', 'MaxConWins', 'AvgConLoss', 'AvgConWin'], 
                                                                var_name='Metric', 
                                                                value_name='Value')

    # Plot using facet grid with scatter plot
    fig = px.scatter(melted_data, 
                     x='Instrument_TimeFrame', 
                     y='Value', 
                     color='Metric', 
                     facet_col='Metric', 
                     facet_col_wrap=2, 
                     title='Consecutive Losses and Wins Metrics by Instrument and Time Frame',
                     labels={'Instrument_TimeFrame': 'Instrument (TimeFrame)', 'Value': 'Value'},
                     symbol='Metric')

    fig.update_layout(xaxis_title='Instrument (TimeFrame)', yaxis_title='Value')
    
    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_win_loss_metrics(win_loss_metrics_df: pd.DataFrame):
    
    
    # Create a new column combining Instrument and TimeFrame
    win_loss_metrics_df['Instrument_TimeFrame'] = win_loss_metrics_df['Instrument'] + ' (' + win_loss_metrics_df['TimeFrame'] + ')'

    # Define the list of metrics to plot
    metrics = ['avg_loss', 'avg_win', 'RR', 'win_probability', 'max_loss', 'max_win']

    

    fig = make_subplots(rows=2, cols=3, subplot_titles=[metric.replace("_", " ").title() for metric in metrics])

    # Add a bar plot for each metric
    for i, metric in enumerate(metrics):
        row = i // 3 + 1
        col = i % 3 + 1

        # Calculate the average and median values for the current metric
        avg_value = win_loss_metrics_df[metric].mean()
        median_value = win_loss_metrics_df[metric].median()

        # Add bar trace
        fig.add_trace(
            go.Bar(
                x=win_loss_metrics_df['Instrument_TimeFrame'],
                y=win_loss_metrics_df[metric],
                name=metric.replace("_", " ").title(),
                marker=dict(color=win_loss_metrics_df[metric], colorscale='Plasma')
            ),
            row=row, col=col
        )

        # Add horizontal line for the average value with hover text
        fig.add_trace(
            go.Scatter(
                x=[win_loss_metrics_df['Instrument_TimeFrame'].iloc[0], win_loss_metrics_df['Instrument_TimeFrame'].iloc[-1]],
                y=[avg_value, avg_value],
                mode="lines",
                line=dict(color='red', dash='dash'),
                name=f'Avg: {avg_value:.2f}',
                hovertext=[f'Avg: {avg_value:.2f}', f'Avg: {avg_value:.2f}'],
                hoverinfo="text"
            ),
            row=row, col=col
        )

        # Add horizontal line for the median value with hover text
        fig.add_trace(
            go.Scatter(
                x=[win_loss_metrics_df['Instrument_TimeFrame'].iloc[0], win_loss_metrics_df['Instrument_TimeFrame'].iloc[-1]],
                y=[median_value, median_value],
                mode="lines",
                line=dict(color='blue', dash='dot'),
                name=f'Median: {median_value:.2f}',
                hovertext=[f'Median: {median_value:.2f}', f'Median: {median_value:.2f}'],
                hoverinfo="text"
            ),
            row=row, col=col
        )

    # Update layout
    fig.update_layout(
        height=800, 
        width=1200, 
        showlegend=False
    )
    
    # Update axis labels
    for i, metric in enumerate(metrics):
        fig['layout'][f'xaxis{i+1}'].update(title='Instrument (Time Frame)')
        fig['layout'][f'yaxis{i+1}'].update(title=metric.replace("_", " ").title())
    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_avg_trades_per_period(avg_trades_per_day):
    
    # Define periods and their multipliers
    periods = { 'Day': 1, 'Week': 7, 'Month': 30, 'Quarter': 90, 'Year': 365 }
    
    # Prepare subplots
    fig = make_subplots(rows=2, cols=3, subplot_titles=list(periods.keys()), shared_xaxes=True, shared_yaxes=True)
    
    for i, (period_name, multiplier) in enumerate(periods.items()):
        # Calculate trades per period
        avg_trades_per_period = avg_trades_per_day['avg_trades_per_day'] * multiplier
        
        # Add traces for each instrument
        for instrument in avg_trades_per_day['instrument'].unique():
            instrument_data = avg_trades_per_day[avg_trades_per_day['instrument'] == instrument]
            period_data = avg_trades_per_period[avg_trades_per_day['instrument'] == instrument]
            
            fig.add_trace(
                go.Scatter(
                    x=instrument_data['time_frame'],
                    y=period_data,
                    mode='markers',
                    marker=dict(size=8),
                    name=f"{instrument} - {period_name}"
                ),
                row=i // 3 + 1,
                col=i % 3 + 1
            )
    
    # Update layout
    fig.update_layout(
        title_text="Average Trades Per Period",
        template="plotly_white",
        height=600,
        width=900
    )
    
    # Update x-axis and y-axis labels
    fig.update_xaxes(title_text='Time Frame')
    fig.update_yaxes(title_text='Average Trades')
    
    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_expectency(expectency_df: pd.DataFrame) -> go.Figure:
    # Calculate expectancy
   

    # Create a subplot for each time frame
    time_frames = expectency_df['TimeFrame'].unique()
    fig = make_subplots(rows=len(time_frames), cols=1, subplot_titles=[f"Time Frame: {tf}" for tf in time_frames])

    for i, time_frame in enumerate(time_frames):
        # Filter data for the current time frame
        filtered_df = expectency_df[expectency_df['TimeFrame'] == time_frame]

        # Define color based on expectancy values
        colors_per_metric = {
            'expectency_per_usd': filtered_df['expectency_per_usd'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'expectency_per_trade': filtered_df['expectency_per_trade'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'expectency_per_day': filtered_df['expectency_per_day'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'expectency_per_week': filtered_df['expectency_per_week'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'expectency_per_month': filtered_df['expectency_per_month'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'expectency_per_quarter': filtered_df['expectency_per_quarter'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'expectency_per_year': filtered_df['expectency_per_year'].apply(lambda x: 'green' if x >= 0 else 'red')
        }

        # Add traces for each expectancy metric
        for metric in ['expectency_per_usd', 'expectency_per_trade', 'expectency_per_day', 
                       'expectency_per_week', 'expectency_per_month', 'expectency_per_quarter', 
                       'expectency_per_year']:
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['Instrument'],
                    y=filtered_df[metric],
                    mode='markers',
                    name=metric.replace('_', ' ').title() + f" {time_frame}",
                    line=dict(shape='linear'),
                    marker=dict(size=8, color=colors_per_metric[metric])
                ),
                row=i + 1, col=1
            )

    # Update layout
    fig.update_layout(
        height=200 * len(time_frames),
        width=1000,
        title_text="Expectency Metrics by Time Frame",
        showlegend=True
    )

    for i, time_frame in enumerate(time_frames):
        fig.update_xaxes(title_text="Instrument", row=i + 1, col=1)
        fig.update_yaxes(title_text="Expectency", row=i + 1, col=1)

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_ulcer_index(ulcer_index_df: pd.DataFrame) -> go.Figure:
    # Create the scatter plot
    fig = px.scatter(
        data_frame=ulcer_index_df,
        x='TimeFrame',  # Use TimeFrame for X-axis
        y='ulcer_index',
        color='ulcer_index',  # Use ulcer_index for color
        color_continuous_scale=px.colors.sequential.Bluered,
        title='Ulcer Index by Time Frame',
        labels={'ulcer_index': 'Ulcer Index', 'TimeFrame': 'Time Frame'},
        hover_name='Instrument'  # Show instrument names on hover
    )

    # Update layout to hide the legend and format hover information
    fig.update_layout(
        xaxis_title="Time Frame",
        yaxis_title="Ulcer Index",
        template="plotly_white",
        showlegend=False  # Hide the legend
    )
    
    # Update hover template to only show instrument name
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>Ulcer Index: %{y}<br>Time Frame: %{x}<extra></extra>'
    )

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_ulcer_index_for_each_tf(ulcer_index_for_each_tf_df: pd.DataFrame):
    # Create a bar trace for the Ulcer Index
    trace = go.Bar(
        x=ulcer_index_for_each_tf_df['TimeFrame'],
        y=ulcer_index_for_each_tf_df['ulcer_index'],
        name='Ulcer Index',
        text=ulcer_index_for_each_tf_df['ulcer_index'],
        textposition='auto'
    )

    # Create the figure and add the trace
    fig = go.Figure(data=[trace])

    # Update layout
    fig.update_layout(
        title='Ulcer Index for Each Time Frame',
        xaxis=dict(title='Time Frame'),
        yaxis=dict(title='Ulcer Index'),
        showlegend=False
    )

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_cagr(cagr_df: pd.DataFrame) -> go.Figure:
    # Define color based on whether CAGR is positive or negative
    cagr_df['Color'] = cagr_df['CAGR'].apply(lambda value: 'green' if value >= 0 else 'red')

    # Create the scatter plot
    fig = px.scatter(
        data_frame=cagr_df,
        x='TimeFrame',  # Use TimeFrame for X-axis
        y='CAGR',
        color='Color',
        color_discrete_map={'green': 'green', 'red': 'red'},
        title='CAGR by Time Frame',
        labels={'CAGR': 'CAGR (%)', 'TimeFrame': 'Time Frame'},
        hover_name='Instrument'  # Show instrument names on hover
    )

    # Update layout to hide the legend and format hover information
    fig.update_layout(
        xaxis_title="Time Frame",
        yaxis_title="CAGR (%)",
        template="plotly_white",
        showlegend=False  # Hide the legend
    )
    
    # Update hover template to only show instrument name
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>CAGR: %{y}<br>Time Frame: %{x}<extra></extra>'
    )

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_cagr_for_each_tf(cagr_for_each_tf_df: pd.DataFrame) -> go.Figure:
    # Define color based on whether CAGR is positive or negative
    cagr_for_each_tf_df['Color'] = cagr_for_each_tf_df['CAGR'].apply(lambda value: 'green' if value >= 0 else 'red')

    # Create the bar plot
    fig = px.bar(
        data_frame=cagr_for_each_tf_df,
        x='TimeFrame',  # Use TimeFrame for X-axis
        y='CAGR',
        color='Color',
        color_discrete_map={'green': 'green', 'red': 'red'},
        title='Average CAGR by Time Frame',
        labels={'CAGR': 'CAGR (%)', 'TimeFrame': 'Time Frame'},
        text='CAGR',  # Show CAGR value on bars
        hover_data={'CAGR': ':.2f'},  # Format hover data to show 2 decimal places
    )

    # Update layout to format hover information and add mean line
    fig.update_layout(
        xaxis_title="Time Frame",
        yaxis_title="Average CAGR (%)",
        template="plotly_white",
        showlegend=False  # Hide the legend
    )
    
    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_ulcer_performance_index(ulcer_performance_index_df: pd.DataFrame) -> go.Figure:
    # Define color based on the magnitude of UPI
    # You can also use a color scale for better visualization

    fig = px.scatter(
        data_frame=ulcer_performance_index_df,
        x='TimeFrame',  # Use TimeFrame for X-axis
        y='ulcer_performance_index',
        color='ulcer_performance_index',  # Use ulcer_performance_index for color
        color_continuous_scale=px.colors.sequential.Greens,  # Color scale from red to green
        title='Ulcer Performance Index by Time Frame',
        labels={'ulcer_performance_index': 'Ulcer Performance Index', 'TimeFrame': 'Time Frame'},
        hover_name='Instrument'  # Show instrument names on hover
    )

    # Update layout to hide the legend and format hover information
    fig.update_layout(
        xaxis_title="Time Frame",
        yaxis_title="Ulcer Performance Index",
        template="plotly_white",
        showlegend=False  # Hide the legend
    )
    
    # Update hover template to only show instrument name
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>Ulcer Performance Index: %{y}<br>Time Frame: %{x}<extra></extra>'
    )

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_upi_for_each_tf(upi_for_each_tf_df: pd.DataFrame) -> go.Figure:
    # Define color based on whether CAGR is positive or negative
    upi_for_each_tf_df['Color'] = upi_for_each_tf_df['ulcer_performance_index'].apply(lambda value: 'green' if value >= 0 else 'red')


    # Create the bar plot
    fig = px.bar(
        data_frame=upi_for_each_tf_df,
        x='TimeFrame',  # Use TimeFrame for X-axis
        y='ulcer_performance_index',
        color='Color',
        color_discrete_map={'green': 'green', 'red': 'red'},
        title='Average UPI by Time Frame',
        labels={'ulcer_performance_index': 'Ucler Perfoemance Index'},
        text='ulcer_performance_index',  # Show CAGR value on bars
        hover_data={'ulcer_performance_index': ':.2f'},  # Format hover data to show 2 decimal places
    )

    # Update layout to format hover information and add mean line
    fig.update_layout(
        xaxis_title="Time Frame",
        yaxis_title="Average UPI (%)",
        template="plotly_white",
        showlegend=False  # Hide the legend
    )

    return fig
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_data
def plot_all_money_management_methods(equity_curves_list, column = "cum_profit_usd" ):
    figs = []

    for equity_curves_dict in equity_curves_list:
        fig = go.Figure()

        name = None
        for method, curve in equity_curves_dict.items():
            fig.add_trace(go.Scatter(x=curve.index, y=curve[column].values, mode='lines', name=method))
            name = f"{curve['Instrument'].iloc[0]} {curve['TimeFrame'].iloc[0]}"


        fig.update_layout(
            title=f"{name} - Comparison of Money Management Methods",
            xaxis_title="Time",
            yaxis_title="Equity",
            height=600,
            width=1000
        )

        figs.append(fig)

    return figs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_monete_carlo_simulation(results_df: pd.DataFrame) -> go.Figure:
    # Get unique time frames
    time_frames = results_df['TimeFrame'].unique()
    num_rows = len(time_frames) * 2  # Two rows per time frame
    
    # Create subplot titles for values and drawdowns
    subplot_titles = []
    for tf in time_frames:
        subplot_titles.append(f"{tf} - Value")
        subplot_titles.append(f"{tf} - Drawdown")

    fig = make_subplots(
        rows=num_rows, cols=1, 
        subplot_titles=subplot_titles
    )

    for i, time_frame in enumerate(time_frames):
        # Filter data for the current time frame
        filtered_df = results_df[results_df['TimeFrame'] == time_frame]

        # Define colors
        colors = {
            'best_scenario': filtered_df['best_scenario'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'avg_scanraio': filtered_df['avg_scanraio'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'worst_scanraio': filtered_df['worst_scanraio'].apply(lambda x: 'green' if x >= 0 else 'red'),
            'max_dd': 'blue',
            'avg_dd': 'orange',
            'best_dd': 'purple'
        }

        # Add traces for scenario metrics in the first row of each time frame
        for metric in ['best_scenario', 'avg_scanraio', 'worst_scanraio']:
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['Instrument'],
                    y=filtered_df[metric],
                    mode='markers',
                    name=f"{metric} {time_frame}",
                    marker=dict(size=8, color=colors[metric])
                ),
                row=2 * i + 1, col=1
            )

        # Add traces for drawdown metrics in the second row of each time frame
        for metric in ['max_dd', 'avg_dd', 'best_dd']:
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['Instrument'],
                    y=filtered_df[metric],
                    mode='markers',
                    name=f"{metric} {time_frame}",
                    marker=dict(size=8, color=colors[metric])
                ),
                row=2 * i + 2, col=1
            )

    # Update layout
    fig.update_layout(
        height=400 * len(time_frames),  # Increased height for better readability
        width=1000,
        title_text="Monte Carlo Simulation Results by Time Frame",
        showlegend=True
    )

    for i, time_frame in enumerate(time_frames):
        fig.update_xaxes(title_text="Instrument", row=2 * i + 1, col=1)
        fig.update_yaxes(title_text="Value", row=2 * i + 1, col=1)
        fig.update_xaxes(title_text="Instrument", row=2 * i + 2, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2 * i + 2, col=1)

    return fig
# ------------------