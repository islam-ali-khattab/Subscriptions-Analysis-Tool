import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


# Page configuration
st.set_page_config(
    page_title="Subscription Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Annual exchange rates for Egyptian Pound (hypothetical - need actual data)
EXCHANGE_RATES = {
    2021: {
        'EGP': 1.0, 'USD': 15.8,   'EUR': 18.5,   'GBP': 22.0,  'AED': 4.3,  'SAR': 4.2,    'MAD': 1.7
    },
    2022: {
        'EGP': 1.0, 'USD': 24.5,   'EUR': 27.0,   'GBP': 30.5,  'AED': 6.7,  'SAR': 6.6,    'MAD': 2.5
    },
    2023: {
        'EGP': 1.0, 'USD': 30.0,   'EUR': 33.5,   'GBP': 37.5,  'AED': 7.6,  'SAR': 7.5,    'MAD': 3.1
    },
    2024: {
        'EGP': 1.0, 'USD': 45.3,   'EUR': 51.0,   'GBP': 57.0,  'AED': 11.8, 'SAR': 11.5,   'MAD': 4.6
    },
    2025: {
        'EGP': 1.0, 'USD': 50.0,   'EUR': 56.0,   'GBP': 63.0,  'AED': 13.0, 'SAR': 12.8,   'MAD': 5.5
    }
}

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv') 
    
    # Convert date columns
    date_columns = ['subscribtion_date', 'activation_date', 'expiration_date']
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])
    
    # Add useful columns for analysis
    data['subscription_year'] = data['subscribtion_date'].dt.year
    data['subscription_month'] = data['subscribtion_date'].dt.month_name()
    data['subscription_quarter'] = data['subscribtion_date'].dt.quarter
    data['activation_year'] = data['activation_date'].dt.year
    data['activation_delay_days'] = (data['activation_date'] - data['subscribtion_date']).dt.days
    data['subscription_duration_days'] = (data['expiration_date'] - data['activation_date']).dt.days
    data['is_active'] = data['expiration_date'] > pd.Timestamp.now()
    
    # Convert all financial amounts to Egyptian Pound
    data['paid_amount_egp'] = data.apply(
        lambda row: row['paid_amount'] * EXCHANGE_RATES[row['subscription_year']][row['currency']], 
        axis=1
    )
    data['base_price_egp'] = data.apply(
        lambda row: row['base_price'] * EXCHANGE_RATES[row['subscription_year']][row['currency']], 
        axis=1
    )
    data['discounted_amount_egp'] = data.apply(
        lambda row: row['discounted_amount'] * EXCHANGE_RATES[row['subscription_year']][row['currency']], 
        axis=1
    )
    
    # Define cohorts (monthly based on subscription date)
    data['cohort_month'] = data['subscribtion_date'].dt.to_period('M')
    data['cohort_index'] = data.groupby('user_id')['subscribtion_date'].rank(method='dense')

    # Calculate key metrics for each subscription
    data['activation_delay_days'] = (data['activation_date'] - data['subscribtion_date']).dt.days
    data['subscription_duration_days'] = (data['expiration_date'] - data['activation_date']).dt.days
    
    data['revenue_category'] = pd.cut(data['paid_amount_egp'], 
                                    bins=[0, 500, 1000, 2000, 5000, 10000, 50000], 
                                    labels=['0-500', '500-1000', '1000-2000', '2000-5000', '5000-10000', '10000+'])
    
    
    return data

# Function to calculate trend compared to previous year
def calculate_trend(current_data, previous_data):  
    if previous_data == 0 and current_data:
        return -100
    
    if previous_data == 0 and current_data == 0:
        return 0

    
    trend_percentage = ((current_data - previous_data) / previous_data) * 100
 

      
    return trend_percentage

# Function to get a specific year data
def get_specific_data(df, specific_year, current_month):
    specific_year_data = df[
        (df['subscribtion_date'].dt.year == specific_year) &
        (df['subscribtion_date'].dt.month <= current_month)
    ]
    return specific_year_data

def create_overview_tab(df):
    st.header("🏠 Overview Dashboard")
    
    # Get current & prvious year data
    if not df.empty:
        current_year = datetime.now().year
        current_month = datetime.now().month
        previous_year = current_year - 1
        current_year_data = get_specific_data(df, current_year, current_month)
        previous_year_data = get_specific_data(df, previous_year, current_month)

    # Calculate metrics for current period
    total_subscriptions = len(current_year_data)
    total_revenue_egp = current_year_data['paid_amount_egp'].sum()
    active_subscriptions = current_year_data['is_active'].sum()
    inactive_subscriptions = df[df['activation_date'].isna()].shape[0]
    avg_revenue_egp = current_year_data['paid_amount_egp'].mean()
    avg_discount = current_year_data['discount_percentage'].mean()
    # activation_conversion_rate = (current_year_data['activation_date'].notna().sum() / len(current_year_data)) * 100 if len(current_year_data) > 0 else 0
    avg_activation_delay = current_year_data['activation_delay_days'].mean()
    unique_customers = current_year_data['user_id'].nunique()
    
    # Calculate metrics for previous period for trends
    if not previous_year_data.empty:
        prev_total_subscriptions = len(previous_year_data)
        prev_total_revenue_egp = previous_year_data['paid_amount_egp'].sum()
        prev_active_subscriptions = previous_year_data[previous_year_data['activation_year'] == previous_year]["activation_year"].count()
        prev_inactive_subscriptions = df[df['activation_date'].isna()].shape[0]
        prev_avg_revenue_egp = previous_year_data['paid_amount_egp'].mean()
        prev_avg_discount = previous_year_data['discount_percentage'].mean()
        # prev_activation_conversion_rate = (previous_year_data['activation_date'].notna().sum() / len(previous_year_data)) * 100 if len(previous_year_data) > 0 else 0
        prev_avg_activation_delay = previous_year_data['activation_delay_days'].mean()
        prev_unique_customers = previous_year_data['user_id'].nunique()
    else:
        prev_total_subscriptions = 0
        prev_total_revenue_egp = 0
        prev_active_subscriptions = 0
        prev_inactive_subscriptions = 0
        prev_avg_revenue_egp = 0
        prev_avg_discount = 0
        # prev_activation_conversion_rate = 0
        prev_avg_activation_delay = 0
        prev_unique_customers = 0
    
    # Calculate trends
    subscriptions_trend = calculate_trend(total_subscriptions, prev_total_subscriptions)
    revenue_trend = calculate_trend(total_revenue_egp, prev_total_revenue_egp)
    active_trend = calculate_trend(active_subscriptions, prev_active_subscriptions)
    inactive_trend = calculate_trend(inactive_subscriptions, prev_inactive_subscriptions)
    avg_revenue_trend = calculate_trend(avg_revenue_egp, prev_avg_revenue_egp)
    discount_trend = calculate_trend(avg_discount, prev_avg_discount)
    # activation_conversion_trend = calculate_trend(activation_conversion_rate, prev_activation_conversion_rate)
    activation_trend = calculate_trend(avg_activation_delay, prev_avg_activation_delay)
    customers_trend = calculate_trend(unique_customers, prev_unique_customers)

    # Key Performance Indicators with trends
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Revenue (EGP)", 
            f"EGP {total_revenue_egp:,.0f}",
            delta=f"{revenue_trend:+.1f}% vs last year",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Active Subscriptions", 
            f"{active_subscriptions:,}",
            delta=f"{active_trend:+.1f}% vs last year",
            delta_color="normal"
        )

    with col3:
        st.metric(
            "Unique Customers", 
            f"{unique_customers:,}",
            delta=f"{customers_trend:+.1f}% vs last year",
            delta_color="normal" 
        )

    with col4:
        st.metric(
            label="Total Subscriptions", 
            value=f"{total_subscriptions:,}",
            delta=f"{subscriptions_trend:.1f}% vs last year",   
            delta_color="normal"                  
        )
    
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Avg Revenue per Subscription (EGP)", 
            f"EGP {avg_revenue_egp:,.0f}",
            delta=f"{avg_revenue_trend:+.1f}% vs last year",
            delta_color="normal" 
        )
        
    with col6:
        st.metric(
            "Average Discount", 
            f"{avg_discount:.1f}%",
            delta=f"{discount_trend:+.1f}% vs last year",
            delta_color="normal" 
        )   

    with col7:
        st.metric(
            "Average Activation Delay", 
            f"{avg_activation_delay:.1f} days",
            delta=f"{activation_trend:+.1f}% vs last year",
            delta_color="normal" 
        )
        

    with col8:
        st.metric(
            "Inactive Subscriptions", 
            f"{inactive_subscriptions:.1f}%",
            delta=f"{inactive_trend:+.1f}% vs last year",
            delta_color="normal"
        )

    st.markdown(
        """
        <div style="margin-top: 50px;"></div>
        """,
        unsafe_allow_html=True
    )


    # First row of charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Yearly Subscription Trend")
        yearly_trend = df.groupby(df['subscribtion_date'].dt.year).size()
        yearly_trend.index = yearly_trend.index.astype(int).astype(str)
        
        fig = px.line(
            x=yearly_trend.index, 
            y=yearly_trend.values, 
            labels={'x': 'Year', 'y': 'Number of Subscriptions'},
        )
        fig.update_xaxes(type='category')
        fig.update_traces(textposition='top center')  # Position the numbers
        st.plotly_chart(fig, use_container_width=True, key="yearly_trend_chart")

    with col2:
        st.subheader("💰 Yearly Revenue Trend (EGP)")
        yearly_revenue = df.groupby(df['subscribtion_date'].dt.year)['paid_amount_egp'].sum()
        yearly_revenue.index = yearly_revenue.index.astype(int).astype(str)
        
        fig = px.bar(
            x=yearly_revenue.index, 
            y=yearly_revenue.values,
            labels={'x': 'Year', 'y': 'Revenue (EGP)'},
            text=yearly_revenue.values  # Add this line
        )
        fig.update_xaxes(type='category')
        fig.update_traces(texttemplate='%{text:,.0f}', textposition='outside')  # Format numbers
        st.plotly_chart(fig, use_container_width=True, key="yearly_revenue_chart")
    

    # Second row of charts
    col1, col2 = st.columns(2)
    
    # Second row of charts - Country Distribution (All, Active, Expired)
    st.subheader("🌍 Subscriptions by Country")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**All Subscriptions**")
        country_data_all = df['country'].value_counts()
        fig = px.pie(values=country_data_all.values, names=country_data_all.index)
        st.plotly_chart(fig, use_container_width=True, key="country_all_pie_chart")
    
    with col2:
        st.markdown("**Active Subscriptions**")
        active_df = df[df['is_active'] == True]
        country_data_active = active_df['country'].value_counts()
        fig = px.pie(values=country_data_active.values, names=country_data_active.index)
        st.plotly_chart(fig, use_container_width=True, key="country_active_pie_chart")
    
    with col3:
        st.markdown("**Expired Subscriptions**")
        expired_df = df[df['is_active'] == False]
        country_data_expired = expired_df['country'].value_counts()
        fig = px.pie(values=country_data_expired.values, names=country_data_expired.index)
        st.plotly_chart(fig, use_container_width=True, key="country_expired_pie_chart")

    # Third row of charts - Plan & Payment Methods Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Most Popular Plans")
        plan_data = df['plan'].value_counts().sort_values(ascending=True)
        fig = px.bar(x=plan_data.values, y=plan_data.index, orientation='h',
                    labels={'x': 'Number of Subscriptions', 'y': 'Plan Type'},
                    text=plan_data.values 
                    )
        fig.update_traces(textfont=dict(color='white', size=16))
        st.plotly_chart(fig, use_container_width=True, key="plan_bar_chart")
    
    with col2:
        st.subheader("💳 Payment Methods")
        payment_data = df['payment_method'].value_counts()
        fig = px.bar(
            x=payment_data.index, 
            y=payment_data.values,
            labels={'x': 'Payment Method', 'y': 'Count'},
            text=payment_data.values 
        )
        fig.update_traces(textposition='outside')  
        st.plotly_chart(fig, use_container_width=True, key="payment_bar_chart")


def create_financial_tab(filtered_df):
    st.header("💵 Financial Analysis")

    # Calculate monthly revenue
    monthly_revenue = filtered_df.groupby(filtered_df['subscribtion_date'].dt.to_period('M'))['paid_amount_egp'].sum()
    monthly_revenue.index = monthly_revenue.index.astype(str)
    monthly_revenue_df = monthly_revenue.reset_index()
    monthly_revenue_df.columns = ['month', 'revenue']
    monthly_revenue_df['month_dt'] = pd.to_datetime(monthly_revenue_df['month'])
    monthly_revenue_df['month_name'] = monthly_revenue_df['month_dt'].dt.strftime('%b %Y')

    # Calculate 3-month moving average for trend line
    monthly_revenue_df['moving_avg'] = monthly_revenue_df['revenue'].rolling(window=3, min_periods=1).mean()

    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_monthly = monthly_revenue_df['revenue'].mean()
        st.metric("Average Monthly Revenue", f"EGP {avg_monthly:,.0f}")

    with col2:
        max_monthly = monthly_revenue_df['revenue'].max()
        st.metric("Highest Monthly Revenue", f"EGP {max_monthly:,.0f}")

    with col3:
        growth = ((monthly_revenue_df['revenue'].iloc[-1] - monthly_revenue_df['revenue'].iloc[0]) / monthly_revenue_df['revenue'].iloc[0]) * 100
        st.metric("Overall Growth", f"{growth:+.1f}%")

    st.markdown(
        """
        <div style="margin-top: 50px;"></div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("📈 Monthly Revenue Analysis (EGP)")

    fig = go.Figure()

    # Actual revenue line
    fig.add_trace(go.Scatter(
        x=monthly_revenue_df['month_dt'],
        y=monthly_revenue_df['revenue'],
        name='Monthly Revenue',
        mode='lines+markers',
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=6),
        hovertemplate='<b>%{x|%b %Y}</b><br>Revenue: EGP %{y:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title="Monthly Revenue Trend",
        xaxis=dict(title='Month', tickformat='%b %Y'),
        yaxis=dict(title='Revenue (EGP)'),
        hovermode='x unified',
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)


        
    st.subheader("🎯 Discount Impact Analysis")
    
    discount_data = filtered_df.groupby('discount_percentage').agg({
        'paid_amount_egp': ['sum', 'mean', 'count'],
        'base_price_egp': ['sum', 'mean']  # Added total base price
    }).round(0)
    
    # Flatten the column names
    discount_data.columns = [
        'total_revenue', 
        'avg_revenue', 
        'subscription_count', 
        'total_base_price',  # New: Total base price
        'avg_base_price'
    ]
    
    # Format large numbers for display
    def format_large_number(num):
        if num >= 1_000_000:
            return f'{num/1_000_000:.1f}M'
        elif num >= 1_000:
            return f'{num/1_000:.1f}K'
        else:
            return str(num)
    
    fig = go.Figure()
    
    # Bar chart for Subscription Count
    fig.add_trace(go.Bar(
        x=discount_data.index,
        y=discount_data['subscription_count'],
        name='Subscription Count',
        text=[format_large_number(x) for x in discount_data['subscription_count']],
        textposition='outside',
        yaxis='y',
        opacity=0.6,
        marker_color='#1f77b4'
    ))
    
    # Lines for revenue metrics
    fig.add_trace(go.Scatter(
        x=discount_data.index,
        y=discount_data['total_revenue'],
        name='Total Revenue (EGP)',
        yaxis='y2',
        mode='lines+markers+text',
        text=[format_large_number(x) for x in discount_data['total_revenue']],
        textposition='top center',
        line=dict(width=3, color='#ff7f0e'),
        marker=dict(size=8, color='#ff7f0e')
    ))
    
    # New line for Total Base Price
    fig.add_trace(go.Scatter(
        x=discount_data.index,
        y=discount_data['total_base_price'],
        name='Total Base Price (EGP)',
        yaxis='y2',
        mode='lines+markers+text',
        text=[format_large_number(x) for x in discount_data['total_base_price']],
        textposition='top right',
        line=dict(width=3, color='#9467bd', dash='dashdot'),
        marker=dict(size=8, color='#9467bd', symbol='star')
    ))

    fig.add_trace(go.Scatter(
        x=discount_data.index,
        y=discount_data['avg_revenue'],
        name='Avg Revenue (EGP)',
        yaxis='y2',
        mode='lines+markers+text',
        text=[format_large_number(x) for x in discount_data['avg_revenue']],
        textposition='bottom center',
        line=dict(width=3, color='#2ca02c', dash='dot'),
        marker=dict(size=8, color='#2ca02c')
    ))
    
    # Line for Average Base Price
    fig.add_trace(go.Scatter(
        x=discount_data.index,
        y=discount_data['avg_base_price'],
        name='Avg Base Price (EGP)',
        yaxis='y2',
        mode='lines+markers+text',
        text=[format_large_number(x) for x in discount_data['avg_base_price']],
        textposition='middle center',
        line=dict(width=3, color='#d62728', dash='dash'),
        marker=dict(size=8, color='#d62728', symbol='diamond')
    ))
    
    fig.update_layout(
        xaxis=dict(title='Discount Percentage (%)'),
        yaxis=dict(title='Subscription Count', side='left'),
        yaxis2=dict(title='Revenue & Price (EGP)', side='right', overlaying='y'),
        legend=dict(x=0, y=1.2, orientation='h'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("💳 Payment Methods Performance")

    col1, col2 = st.columns(2)

    with col1:
        payment_data = filtered_df.groupby('payment_method').agg({
            'paid_amount_egp': 'sum',
            'user_id': 'count',
            'discount_percentage': 'mean'
        }).sort_values('paid_amount_egp', ascending=False)
        
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
            x=payment_data.index,
            y=payment_data['paid_amount_egp'],
            mode='markers+text',
            marker=dict(
                size=25,
                color='#1f77b4',
                line=dict(width=2, color='darkblue')
            ),
            text=payment_data['paid_amount_egp'].apply(lambda x: f'EGP {x:,.0f}'),
            textposition='top center',
            textfont=dict(color='white', size=11, family="Arial Black"),  
            hovertemplate='<b>%{x}</b><br>Revenue: EGP %{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Revenue by Payment Method",
            xaxis=dict(
                title='Payment Method',
                tickangle=0, 
                tickfont=dict(size=12)
            ),
            yaxis=dict(title='Total Revenue (EGP)'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col2:

        # Bar chart for subscription count
        fig1 = px.bar(
            x=payment_data.index,
            y=payment_data['user_id'],
            title="Subscription Count by Payment Method",
            labels={'x': 'Payment Method', 'y': 'Number of Subscriptions'},
            text_auto=True,
        )
        fig1.update_traces(marker_color='#1f77b4')
        st.plotly_chart(fig1, use_container_width=True)
        
    st.subheader("📊 Plan Performance Overview")
    # Financial charts
    col1, col2 = st.columns(2)
    
    with col1:
        revenue_by_plan = filtered_df.groupby('plan')['paid_amount_egp'].sum().sort_values(ascending=False)

        # Create pie chart
        fig = px.pie(
            title="Revenue by Plan (EGP)",
            values=revenue_by_plan.values,
            names=revenue_by_plan.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )

        # Add percentage labels and formatting
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Revenue: EGP %{value:,.0f}<br>Percentage: %{percent}<extra></extra>'
        )

        # Improve layout
        fig.update_layout(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True, key="revenue_plan_pie_chart")

        with col2:
            subscriptions_count_by_plan = filtered_df.groupby('plan')['paid_amount_egp'].count().sort_values(ascending=False)
            fig = px.bar(x=subscriptions_count_by_plan.index, y=subscriptions_count_by_plan.values,
                        title="Subscriptions count by Plan",
                        labels={'x': '', 'y': ''},
                        text=subscriptions_count_by_plan)
            # Format text and position
            fig.update_traces(
                texttemplate='%{text:,}',   # add comma formatting
                textposition='outside'      # show outside bar
            )

            st.plotly_chart(fig, use_container_width=True, key="revenue_plan_chart")

    st.subheader("🌐 Country Performance Overview")

    # Financial charts
    col1, col2 = st.columns(2)

    with col1:

        revenue_by_country = (
            filtered_df.groupby('country')['paid_amount_egp']
            .sum()
            .sort_values(ascending=False)
        )

        fig_pie = px.pie(
            title="Revenue by Country (EGP)",
            values=revenue_by_country.values,
            names=revenue_by_country.index,
            hole=0.3,  # donut style
        )

        # Improve text readability
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_traces(textfont=dict(size=14, color="white"))

        st.plotly_chart(fig_pie, use_container_width=True, key="revenue_country_pie")

    
    with col2:
        
        revenue_by_country = (
            filtered_df.groupby('country')['paid_amount_egp']
            .count()
            .sort_values(ascending=True)
        )
        
        fig_bar = px.bar(
            title="Subscriptions count by Country",
            x=revenue_by_country.values,
            y=revenue_by_country.index,
            orientation='h',
            labels={'x': '', 'y': ''},
            text=revenue_by_country
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_traces(textfont=dict(size=12, color="white"))
        
        st.plotly_chart(fig_bar, use_container_width=True, key="revenue_country_bar")

    st.subheader("💹 Original Currency vs EGP Conversion")


    
    col1, col2 = st.columns(2)

    with col1:
            
        st.markdown(
            """
            <div style="margin-top: 50px;"></div>
            """,
            unsafe_allow_html=True
        )
        
        sample_conversion = filtered_df[['currency', 'paid_amount', 'paid_amount_egp']].groupby("currency").sum().sort_values(by="paid_amount_egp", ascending=False)
        st.write(sample_conversion, use_container_width=True)
        
    with col2:
        fig = px.pie(
            sample_conversion.reset_index(),
            values='paid_amount_egp',
            names='currency',
            hole=0.3,  # donut style optional
        )

        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)


def create_plan_analysis_tab(filtered_df):

    st.header("📋 Plan & Subscription Analysis")

    st.markdown(
        """
        <div style="margin-top: 50px;"></div>
        """,
        unsafe_allow_html=True
    )

    # Detailed plan analysis
    st.subheader("📊 Plan Comparison Analysis")

    plan_comparison = filtered_df.groupby('plan').agg({
        'paid_amount_egp': ['sum', 'mean', 'count'],
        'base_price_egp': 'mean',
        'discount_percentage': 'mean',
        'subscription_duration_days': 'mean',
        'user_id': 'nunique'
        }).round(0)

    plan_comparison.columns = [
        'Total Revenue (EGP)', 'Avg Revenue (EGP)', 'Subscription Count',
        'Avg Base Price (EGP)', 'Avg Discount %', 
        'Avg Subscription Duration (Days)', 'Unique Customers'
    ]
    
    st.dataframe(plan_comparison, use_container_width=True)
    
    # Plan comparison charts
    col1, col2 = st.columns(2)
    
    
    with col1:
        st.subheader("📈 Average Revenue Per Plan (EGP)")
        
        avg_prices = filtered_df.groupby('plan')['paid_amount_egp'].mean().sort_values(ascending=False)
        
        fig = px.bar(x=avg_prices.index, y=avg_prices.values,
                    labels={'x': 'Plan Type', 'y': 'Average Price (EGP)'})
        st.plotly_chart(fig, use_container_width=True, key="avg_price_chart")
    
    with col2:
        st.subheader("💰 Revenue Distribution by Plan")

        revenue_by_plan = plan_comparison['Total Revenue (EGP)']

        fig = px.pie(
            names=revenue_by_plan.index,
            values=revenue_by_plan.values,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    
    # Conversion rate analysis by plan
    st.subheader("📊 Average Activation Delay by Plan")

    activation_analysis = filtered_df.groupby('plan')['activation_delay_days'].mean().reset_index().sort_values(by='activation_delay_days', ascending=False)


    col1, col2 = st.columns(2)


    with col1:
        # Column chart
        fig = px.bar(
            activation_analysis,
            x='plan',
            y='activation_delay_days',
            text='activation_delay_days',  # Show values on top
            labels={'plan': 'Plan', 'activation_delay_days': 'Avg Activation Delay (Days)'},
        )

        # Format the text labels
        fig.update_traces(
            texttemplate='%{text:.1f} days',
            textposition='outside',
            marker_color='indianred'
        )

        # Improve layout
        fig.update_layout(
            yaxis=dict(title='Avg Activation Delay (Days)'),
            xaxis=dict(title='Plan'),
            uniformtext_minsize=12,
            uniformtext_mode='hide'
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:  
        
        st.markdown(
            """
            <div style="margin-top: 100px;"></div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(activation_analysis, use_container_width=True)



def cohort_analysis_tab(filtered_df):

    st.header("⏳ Cohort Analysis")

    st.markdown(
    """
    <div style="margin-top: 50px;"></div>
    """,
    unsafe_allow_html=True
)

    # Sort cohorts
    cohorts = sorted(filtered_df['cohort_month'].unique())

    
    # Create retention matrix
    retention_matrix = pd.DataFrame(index=cohorts, columns=range(13))


    for cohort in cohorts:
        cohort_data = filtered_df[filtered_df['cohort_month'] == cohort]
        retention_matrix.loc[cohort, 0] = 100  # Month 0 = 100%
        
        for month in range(1, 13):
            cutoff_date = cohort_data['subscribtion_date'].min() + pd.DateOffset(months=month)
            retained_users = cohort_data[cohort_data['expiration_date'] >= cutoff_date]['user_id'].nunique()
            retention_rate = (retained_users / len(cohort_data)) * 100
            retention_matrix.loc[cohort, month] = retention_rate

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        retention_matrix.astype(float),
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        ax=ax
    )
    ax.set_title('Cohort Retention Rates Over Time')
    ax.set_ylabel('Cohort Month')
    ax.set_xlabel('Months After Subscription')
    
    # Display in Streamlit
    st.pyplot(fig)

def main():
    st.title("📊 Subscription Analytics Dashboard")
    st.markdown("Comprehensive subscription performance and customer analytics tool")
    
    # Load data
    df = load_data()

    # Create tabs
    tab = st.radio(
        "",
        ["🏠 Overview", "💵 Financial Analysis", "📋 Plan Analysis", "⏳ Cohort Analysis"],
        horizontal=True
    )
    
    if tab != "🏠 Overview":        
        # Year filter - default to current year
        current_year = datetime.now().year
        available_years = sorted(df['subscription_year'].unique())

                
        # ===============================
        # Sidebar Filters
        # ===============================
        st.sidebar.header("🔍 Filter Options")

        # --- Date Range Filter ---
        st.sidebar.subheader("📅 Date Range Filter")

        date_preset = st.sidebar.radio(
            "Quick Date Presets:",
            ["Full Year", "Custom Range"]
        )

        if date_preset == "Full Year":
            selected_year = st.sidebar.selectbox(
                "Select Year:", 
                options=available_years,
                index=available_years.index(current_year) if current_year in available_years else 0
            )
            start_date = datetime(selected_year, 1, 1)
            end_date = datetime(selected_year, 12, 31)

        elif date_preset == "Custom Range":
            min_date = df['subscribtion_date'].min().date()
            max_date = df['subscribtion_date'].max().date()

            start_date = st.sidebar.date_input(
                "Start Date:",
                min_value=min_date,
                max_value=max_date,
                value=min_date
            )
            end_date = st.sidebar.date_input(
                "End Date:",
                min_value=start_date,
                max_value=max_date,
                value=max_date
            )

        # --- Country Filter ---
        countries = ['All Countries'] + sorted(df['country'].dropna().unique().tolist())
        selected_country = st.sidebar.selectbox("🌍 Country:", countries)

        # --- Plan Filter ---
        plans = ['All Plans'] + sorted(df['plan'].dropna().unique().tolist())
        selected_plan = st.sidebar.selectbox("📦 Plan Type:", plans)

        # --- Subscription Status Filter ---
        status_options = ['All', 'Active Only', 'Expired Only']
        selected_status = st.sidebar.selectbox("📌 Subscription Status:", status_options)

        # --- Currency Filter ---
        currencies = ['All Currencies'] + sorted(df['currency'].dropna().unique().tolist())
        selected_currency = st.sidebar.selectbox("💱 Original Currency:", currencies)

        # ===============================
        # Apply Filters
        # ===============================
        filtered_df = df.copy()

        # Filter by date range
        filtered_df = filtered_df[
            (filtered_df['subscribtion_date'] >= pd.to_datetime(start_date)) &
            (filtered_df['subscribtion_date'] <= pd.to_datetime(end_date))
        ]

        # Filter by country
        if selected_country != 'All Countries':
            filtered_df = filtered_df[filtered_df['country'] == selected_country]

        # Filter by plan
        if selected_plan != 'All Plans':
            filtered_df = filtered_df[filtered_df['plan'] == selected_plan]

        # Filter by subscription status
        if selected_status == 'Active Only':
            filtered_df = filtered_df[filtered_df['status'] == 'Active']
        elif selected_status == 'Expired Only':
            filtered_df = filtered_df[filtered_df['status'] == 'Expired']

        # Filter by currency
        if selected_currency != 'All Currencies':
            filtered_df = filtered_df[filtered_df['currency'] == selected_currency]
        
        # Additional information in sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Quick Statistics")
        st.sidebar.metric("Filtered Records", f"{len(filtered_df):,}")
        st.sidebar.metric("Data Coverage", f"{(len(filtered_df)/len(df)*100):.1f}%")
        
    
    if tab == "🏠 Overview":
        create_overview_tab(df)
    
    if tab == "💵 Financial Analysis":
        create_financial_tab(filtered_df)
    
    if tab == "📋 Plan Analysis":
        create_plan_analysis_tab(filtered_df)
    
    if tab == "⏳ Cohort Analysis":
        cohort_analysis_tab(filtered_df)
    
   

if __name__ == "__main__":
    main()