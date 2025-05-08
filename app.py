import streamlit as st
import pandas as pd
import preprocessor,helper
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

df = pd.read_parquet('compressed.parquet')
region_df = pd.read_csv('noc_regions.csv')

df = preprocessor.preprocess(df,region_df)

st.sidebar.title('Olympic Analysis')
st.sidebar.image('https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png')

user_menu = st.sidebar.radio('Select an option',('Medal Tally','Overall Analysis','Country-wise Analysis','Athlete-wise Analysis'))


if user_menu == 'Medal Tally':
    st.sidebar.header('Medal Tally')

    years,country = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox('select year',years)
    selected_country = st.sidebar.selectbox('select country',country)

    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)

    if selected_year == "Overall" and selected_country == "Overall":
        st.title("Overall Tally")
    if selected_year != "Overall" and selected_country == "Overall":
        st.title("Medal Tally in " + str(selected_year) + " Olympics")
    if selected_year == "Overall" and selected_country != "Overall":
        st.title(selected_country + " Overall Performance")
    if selected_year != "Overall" and selected_country != "Overall":
        st.title(selected_country + " performance in " + str(selected_year) + " Olympics")

    st.table(medal_tally)

if user_menu == "Overall Analysis":
    editions = df['Year'].unique().shape[0]-1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]

    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Editions")
        st.title(editions)
    with col2:
        st.header("Hosts")
        st.title(cities)
    with col3:
        st.header("Sports")
        st.title(sports)

    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)

    nations_over_time = helper.data_over_time(df, 'region')
    fig = px.line(nations_over_time, x='Edition', y='region')
    st.title("Participating Nations over the years")
    st.plotly_chart(fig)

    events_over_time = helper.data_over_time(df, 'Event')
    fig = px.line(events_over_time, x='Edition', y='Event')
    st.title("Events over the years")
    st.plotly_chart(fig)

    athletes_over_time = helper.data_over_time(df, 'Name')
    fig = px.line(athletes_over_time, x='Edition', y='Name')
    st.title("Athletes over the years")
    st.plotly_chart(fig)

    st.title("No. of Events over time (every sport)")
    fig,ax = plt.subplots(figsize=(20,20))
    x = df.drop_duplicates(['Year','Sport','Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),annot=True)
    st.pyplot(fig)

    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')
    selected_sport = st.selectbox('select a sport',sport_list)
    x = helper.most_successful(df,selected_sport)
    st.table(x)

if user_menu == "Country-wise Analysis":
    st.sidebar.title("Country-wise Analysis")

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()
    selected_country = st.sidebar.selectbox('select a country',country_list)
    
    country_df = helper.yearwise_medal_tally(df,selected_country)

    fig = px.line(country_df, x='Year', y='Medal')
    st.title(selected_country + " Medal Tally over the years")
    st.plotly_chart(fig)

    st.title(selected_country + " excels in the following games")
    pt = helper.country_event_heatmap(df, selected_country)
    fig,ax = plt.subplots(figsize=(20,20))

    ax = sns.heatmap(pt,annot=True)
    st.pyplot(fig)


    st.title("Top 10 athletes of " + selected_country)
    Top10_df = helper.most_successful_countrywise(df,selected_country)
    st.table(Top10_df)

if user_menu == "Athlete-wise Analysis":
    st.title("Olympics Medal Age Distribution Analysis")

    medal_df = df[df['Medal'].notnull() & df['Age'].notnull()]
    # Create empty figure
    fig = go.Figure()
    # Color map for medals
    colors = {'Gold': 'gold', 'Silver': 'silver', 'Bronze': 'peru'}
    # Loop through each medal type
    for medal in ['Gold', 'Silver', 'Bronze']:
      data = medal_df[medal_df['Medal'] == medal]['Age']
      # Get KDE using seaborn
      kde = sns.kdeplot(data, bw_adjust=1)
      x_vals = kde.get_lines()[0].get_xdata()
      y_vals = kde.get_lines()[0].get_ydata()
      kde.figure.clf()  # Clear the seaborn plot so it doesn't appear
      # Add line to plotly figure
      fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode='lines',
        name=f"{medal} Medal",
        line=dict(color=colors[medal])))
      
    # Update layout
    fig.update_layout(
    title='Age Distribution of Medal Winners by Medal Type',
    xaxis_title='Age',
    yaxis_title='Density (Smoothed)',
    template='simple_white',
    height=500)

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


    st.title("Distribution of Age wrt Sports (Gold Medalist)")
    # Filter only gold medal winners with known age
    gold_df = df[(df['Medal'] == 'Gold') & (df['Age'].notnull())]

    # Select sports with sufficient data (e.g. 30+ gold medalists)
    sport_counts = gold_df['Sport'].value_counts()
    valid_sports = sport_counts[sport_counts > 30].index.tolist()

    fig = go.Figure()

    for sport in valid_sports:
        sport_data = gold_df[gold_df['Sport'] == sport]['Age']
        # Generate KDE using seaborn
        kde = sns.kdeplot(sport_data, bw_adjust=1)
        x_vals = kde.get_lines()[0].get_xdata()
        y_vals = kde.get_lines()[0].get_ydata()
        kde.figure.clf()  # Clear figure so seaborn doesn't output it

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name=sport,
            hovertemplate=f"{sport}<br>Age: %{{x:.2f}}<br>Density: %{{y:.3f}}"
        ))

    fig.update_layout(
        title="Distribution of Age wrt Sports (Gold Medalist)",
        xaxis_title="Age",
        yaxis_title="Density",
        template="simple_white",
        height=600,
        legend=dict(
            title="Sport",
            orientation="v",
            font=dict(size=10),
            x=1.05
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.title("Height Vs Weight")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')
    selected_sport = st.selectbox('select a sport',sport_list)
    temp_df = helper.weight_v_height(df, selected_sport)
    fig,ax = plt.subplots()
    ax = sns.scatterplot(temp_df,x="Weight", y="Height", hue="Medal", style="Sex", s=60)

    st.pyplot(fig)

    st.title("Men Vs Women participation over the years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x="Year", y=['Male','Female'])
    st.plotly_chart(fig)


    



