# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 22:39:19 2026

@author: pavani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset with correct encoding
df = pd.read_csv("Global YouTube Statistics.csv", encoding="latin1")

#Preview data
print(df.head())
print("\nColumns:")
print(df.columns)

print("\nShape:", df.shape)

# Clean column names for clarity
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "")
    .str.replace(")", "")
)

print(df.columns)

numeric_cols = [
    "subscribers",
    "video_views",
    "uploads",
    "video_views_for_the_last_30_days",
    "lowest_monthly_earnings",
    "highest_monthly_earnings",
    "lowest_yearly_earnings",
    "highest_yearly_earnings",
    "subscribers_for_last_30_days",
    "population",
    "urban_population",
    "unemployment_rate",
    "latitude",
    "longitude"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(df.isnull().sum().sort_values(ascending=False).head(10))

# =========================
# QUESTION 1
# Top 10 YouTube Channels by Subscribers
# =========================

top_10_channels = (
    df[["youtuber", "subscribers"]]
    .dropna()
    .sort_values(by="subscribers", ascending=False)
    .head(10)
)

print(top_10_channels)

#Visualisation
plt.figure()
sns.barplot(
    data=top_10_channels,
    x="subscribers",
    y="youtuber"
)
plt.title("Top 10 YouTube Channels by Subscribers")
plt.show()

# =========================
# QUESTION 2
# Category with highest average subscribers
# =========================

avg_subs_by_category = (
    df.groupby("category")["subscribers"]
    .mean()
    .sort_values(ascending=False)
)

print(avg_subs_by_category.head(10))

#Visualisation
plt.figure()
avg_subs_by_category.head(10).plot(kind="bar")
plt.title("Top Categories by Average Subscribers")
plt.xlabel("Category")
plt.ylabel("Average Subscribers")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================
# QUESTION 3
# Average number of uploads per category
# =========================

avg_uploads_by_category = (
    df.groupby("category")["uploads"]
    .mean()
    .sort_values(ascending=False)
)

print(avg_uploads_by_category.head(10))

#Visualisation
plt.figure()
avg_uploads_by_category.head(10).plot(kind="bar")
plt.title("Average Number of Uploads per Category")
plt.xlabel("Category")
plt.ylabel("Average Uploads")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================
# QUESTION 4
# Top 5 countries by number of YouTube channels
# =========================

top_countries = (
    df["country"]
    .value_counts()
    .head(5)
)

print(top_countries)

#Visualisation
plt.figure()
top_countries.plot(kind="bar")
plt.title("Top 5 Countries by Number of YouTube Channels")
plt.xlabel("Country")
plt.ylabel("Number of Channels")
plt.tight_layout()
plt.show()

# =========================
# QUESTION 5
# Distribution of channel types across categories
# =========================

channel_type_dist = pd.crosstab(df["category"], df["channel_type"])

print(channel_type_dist.head())

#Visualisation
plt.figure(figsize=(10, 6))
sns.heatmap(channel_type_dist, cmap="Blues")
plt.title("Distribution of Channel Types Across Categories")
plt.xlabel("Channel Type")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

# =========================
# QUESTION 6
# Correlation between subscribers and video views
# =========================

corr_value = df["subscribers"].corr(df["video_views"])
print("Correlation between subscribers and video views:", corr_value)

#Visualisation
plt.figure()
sns.scatterplot(
    data=df,
    x="subscribers",
    y="video_views"
)
plt.title("Subscribers vs Video Views")
plt.xlabel("Subscribers")
plt.ylabel("Total Video Views")
plt.show()

# =========================
# QUESTION 7
# Monthly earnings variation across categories
# =========================

df["avg_monthly_earnings"] = (
    df["lowest_monthly_earnings"] + df["highest_monthly_earnings"]
) / 2

monthly_earnings_by_category = (
    df.groupby("category")["avg_monthly_earnings"]
    .mean()
    .sort_values(ascending=False)
)

print(monthly_earnings_by_category.head(10))

#Visualisation
plt.figure()
monthly_earnings_by_category.head(10).plot(kind="bar")
plt.title("Average Monthly Earnings by Category")
plt.xlabel("Category")
plt.ylabel("Average Monthly Earnings")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# =========================
# QUESTION 8
# Subscribers gained in last 30 days
# =========================

print(df["subscribers_for_last_30_days"].describe())

#Visualisation
plt.figure()
sns.histplot(df["subscribers_for_last_30_days"], bins=30)
plt.title("Distribution of Subscribers Gained in Last 30 Days")
plt.xlabel("Subscribers Gained (Last 30 Days)")
plt.ylabel("Frequency")
plt.show()

# =========================
# QUESTION 9
# Outliers in yearly earnings
# =========================

df["avg_yearly_earnings"] = (
    df["lowest_yearly_earnings"] + df["highest_yearly_earnings"]
) / 2

#Visualisation
plt.figure()
sns.boxplot(y=df["avg_yearly_earnings"])
plt.title("Outliers in Yearly Earnings of YouTube Channels")
plt.ylabel("Average Yearly Earnings")
plt.show()

# =========================
# QUESTION 10
# Distribution of channel creation dates
# =========================

df["created_year"] = pd.to_numeric(df["created_year"], errors="coerce")

channels_by_year = (
    df["created_year"]
    .value_counts()
    .sort_index()
)

print(channels_by_year.head())

#Visualisation
plt.figure()
channels_by_year.plot()
plt.title("Trend of YouTube Channel Creation Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Channels")
plt.show()

# =========================
# QUESTION 11
# Education enrollment vs number of channels
# =========================

country_channels = df.groupby("country").agg(
    channel_count=("youtuber", "count"),
    education=("gross_tertiary_education_enrollment_%", "mean")
)

print(country_channels.corr())

#Visualisation
plt.figure()
sns.scatterplot(
    data=country_channels,
    x="education",
    y="channel_count"
)
plt.title("Education Enrollment vs Number of YouTube Channels")
plt.show()

# =========================
# QUESTION 12
# Unemployment rate of top 10 countries
# =========================

top10_countries = df["country"].value_counts().head(10).index

unemployment_top10 = (
    df[df["country"].isin(top10_countries)]
    .groupby("country")["unemployment_rate"]
    .mean()
)

print(unemployment_top10)

#Visualisation
plt.figure()
unemployment_top10.sort_values().plot(kind="barh")
plt.title("Average Unemployment Rate in Top 10 YouTube Countries")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# =========================
# QUESTION 13
# Average urban population
# =========================

print(df["urban_population"].mean())

# =========================
# QUESTION 14
# Geographic distribution of channels
# =========================

plt.figure()
sns.scatterplot(
    data=df,
    x="longitude",
    y="latitude"
)
plt.title("Geographic Distribution of YouTube Channels")
plt.show()

# =========================
# QUESTION 15
# Subscribers vs population
# =========================

print(df["subscribers"].corr(df["population"]))

#Visualisation
plt.figure()
sns.scatterplot(
    data=df,
    x="population",
    y="subscribers"
)
plt.title("Subscribers vs Country Population")
plt.xlabel("Country Population")
plt.ylabel("Subscribers")
plt.show()

# =========================
# QUESTION 16
# Population of top 10 countries
# =========================

population_top10 = (
    df[df["country"].isin(top10_countries)]
    .groupby("country")["population"]
    .mean()
)

print(population_top10)

#Visualisation
plt.figure()
population_top10.sort_values().plot(kind="barh")
plt.title("Population of Top 10 Countries with Most YouTube Channels")
plt.xlabel("Population")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# =========================
# QUESTION 17
# Subscribers gained vs unemployment rate
# =========================

print(df["subscribers_for_last_30_days"].corr(df["unemployment_rate"]))

#Visualisation
plt.figure()
sns.scatterplot(
    data=df,
    x="unemployment_rate",
    y="subscribers_for_last_30_days"
)
plt.title("Subscribers Gained vs Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Subscribers Gained (Last 30 Days)")
plt.show()

# =========================
# QUESTION 18
# Video views last 30 days by channel type
# =========================

views_by_type = (
    df.groupby("channel_type")["video_views_for_the_last_30_days"]
    .mean()
)

print(views_by_type)

#Visualisation
plt.figure()
views_by_type.plot(kind="bar")
plt.title("Avg Video Views (Last 30 Days) by Channel Type")
plt.show()

# =========================
# QUESTION 19
# Seasonal trends in uploads
# =========================

print("Seasonal trends cannot be analyzed due to lack of upload date data.")

# =========================
# QUESTION 20
# Avg subscribers gained per month
# =========================

current_year = 2024
df["channel_age_months"] = (current_year - df["created_year"]) * 12

df["avg_subs_per_month"] = (
    df["subscribers"] / df["channel_age_months"]
)

print(df["avg_subs_per_month"].describe())
