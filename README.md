# 2025-HUFS-CollegeofEnglish - HUFSTAG

A Two-Stage Study on Hashtag Structures and Audience Reactions on X (formerly Twitter)

\n
[Overview]

This repository contains the research code, analysis scripts, and documentation for a two-stage study conducted by Team #HUFSTAG,
a participant group in the 2025 HUFS (Hankuk University of Foreign Studies) College of English Academic Festival.

The project investigates how the structural features of English hashtags influence user perception and emotional response across social media platforms.
The study is divided into two main stages:

  Stage 1 — Quantitative Sentiment Analysis using real SNS data (mainly from X, formerly Twitter)

  Stage 2 — Qualitative Perception Survey, validating results through user evaluation

  Stage 1: Data-Based Sentiment Analysis

[Objective]

To analyze how hashtag structures (e.g., compounding, abbreviation, phrasing) influence user sentiment expressed in comments.


[Data Collection]

Main platform: X (formerly Twitter)

Tools: Octoparse for large-scale automated scraping

Dataset contents: Post text, upload date, comments, and reaction indicators(likes or retweets, comments)


[Hashtag Categorization]

Hashtags are manually classified into three functional purposes:

  Unity — expressing collective identity or belonging

  Viral — designed for promotion, visibility, and shareability

  Lifestyle — reflecting personal taste, daily life, and aesthetic values


[Method]

Conduct sentiment analysis on comments (not post bodies) using VADER, a sentiment analysis tool optimized for social media text.

Compute per-comment sentiment scores (neg, neu, pos, compound) and derive means and ratios per hashtag.

Analyze how each hashtag’s structure correlates with emotional responses among users.


[Tools and Environment]

  Language: Python

  Libraries: pandas, nltk, langdetect, pandarallel, openpyxl

  Environment: Google Colab

  Visualization: Excel and Tableau (bar charts, donut charts, heatmaps)



Stage 2: User Perception Survey (Offline)

[Objective]

To verify whether the optimized structural patterns identified in Stage 1 are also valid from the perspective of user perception.


[Design]

Stimuli: Pairs of hashtags with the same purpose but different structural characteristics

  Unity: compound forms with different spellings

  Viral: abbreviations vs. compound forms

  Lifestyle: compound vs. sentence-like expressions

Participants: The participants will consist of students enrolled in Interacting in Spoken English (2), Research Writing (2), and Programming for English Data Processing (2), and the survey will be conducted face-to-face during class sessions.

Format: Google Form online questionnaire

[Questionnaire Structure]

Purpose Reflection:
Rate how well each hashtag reflects its intended purpose on a 6-point Likert scale (1 = strongly disagree, 2 = disagree, 3 = slightly disagree, 4 = slightly agree, 5 = agree,  6 = strongly agree).

Perceived Effectiveness:
A Yes/No question asking whether the hashtag effectively fulfills its communicative goal.

Expression Preference:
Choose between hashtag expression and ordinary sentence of equivalent meaning, based on which feels more natural or engaging.

[Data Analysis]

Basic descriptive statistics (mean, standard deviation, response ratio)

t-tests: to compare structural pairs within the same purpose

One-way ANOVA: to test mean differences across purposes

Visualization:

Bar charts (mean comparison)

Pie charts (positive/negative/neutral sentiment ratio)

Heatmaps (response distribution)
