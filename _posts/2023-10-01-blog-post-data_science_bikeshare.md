---
title: 'Data Science for Bikeshare'
date: 2023-10-01
permalink: /posts/2023/10/blog-post-data-science-bikeshare/
tags:
  - bikeshare
---
**What is covered?**
- On this page I introduce how data science can be used to tackle challenges facing bikeshare schemes 

Bikeshare research
===
Scholarly research into the use of bikeshare schemes (BSS) has grown significantly over the past decade (see figure 1), with great interest in their contribution towards decarbonising transport (see Shaheen et al., 2018) off-setting societal health costs (see Otero et al., 2018) and reducing the number of vehicles needed per person. BSS are a pooled system of bicycles which users hire on a short-term basis for a fee per minute (Urban Transport Group, 2021). Increasingly, bikeshare has become a popular form of urban transportation, growing from 16 schemes in 2004 to over 2,000 worldwide in 2021 (DeMaio, 2021). 

Challenges
===
There are difficulties, however, concerning the _planning, implementation and operational efficiency_ of new and existing schemes. Specifically, is: 
1. the pre-deployment concern of **selecting docking station locations** (Liu et al., 2015);
2. the post-deployment requirement of **redistributing bikes** around the system (Zhang et al., 2022).

These problems remain a complex and ubiquitous challenge for bikeshare operators, impacting the capacity to operate effectively and generate revenue. 

Ultimately, if left unaddressed, they may lead to underuse of a scheme and, in some cases, even result in scheme closure (DeMaio, 2021). Researchers have attempted to develop and implement methods for addressing these issues by utilising: 
- statistical models;
- machine learning algorithms;
- optimisation techniques. 

Data Science Solutions
---
There are a number of data science methods that can be used address these issues.

**Extracting trip data**
- Web-scraping bikeshare operator APIs

**Feature extraction**
- Exporting and pre-processing built environment data
- Street morphology feature extraction

**Selecting docking station locations**:
- Regression analysis of the factors influencing demand
- Regression to predict long-term demand
- Constrained optimisation of station locations

**Redistributing bikes between stations**:
- Forecasting short-term demand
- Optimising bike availability
