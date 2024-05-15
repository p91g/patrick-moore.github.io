---
title: 'Data Science for Bikeshare'
date: 2023-10-01
permalink: /posts/2023/10/blog-post-data-science-bikeshare/
tags:
  - bikeshare
---
**What is covered?**
- On this page I introduce how data science can be used to tackle challenges facing bikeshare schemes 

Bikeshare growth
===
- Bikeshare has grown rapidly in the last 30 years, going from 350 operating worldwide in 2010 to over 2,000 worldwide in 2021 (see figure 1)
- There are many benefits promoted from bikeshare, including improved physical health, cost-saving, efficient travel, car use reduction, less pollution and public transport integration

<br/><img src='https://p91g.github.io/patrick-moore.github.io/images/2023-06-30 11_46_32-Microsoft Word - Meddin map mid-2022 report_FINAL.docx (2).png'>


Challenges
===
Barriers to use persist, however, including  **docking station locations**
- The largest barrier to using bikeshare cited among UK respondents (CoMo, 2022), at 56%, is availability/bike locations. 

<br/><img src='https://p91g.github.io/patrick-moore.github.io/images/find_bss.png' width='600' height='auto'>

- Under-use threatens viability, with more than 28 UK schemes closing since 2010 🚫
- These problems remain a complex and ubiquitous challenge for bikeshare operators, impacting the capacity to operate effectively and generate revenue 💷💷💷. 

What can be done?
===
A natural question that arises is: what are the factors that determine bikeshare use and how can they inform bikeshare operations?

My research focussed on the influence of the built environment on use in order to model and optimise the location of docking stations in four UK cities. 

This comprised of **four** methodological stages using web-scraped bikeshare trip data and open-source built environment data.

![image](https://github.com/p91g/patrick-moore.github.io/assets/93223269/0ebd4b83-a38c-4601-8e3f-2891071d0ff5)

**Stage 1**:
- Obtain bikeshare trip data from docking station APIs from multiple UK cities
- Extract open-source built environment data and define buffer zones

**Stage 2**:
- Pre-process built environment variables and spatially join with buffer zones

**Stage 3**:
- Generate statistical models to **understand relationships** and identify predictors.
- Supported by machine learning and statistical models to make **predictions** for potential locations.
- Variation within and between cities is captured in one model using random effects. 

**Stage 4**:
- Use an evolutionary optimisation algorithm to **search for solutions** for suitable station locations.
- Objectives of efficiency and accessibility are compared, incorporating urbam morphology information. 

**Methodology diagram**
![image](https://github.com/p91g/patrick-moore.github.io/assets/93223269/a1cc1183-8aee-4fda-ae0f-1a6ca3d5b3d9)
_Diagram of the data pre-processing, statistical modelling and analysis stages. Built environment data is processed and joined to the docking station buffers. Built environment variables are used to model trips, with random effects and splines fitted. An optimisation algorithm is used to determine locations that maximise coverage or minimise distance while maximising trips._


You can find out how I have applied data science aproaches to these specific challenges in my individual posts. 
