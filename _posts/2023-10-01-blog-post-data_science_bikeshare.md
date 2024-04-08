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

This comprised of three methodological stages using web-scraped bikeshare trip data and open-source built environment data.

![image](https://github.com/p91g/patrick-moore.github.io/assets/93223269/dd61f88d-1732-4c36-a20c-56b088c59707)

**Stage 1**:
- Obtain bikeshare trip data from docking station APIs
- Trip data spatially joined to built environment data, optained from OpenStreetMap

![image](https://github.com/p91g/patrick-moore.github.io/assets/93223269/0ebd4b83-a38c-4601-8e3f-2891071d0ff5)


**Stage 2**:
Generate statistical models to **understand relationships** and identify predictors. Supported by machine learning and statistical models to make **predictions** for potential locations.

![gamm_diagram](https://github.com/p91g/patrick-moore.github.io/assets/93223269/93b3a051-5d4d-4089-b52c-db6c87139ed5)


**Stage 3**:
Use optimisation algorithms to **search for solutions** for suitable station locations. 
![opti_model](https://github.com/p91g/patrick-moore.github.io/assets/93223269/5e876da5-8c15-49e4-8b8c-3010740e87a4)


You can find out how I have applied data science aproaches to these specific challenges in my individual posts. 
