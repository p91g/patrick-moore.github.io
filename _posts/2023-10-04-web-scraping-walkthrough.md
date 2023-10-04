---
title: "Web-scraping London bikeshare bike availability"
date: 2023-10-03
permalink: /posts/2023/10/web-scraping-walkthrough2
---


This code is an example of how to use the R programming language to access and process data from a bikeshare operator and obtain docking station bike availability data for that moment in time. This is an example for one city, but in principle the same could be done for many cities and then stored in a combined database.

The code will achieve the following objectives:

- It will send a request to a web service that provides information about the status of the cycle hire stations in London, such as the number of bikes and docks available, the location, and the name of each station.
- It will parse the response from the web service, which is in XML format, into a tree structure that can be easily manipulated and queried.
- It will display the content of the XML tree on the console, so that the user can inspect the structure and elements of the data.
- It will get the current system time and store it in a variable.
- It will convert the XML tree into a data frame, which is a tabular format that can be used for further analysis. It will also add a new column with the current time to the data frame, so that the user can keep track of when the data was obtained.


**Get docking station data for current time**
First we need to load the necessary libraries

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, message=FALSE)
```

```r?example=spatial
library(tidyverse)
library(xml2)
library(XML)
library(knitr)
library(RSQL) #Generate and Process 'SQL' Queries in R
library(RSQLite) #Can create an in-memory SQL database
library(odbc) #Contains drivers to connect to a database
library(DBI) #Contains functions for interacting with the database
```

**Get the data**

To find the url needed to obtain the data via an API, a bit of exploration is needed on the bikeshare operator site. 

Below describes the process for obtaining data from the London scheme.

The first line uses the httr::GET() function to send a GET request to the URL “https://tfl.gov.uk/tfl/syndication/feeds/cycle-hire/livecyclehireupdates.xml”, which is an XML file that contains information about the status of the cycle hire stations in London. The result of the request is stored in the object status_api_call.

```r
status_api_call <- httr::GET("https://tfl.gov.uk/tfl/syndication/feeds/cycle-hire/livecyclehireupdates.xml")
```

The second line uses the xmlInternalTreeParse() function to parse the XML content of status_api_call into an internal tree representation, which is easier to manipulate and query. The result is stored in the object api_xml2.

```r
api_xml2 = xmlInternalTreeParse(status_api_call)
```

The printed output is in the wrong format, so we need convert into a df.
```r 
#cat(saveXML(api_xml2, nchars = 10))

```

The fourth line uses the Sys.time() function to get the current system time and store it in the object current_time. This is needed to ensure that the date and time is correct in the final df, as sometime bikeshare operators have errors with the datetime provided. 

```r
current_time = Sys.time()
```

The fifth line uses the xmlToDataFrame() function to convert the XML tree api_xml2 into a data frame, which is a tabular format that can be used for further analysis. The data frame is then piped (|>) to the mutate() function, which adds a new column called datetime with the value of current_time. This can help to keep track of when the data was obtained. The result is stored in the object xmldf.

```r 
xmldf = xmlToDataFrame(api_xml2) |>
  dplyr::mutate(datetime = current_time)

# print the data frame as a table
knitr::kable(xmldf[1:5, ], caption = "Data frame from XML file")

```


We then need to save this data to a database. This code is an example of how to use R to interact with a SQLite database.

**CREATE A DATABASE**

The first line uses the dbConnect() function to establish a connection to a SQLite database file located at “C:/Users/user/Documents/name/db_name”. The result of the connection is stored in the object con

```r
con <- dbConnect(RSQLite::SQLite(), dbname="C:/Users/patri/Documents/R projs/city_comparison/cities_database.db")
```

**WRITE TO DATABASE**

The second line uses the dbWriteTable() function to write the data frame xmldf to a table named ‘cities_database_table’ in the database. The append=TRUE argument indicates that the data will be added to the existing table, rather than replacing it

```r
# RSQLite::dbWriteTable(con, name='cities_database_table', xmldf,append=TRUE)
```

Check what tables exist in your database

The third line uses the dbListTables() function to list the names of all the tables in the database. This can help to check if the table was created successfully.
```r
# RSQLite::dbListTables(con)
```

create dataframe from DB

The fourth line uses the dbGetQuery() function to execute a SQL query that selects all the rows and columns from the table ‘cities_database_table’ and returns them as a data frame. The result is stored in the object cities_db_df.
```r
# cities_db_df <- dbGetQuery(con, "SELECT * FROM cities_database_table")
```

**Setup a script schedule**

Separate to this script, we need to ensure that we collect this data every 10 minutes by using a timer feature to re-run the script. 

```r
# library(taskscheduleR)
# taskscheduler_create(taskname = "my_script_task", 
                     #rscript = "C:/Users/username/Documents/my_script.R", 
                     #schedule = "MINUTE", 
                     #starttime = format(Sys.time() + 50, "%H:%M"), 
                     #modifier = 10)
```
This will create a task named my_script_task that will run your script every ten minutes, starting from 50 seconds after the current time. You can modify the arguments of the function to suit your needs, such as changing the task name, the script path, the schedule frequency, or the start time.
