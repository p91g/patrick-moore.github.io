# Predicting hourly bikeshare trips with a Temporal Graph Convolutional Network (T-GCN) using PyTorch
**_Predicting short-term bikeshare demand_**

There are three primary objectives associated with the prediction of bikeshare demand: **1)** that of bike inventory management and rebalancing, **2)** bike availability recommendation systems, and **3)** docking station location selection. The focus for the first two objectives is primarily for short-term prediction, with the number of future bikes forecast to be available within an hourly window for particular times of day. The imbalance of bikes across the network of stations is a ubiquitous challenge facing schemes; demand concentrated in particular areas and times of day (i.e. morning peaks associated with commuters) can leave the supply short in other stations. If operators are accurately able to predict how many bikes will be available for each station, effective rebalancing strategies can be devised, utilising the stock of bikes and maximising the amount of revenue generated. 

Some scholars (Faghih-Imani and Eluru, 2016; Ma et al., 2022; Yi et al., 2021) have argued that research efforts have neglected to adequately assess the spatial and temporal interaction of bikeshare demand that may arise between stations. For example, departure volume at one station may deplete bicycle stocks and induce departures at neighbouring stations, which in turn could lead to higher volumes of arrivals at other stations. Furthermore, it is possible that the arrivals and departures that occur at particular times of day are influenced by the demand for that station and nearby stations earlier in the day. If spatio-temporal dependencies in historical bikeshare trip data are not incorporated, models may be less precise in predictive accuracy. 

Convolutional neural networks (CNNs) have increasingly been used to predict bikeshare demand due to their ability to capture spatial dependencies in a flexible and adaptable manner, robust to outlier issues and specification complexities facing Spatial Lag Models and Spatial Error Models  (Zhang et al., 2022; Ai et al., 2018). There are challenges particular to data format however, as commonly the city space of bikeshare use is represented by a grid (Li et al., 2023) in which the convolutional kernel is mapped to. If the grid cells are too large, multiple docking stations will be captured and if too small, there can be a large number of zero values and computational redundancy (Lin et al., 2018). 

**_Why Graphs_**

To overcome this limitation, graph structures have been utilised instead, where each docking station is a vertice connected to the nearest docking station via an edge, resulting in an adjacency matrix of ones and zeros to represent the dependencies between stations. Graph convolutions can be performed on the data structures, known as Graph CNNs (GCNNs). Recent work has combined recurrent networks architectures with GCNNs to simultaneously capture spatial and temporal dependencies to predict short-term bikeshare demand (Ma et al., 2022; Kim et al., 2019; Li et al., 2023; Lin et al., 2018). Ma et al. (2022) found that for larger temporal granularity (i.e. 60 mins), GCNNS captured more hidden spatial information while for smaller time windows (i.e. 15 mins), LSTMs contributed more to prediction accuracy. Combining the two architectures, however, performed better than either alone as it integrated the spatial and temporal attributes into one model.

**_Objective_**

In this notebook, I have implemented a Temporal Graph Convolutional Network (T-GCN) using PyTorch. This model was originally created for forecasting traffic (see original implementation [here](https://github.com/martinwhl/T-GCN-PyTorch?tab=readme-ov-file)), making it a strong fit for forecasting bikeshare demand. I opted for this particular archictecure, as it performed better in my tests, than other similar ones: GConvLSTM, GCLSTM. 

The goal is to train the model on two months of hourly station use data and make hourly predictions for the subsequent two weeks of use. 

The trip data is for the city of Leicester and has been pre-processed as part of my PhD research. The data consists of trips recorded per docking station, per hour, making it spatio-temporal in nature. There are 45 stations, generating 34,075 trips recored in 30 minute intervals between June - Aug 2022, resulting in 178,910 observations. 


```python
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from geopandas import points_from_xy

import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj
import torch_geometric_temporal
import torch.nn.functional as F
from torch_geometric.nn import GraphConv,Linear
from torch_geometric_temporal.nn.recurrent import GConvLSTM, GCLSTM, TGCN
from torch_geometric_temporal.signal.static_graph_temporal_signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric.utils import dense_to_sparse
```


```python
print(pd.__version__)
print(torch_geometric.__version__)
print(torch_geometric_temporal.__version__)
```

    1.3.5
    2.2.0
    0.54.0
    

### Load in the trips data and define the temporal range, with trips per hour


```python
trips_30 = pd.read_csv(r"C:\Users\patri\Documents\CDRC Bikeshare Station Data\processed_trips\mins_30_data_trips.csv")
```


```python
# HOUR
lec_trips_30 = trips_30.loc[trips_30['city']=='Leicester'].copy()
lec_trips_30['timestamp'] = pd.to_datetime(lec_trips_30['timestamp'])
lec_trips_30['date'] = pd.to_datetime(lec_trips_30['date'])
lec_trips_30['id'] = lec_trips_30['id'].astype(int)
lec_trips_30['date_hour'] = pd.to_datetime(lec_trips_30['date'].dt.strftime('%Y-%m-%d') + ' ' + lec_trips_30['hour'].astype(str) + ':00')
```


```python
# filter JUN TO AUF
lec_trips_30_JUN_AUG = lec_trips_30.loc[ (lec_trips_30['month'] >= 6) & (lec_trips_30['month'] <= 8) ]
lec_trips_30_JUN_AUG[['id', 'trips','date_hour','timestamp' ,'lon', 'lat']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>trips</th>
      <th>date_hour</th>
      <th>timestamp</th>
      <th>lon</th>
      <th>lat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10154387</th>
      <td>3</td>
      <td>0.0</td>
      <td>2022-06-01 00:00:00</td>
      <td>2022-06-01 00:00:00</td>
      <td>-1.134276</td>
      <td>52.652653</td>
    </tr>
    <tr>
      <th>10154388</th>
      <td>3</td>
      <td>0.0</td>
      <td>2022-06-01 00:00:00</td>
      <td>2022-06-01 00:30:00</td>
      <td>-1.134276</td>
      <td>52.652653</td>
    </tr>
    <tr>
      <th>10154389</th>
      <td>3</td>
      <td>0.0</td>
      <td>2022-06-01 01:00:00</td>
      <td>2022-06-01 01:00:00</td>
      <td>-1.134276</td>
      <td>52.652653</td>
    </tr>
    <tr>
      <th>10154390</th>
      <td>3</td>
      <td>0.0</td>
      <td>2022-06-01 01:00:00</td>
      <td>2022-06-01 01:30:00</td>
      <td>-1.134276</td>
      <td>52.652653</td>
    </tr>
    <tr>
      <th>10154391</th>
      <td>3</td>
      <td>0.0</td>
      <td>2022-06-01 02:00:00</td>
      <td>2022-06-01 02:00:00</td>
      <td>-1.134276</td>
      <td>52.652653</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10774083</th>
      <td>964928924</td>
      <td>0.0</td>
      <td>2022-08-31 21:00:00</td>
      <td>2022-08-31 21:30:00</td>
      <td>-1.133090</td>
      <td>52.639800</td>
    </tr>
    <tr>
      <th>10774084</th>
      <td>964928924</td>
      <td>0.5</td>
      <td>2022-08-31 22:00:00</td>
      <td>2022-08-31 22:00:00</td>
      <td>-1.133090</td>
      <td>52.639800</td>
    </tr>
    <tr>
      <th>10774085</th>
      <td>964928924</td>
      <td>0.5</td>
      <td>2022-08-31 22:00:00</td>
      <td>2022-08-31 22:30:00</td>
      <td>-1.133090</td>
      <td>52.639800</td>
    </tr>
    <tr>
      <th>10774086</th>
      <td>964928924</td>
      <td>0.5</td>
      <td>2022-08-31 23:00:00</td>
      <td>2022-08-31 23:00:00</td>
      <td>-1.133090</td>
      <td>52.639800</td>
    </tr>
    <tr>
      <th>10774087</th>
      <td>964928924</td>
      <td>0.0</td>
      <td>2022-08-31 23:00:00</td>
      <td>2022-08-31 23:30:00</td>
      <td>-1.133090</td>
      <td>52.639800</td>
    </tr>
  </tbody>
</table>
<p>178910 rows × 6 columns</p>
</div>




```python
trips_all_hour = lec_trips_30_JUN_AUG.groupby(['date','timestamp'])['trips'].sum()
trips_all_hour = pd.DataFrame(trips_all_hour).reset_index()
trips_all_hour['date'] = trips_all_hour.timestamp.dt.date
trips_all_hour['date'] = pd.to_datetime(trips_all_hour['date'])
trips_all_hour['week_number'] = trips_all_hour['date'].dt.isocalendar().week
trips_all_hour['day_name'] = trips_all_hour['date'].dt.day_name()
trips_all_hour['hour'] = trips_all_hour['timestamp'].dt.hour

trips_all_hour = trips_all_hour.groupby(['date','day_name' ,'hour'])['trips'].sum().reset_index()
trips_all_hour['date_hour'] = pd.to_datetime(trips_all_hour['date'].dt.strftime('%Y-%m-%d') + ' ' + trips_all_hour['hour'].astype(str) + ':00')

```


```python
# Specify the start and end dates for the date range
start_date= '2022-06-01'
end_date = '2022-06-06'

# Filter the DataFrame using loc and a boolean mask
trips_all_hour = trips_all_hour.loc[(trips_all_hour['date'] >= start_date) & (trips_all_hour['date'] <= end_date)]
```

### Temporal variation
Taking a snap-shot of 7 days of trips per hour, it is evident that there is a morning and afternoon peak each weekday, corresponding to the commuter trips. During the weekend, the hourly trips are much more consistent. Note, the day-type is not explicitly labelled for the model, meaning it is using this information for forecasting.


```python
import matplotlib.dates as mdates
# Increase the default font size
plt.rcParams.update({'font.size': 16})
# Create a figure and axis
fig, ax = plt.subplots(figsize=(20, 8))
ax.plot(trips_all_hour['date_hour'], trips_all_hour['trips'], linewidth=3)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%A'))  # %A gives the full name of the day
ax.xaxis.set_major_formatter(mdates.DateFormatter('%a:%H:%M'))
ax.xaxis.set_major_locator(mdates.DayLocator())

# Find the start of Saturday and end of Sunday
start_of_saturdays = trips_all_hour[trips_all_hour['date_hour'].dt.dayofweek == 5]
start_of_saturdays = start_of_saturdays[start_of_saturdays['date_hour'].dt.hour == 0]['date_hour']

end_of_sundays = trips_all_hour[trips_all_hour['date_hour'].dt.dayofweek == 6]
end_of_sundays = end_of_sundays[end_of_sundays['date_hour'].dt.hour == 23]['date_hour']

# Add vertical line at the start of each Saturday
for start_of_saturday in start_of_saturdays:
    ax.axvline(start_of_saturday, color='green', linestyle='--', alpha=0.7, label='Start of Saturday')

# Add vertical line at the end of each Sunday
for end_of_sunday in end_of_sundays:
    ax.axvline(end_of_sunday, color='red', linestyle='--', alpha=0.7, label='End of Sunday')

# Only show unique labels in the legend
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
ax.legend(unique_labels.values(), unique_labels.keys())
ax.set_title('Hourly trips by day of week (1st - 6th June 2022)')

# Set x-axis major ticks to every 6 hours
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
plt.xticks(rotation=45)

# Labeling the axes
plt.xlabel('Day and Time')
plt.ylabel('Hourly Trips')

# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.show()
```


    
![png](C%3A/Users/patri/Documents/bss_graph_venv/2024-08-01-Predicting_hourly_bikeshare_trips_with_a_Temporal_Graph_Convolutional_Network_10_0.png)
    



```python
# Get unique id and coordinates for each station # 
def create_coords_df(df):
    coords_df = df[['id', 'lon', 'lat']].drop_duplicates()
    coords_df['id'] = coords_df['id'].astype(int)

    coords_df = gpd.GeoDataFrame(coords_df, geometry=points_from_xy(coords_df.lon, coords_df.lat))
    coords_df.set_crs(4326, inplace=True)
    coords_df.to_crs(27700, inplace=True)

    coords_df['lon'] = coords_df.geometry.x
    coords_df['lat'] = coords_df.geometry.y

    coords_df = coords_df.reset_index(drop=True)
    coords_df = coords_df[['id', 'lon', 'lat']]

    coords_df['node_id'] = coords_df.index.astype(int) # 'node_' + 

    return coords_df
```


```python
lec_trips_30_coords = create_coords_df(lec_trips_30_JUN_AUG)
print(f"The shape is {lec_trips_30_coords.shape} and the columns are {list(lec_trips_30_coords.columns)}")
```

    The shape is (43, 4) and the columns are ['id', 'lon', 'lat', 'node_id']
    


```python
# JOIN HOURLY TRIPS TO COORDS
lec_trips_30_JUN_AUG = lec_trips_30_JUN_AUG.drop(columns=['lon', 'lat'])
lec_trips_30_JUN_AUG = pd.merge(lec_trips_30_JUN_AUG, lec_trips_30_coords, on=['id'], how='left')
```


```python
lec_trips_30_JUN_AUG = lec_trips_30_JUN_AUG[['timestamp', 'trips', 'node_id']]
lec_trips_30_JUN_AUG = lec_trips_30_JUN_AUG.sort_values('timestamp')
```


```python
# Pivot table - rows = timestamp, columns = nod_id, value = trips
def pivot_trips(df):
    piv = df.pivot_table(index='timestamp', columns='node_id', values=['trips'])
    piv_og = piv.index.copy()
    piv_og = pd.DataFrame(piv_og)
    piv_og['timestamp'] = pd.to_datetime(piv_og['timestamp'])
    piv.fillna(0, inplace=True)
    piv.columns = piv.columns.droplevel(0)
    piv.reset_index(inplace=True, drop=True)
    piv.insert(0, 'timestamp', piv.index)

    return piv, piv_og
```


```python
df_pivot, df_pivot_og = pivot_trips(lec_trips_30_JUN_AUG)
```

We have 43 node ids, one for each docking station. Each column will be an input vector for the neural network. 
These will be used later to define the adjacency matrix.


```python
df_pivot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>node_id</th>
      <th>timestamp</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4410</th>
      <td>4410</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4411</th>
      <td>4411</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4412</th>
      <td>4412</td>
      <td>0.0</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4413</th>
      <td>4413</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4414</th>
      <td>4414</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>4415 rows × 44 columns</p>
</div>




```python
### TRAIN TEST SPLIT # 
def train_test_split(data, train_ratio, val_ratio):
    time_len = data.shape[0]
    train_size = int(time_len * train_ratio)
    val_size = int(time_len * val_ratio)
    test_size = time_len - train_size - val_size

    train_data = data.iloc[:train_size, :]
    val_data = data.iloc[train_size:train_size + val_size, :]
    test_data = data.iloc[train_size + val_size:, :]
    
    return train_data, val_data, test_data

train_ratio = 0.7
val_ratio = 0.1
```


```python
train_data, val_data, test_data = train_test_split(df_pivot, train_ratio, val_ratio)
```


```python
print(f"train shape is {train_data.shape}, val shape is {val_data.shape}, test shape is {test_data.shape} total data shape is {df_pivot.shape}")
```

    train shape is (3090, 44), val shape is (441, 44), test shape is (884, 44) total data shape is (4415, 44)
    


```python
train_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>node_id</th>
      <th>timestamp</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>33</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3085</th>
      <td>3085</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3086</th>
      <td>3086</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3087</th>
      <td>3087</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>1.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3088</th>
      <td>3088</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>...</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3089</th>
      <td>3089</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3090 rows × 44 columns</p>
</div>




```python
# SCALE THE DATA # 
def scale_df(df):
    mean = df.iloc[:, 1:].values.mean()
    std = df.iloc[:, 1:].values.std()

    # Standardize the values in the pivot table
    # Create a copy of the train_data DataFrame to store the standardized values
    scaled = df.copy()

    # Standardize the values in the DataFrame from the 1st column to the end
    scaled.iloc[:, 1:] = (df.iloc[:, 1:] - mean) / std
    
    return scaled, mean, std

def unscale_array(scaled_array, mean, std):
    # Unscale the numpy array using the mean and standard deviation
    unscaled_array = scaled_array * std + mean
    
    return unscaled_array
```


```python
train_scaled_piv, train_mean, train_std = scale_df(train_data)
val_scaled_piv, val_mean, val_std = scale_df(val_data)
test_scaled_piv, test_mean, test_std = scale_df(test_data)
```

### Creating sequences
Now we have processed the basic format of the dataframes, we need to create temporal sequences, which are fed into the network as training examples. 
The create_sequences function will handle this, generating input-output pairs for time series prediction.
- It takes two parameters: data (a 2D array of time series data) and n_past_steps (number of past time steps to use for prediction).
- It creates two lists: X for input sequences and y for target values.
- The loop iterates through the data, creating sequences of n_past_steps length for input (X) and the corresponding next value for the target (y).


```python
# CREATE SEQUENCES # 
def create_sequences(data, n_past_steps):
    X, y = [], []
    for i in range(n_past_steps, len(data)):
        X.append(data[i-n_past_steps:i, :]) # trips from n previous timesteps to current time
        y.append(data[i, 0]) #trips at current timestep 
        
    return np.array(X), np.array(y)
```

The sequence_lists function applies the create_sequences function to multiple time series (stations) and reshapes the data.
Sequences are generated for each node in the graph (i.e. docking station) within a loop.
The sequence lists, X_list,  have an initial shape of (n_stations, n_samples, n_time_steps, n_features). These need to be reshaped
- np.transpose(..., axes=[3,1,0,2]) rearranges the dimensions:
    - 3 (n_features) becomes the first dimension
    - 1 (n_samples) becomes the second dimension
    - 0 (n_stations) becomes the third dimension
    - 2 (n_time_steps) becomes the fourth dimension

- This results in a shape of (n_features, n_samples, n_stations, n_time_steps).
- X_list_reshaped[0] selects only the first feature, resulting in a 3D array with shape (n_samples, n_stations, n_time_steps).
- For y_list_reshaped, we simply transpose the 2D array, swapping the station and sample dimensions.
- The final shapes of the data are:
    X: (n_samples, n_stations, n_time_steps)
    y: (n_samples, n_stations)




```python
def sequence_lists(coords_df, df_piv, n_time_steps):

    X_list, y_list = [], []

    n_past_steps = n_time_steps  # Number of past timesteps you want to use for prediction

    for station in coords_df.node_id:
        if station in df_piv.columns:
            station_trips = df_piv[[station]]
            
            features = station_trips.values

            city_X, city_y = create_sequences(features, n_past_steps)

            X_list.append(city_X)
            y_list.append(city_y)

    #RESHAPE 
    X_list_reshaped = np.transpose(np.array(X_list),axes=[3,1,0,2]) .astype(np.float16)
    X_list_reshaped = X_list_reshaped[0]
    y_list_reshaped = np.array(y_list).T .astype(np.float16)   
    
    return X_list_reshaped, y_list_reshaped  
```

We use 2 timesteps, meaning we look back an hour.


```python
X_train, y_train = sequence_lists(lec_trips_30_coords, train_scaled_piv, 2) # 
X_val, y_val = sequence_lists(lec_trips_30_coords, val_scaled_piv, 2) # 
X_test, y_test = sequence_lists(lec_trips_30_coords, test_scaled_piv, 2) # 
```


```python
print(f"input training data shape is {X_train.shape}, target training data shape is {y_train.shape},\n"
      f"input validation shape is {X_val.shape}, target validation shape is {y_val.shape},\n"
      f"test input shape is {X_test.shape}, target test shape is {y_test.shape}")
```

    input training data shape is (3088, 43, 2), target training data shape is (3088, 43),
    input validation shape is (439, 43, 2), target validation shape is (439, 43),
    test input shape is (882, 43, 2), target test shape is (882, 43)
    


```python
X_train
```




    array([[[-0.4355, -0.4355],
            [-0.4355, -0.4355],
            [-0.4355, -0.4355],
            ...,
            [-0.4355, -0.4355],
            [ 2.027 ,  0.796 ],
            [-0.4355, -0.4355]],
    
           [[-0.4355, -0.4355],
            [-0.4355, -0.4355],
            [-0.4355, -0.4355],
            ...,
            [-0.4355,  0.796 ],
            [ 0.796 ,  3.258 ],
            [-0.4355, -0.4355]],
    
           [[-0.4355, -0.4355],
            [-0.4355, -0.4355],
            [-0.4355, -0.4355],
            ...,
            [ 0.796 , -0.4355],
            [ 3.258 , -0.4355],
            [-0.4355, -0.4355]],
    
           ...,
    
           [[-0.4355, -0.4355],
            [-0.4355, -0.4355],
            [ 0.796 , -0.4355],
            ...,
            [-0.4355,  0.796 ],
            [ 3.258 , -0.4355],
            [-0.4355, -0.4355]],
    
           [[-0.4355, -0.4355],
            [-0.4355, -0.4355],
            [-0.4355,  0.796 ],
            ...,
            [ 0.796 ,  0.796 ],
            [-0.4355,  3.258 ],
            [-0.4355,  2.027 ]],
    
           [[-0.4355,  0.796 ],
            [-0.4355, -0.4355],
            [ 0.796 ,  0.796 ],
            ...,
            [ 0.796 ,  0.796 ],
            [ 3.258 ,  2.027 ],
            [ 2.027 , -0.4355]]], dtype=float16)



### Create an adjacency matrix


```python
## CREATE ADJECENCY MATRIX # 

from scipy.spatial.distance import cdist
# Step 1: Extract the 'Lon' and 'Lat' values from the 'nodes' DataFrame
coords = lec_trips_30_coords[['lon', 'lat']].values

# Step 2: Calculate the pairwise distances between nodes using Euclidean distance
distances = cdist(coords, coords, metric='euclidean')
#Make it a tensor
distances_tensor = torch.tensor(distances)
edge_indices, values = dense_to_sparse(distances_tensor)
values = values.numpy()
edges = edge_indices.to(torch.long)  # Convert edge_indices to torch.long
edge_weights = values    

edges.shape
edge_weights.shape
```




    (1806,)




```python
distances
```




    array([[   0.        , 2204.35339159,  985.52043285, ..., 2806.59493104,
            4047.47308246, 1432.05725317],
           [2204.35339159,    0.        , 1218.83392367, ..., 1552.56752674,
            2089.96485707, 1254.12980523],
           [ 985.52043285, 1218.83392367,    0.        , ..., 2057.91211253,
            3132.46093605,  798.43199904],
           ...,
           [2806.59493104, 1552.56752674, 2057.91211253, ...,    0.        ,
            1502.78642813, 1375.1215128 ],
           [4047.47308246, 2089.96485707, 3132.46093605, ..., 1502.78642813,
               0.        , 2674.22789381],
           [1432.05725317, 1254.12980523,  798.43199904, ..., 1375.1215128 ,
            2674.22789381,    0.        ]])




```python
# GENERATE TEMPORAL GRAPH DATASET # 

train_bike_dataset = StaticGraphTemporalSignal(edge_index = edges, 
                                        edge_weight = edge_weights, 
                                        features=X_train, targets=y_train)

val_bike_dataset = StaticGraphTemporalSignal(edge_index = edges, 
                                        edge_weight = edge_weights, 
                                        features=X_val, targets=y_val)

test_bike_dataset = StaticGraphTemporalSignal(edge_index = edges, 
                                        edge_weight = edge_weights, 
                                        features=X_test, targets=y_test)
```

### Define the model


```python
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        
        #self.recurrent = GConvLSTM(node_features, 64, 1)
        # self.recurrent = GCLSTM(node_features, 64, 1)
        self.recurrent = TGCN(node_features, 64, 1)

        # Two fully-connected layers
        self.fc1 = Linear(64, 32)
        self.fc2 = Linear(32, 16)
        
        # self.linear = Linear(8, 1)
        self.linear = Linear(16, 1)
        # self.linear = Linear(32, 1)

    def forward(self, x, edge_index, edge_weight, h): # c
        h_0  = self.recurrent(x, edge_index, edge_weight, h) # c_0 = (c)
        
        h = F.relu(h_0)

        h = F.dropout(h, p=0.2, training=self.training)  # Apply dropout after the first activation

        h = self.fc1(h)
        h = F.relu(h)
        
        h = F.dropout(h, p=0.2, training=self.training)  # Apply dropout after the second activation

        h = self.fc2(h)
        h = F.relu(h)

        h = F.dropout(h, p=0.2, training=self.training)  # Apply dropout after the second activation
        
        h = self.linear(h)
        
        return h, h_0 # c_0
```

### Inititalise and train the model


```python
import os
print("Current working directory:", os.getcwd())
checkpoint_dir = "\\model_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

n_past_steps = 2
weight_decay = 5e-4
model = RecurrentGCN(node_features=n_past_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=weight_decay)
model.train()
```


```python
train_losses = []
val_losses = []

# TRAINING
#tqdm here is to enable a progress bar for your loop
for epoch in tqdm(range(40)):
    loss = 0
    y_pred_train = []
    y_pred_test = []
    h = None # c = None
    for time, snapshot in enumerate(train_bike_dataset):
        y_hat, h = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h) # c = (c) # performs a forward pass through the model using the current snapshot's features (snapshot.x), 
                                                                                            #edge indices (snapshot.edge_index), edge attributes (snapshot.edge_attr), 
                                                                                            # and the previous hidden state (h). The model returns the predicted values (y_hat) and the updated hidden state (h).
        
        loss = loss + torch.mean((y_hat-snapshot.y)**2) # computes the mean squared error (MSE) loss between the predicted values (y_hat) and the true values (snapshot.y) for the current snapshot. 
                                                         # The loss is accumulated by adding it to the loss variable.
        
        y_pred_train.append(y_hat.detach().numpy()) # appends the predicted values (y_hat) to the y_pred_train list 
                                                    # with detach only want to do forward computations through the network. We stop tracking computations for the gradient
    
    loss = loss / (time+1) # computes the average loss over the snapshots seen so far by dividing the accumulated loss by the number of snapshots processed (time+1)
    loss.backward() # performs backpropagation to compute the gradients of the loss with respect to the model's parameters.
    optimizer.step() # updates the model's parameters using the computed gradients and the specified optimizer.
    optimizer.zero_grad() # sets the gradients of all model parameters to zero, preparing for the next iteration.

    train_losses.append(loss.item()) # ppends the current loss value to the train_losses list, which stores the training losses for each epoch.

    # VALIDATION
    model.eval()
    val_loss = 0
    y_pred_val = []
    h = None
    
    with torch.no_grad():
        for time, snapshot in enumerate(val_bike_dataset):
            y_hat, h = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h)
            val_loss = val_loss + torch.mean((y_hat - snapshot.y)**2)
            y_pred_val.append(y_hat.detach().numpy())
    
    val_loss = val_loss / (time + 1)
    val_losses.append(val_loss.item())

    print(f"\nEpoch: {epoch+1}")
    print(f"Training loss: {loss.item():.4f}")
    print(f"Validation loss: {val_loss.item():.4f}")

    # Save the model checkpoint if it has the lowest validation loss
    if val_losses[-1] == min(val_losses):
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pth"))
```


```python
# Load the best model checkpoint
best_epoch = val_losses.index(min(val_losses)) + 1
best_model_path = os.path.join(checkpoint_dir, f"best_model_epoch_{best_epoch}.pth")
model.load_state_dict(torch.load(best_model_path))
```


```python
# TESTING
model.eval()
test_loss = 0
y_pred_test = []
h = None

with torch.no_grad():
    for time, snapshot in tqdm(enumerate(test_bike_dataset)):
        y_hat, h = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h)
        test_loss = test_loss + torch.mean((y_hat - snapshot.y)**2)
        y_pred_test.append(y_hat.detach().numpy())

test_loss = test_loss / (time + 1)

print(f"\nTest loss: {test_loss.item():.4f}")
```


```python
#############################################
## UNSCALE #################################
############################################

y_pred_train = np.array(y_pred_train)[:, :, 0] # convert y_pred_train list into array of shape (3530, 43, 1) and select all elements of the third dimension
y_pred_test = np.array(y_pred_test)[:, :, 0] # convert y_pred_train list into array of shape (881, 43, 1) and select all elements of the third dimension

y_pred_train_unscaled = unscale_array(y_pred_train, train_mean, train_std)
y_pred_test_unscaled_piv = unscale_array(y_pred_test, test_mean, test_std)
```

### Plot the results


```python
########################################
### PLOTTING ########################
##################################
import matplotlib.dates as mdates

# PLOTTING TRAINING AND TEST LOSS
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#plt.ylim(0,0.1)
plt.legend()
plt.show()
```


```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

plt.rcParams.update({'font.size': 16})

def plot_full_and_zoomed(df_pivot_og, train_data, test_data, y_pred_train_unscaled, y_pred_test_unscaled_piv, 
                         zoom_start_hours=24, zoom_duration_hours=48):
    training_len = train_data.shape[0]
    test_len = test_data.shape[0]
    test_start = training_len

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

    # Function to plot data on a given axis
    def plot_data(ax, linewidth=5):
        # Ground truth
        # ax.plot(df_pivot_og['timestamp'][2:training_len+2], train_data.iloc[:, 1:].values.sum(axis=1), label='Training Ground Truth', linewidth=linewidth)
        ax.plot(df_pivot_og['timestamp'][test_start:test_start+test_len], test_data.iloc[:, 1:].values.sum(axis=1), 'orange', label='Test Ground Truth', linewidth=linewidth)

        # Predictions
        # ax.plot(df_pivot_og['timestamp'][2:training_len], y_pred_train_unscaled.sum(axis=1), '#339933', label='Training Predictions', linewidth=linewidth)
        ax.plot(df_pivot_og['timestamp'][test_start:test_start+test_len-2], y_pred_test_unscaled_piv.sum(axis=1), '#339933', label='Test Predictions', linewidth=linewidth)

        train_end_timestamp = df_pivot_og['timestamp'][y_pred_train_unscaled.shape[0] + 2]
        # ax.axvline(x=train_end_timestamp, color='red', ls='--', lw=3, label='Train-Test Split')

        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))

    # Plot full range
    plot_data(ax1, linewidth=3)
    ax1.set_title('Hourly trips: test ground truth vs test predictions ')
    # Labeling the axes
    ax1.set_xlabel('Date / Time')
    ax1.set_ylabel('Hourly Trips')

    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=24))
    ax1.tick_params(axis='x', rotation=45)

    # Plot zoomed range with increased linewidth
    plot_data(ax2, linewidth=3)
    zoom_start = df_pivot_og['timestamp'][test_start + zoom_start_hours]
    zoom_end = zoom_start + timedelta(hours=zoom_duration_hours)
    ax2.set_xlim(zoom_start, zoom_end)
    ax2.set_title(f'4th - 6th June - hourly trip predictions ({zoom_duration_hours} Hours)')

    # Custom legend for the zoomed plot
    custom_lines = [
        plt.Line2D([0], [0], color='orange', lw=3, label='Test Ground Truth'),
        plt.Line2D([0], [0], color='#339933', lw=3, label='Test Predictions'),
    ]
    ax2.legend(handles=custom_lines, loc='best')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b: %H:%M'))

    ax2.set_xlabel('Date / Time')
    ax2.set_ylabel('Hourly Trips')

    # Set x-axis major ticks to every 6 hours
    ax2.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
```


```python
#plot
plot_full_and_zoomed(df_pivot_og, train_data, test_data, y_pred_train_unscaled, y_pred_test_unscaled_piv, 
                     zoom_start_hours=24, zoom_duration_hours=48)
```


    
![png](C%3A/Users/patri/Documents/bss_graph_venv/2024-08-01-Predicting_hourly_bikeshare_trips_with_a_Temporal_Graph_Convolutional_Network_48_0.png)
    



```python
# one station
station_id = 2

plt.figure(figsize=(20, 8))
training_len = train_data.iloc[:, 1:].values.sum(axis=1).shape[0]
test_len = test_data.iloc[:, 1:].values.sum(axis=1).shape[0]
#ground truth
# plt.plot(df_pivot_og['timestamp'][2:training_len+2], train_data.iloc[:, 1:].values[:,station_id], label='Training Ground Truth')
plt.plot(df_pivot_og['timestamp'][training_len:training_len+test_len], test_data.iloc[:, 1:].values[:,station_id], label='Test Ground Truth')
#predictions
# plt.plot(df_pivot_og['timestamp'][2:training_len], y_pred_train_unscaled[:,station_id]) # , label='Training Predictions'
plt.plot(df_pivot_og['timestamp'][training_len:training_len+test_len-2], y_pred_test_unscaled_piv[:,station_id], label='Test Predictions', linewidth=4)


train_end_timestamp = df_pivot_og['timestamp'][y_pred_train.shape[0] + 2]
# plt.axvline(x=train_end_timestamp, color='red', ls='--', lw=3)
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %B'))
plt.title(f'Hourly trip predictions for station # {station_id}')
# Rotate and align the tick labels so they look better
plt.gcf().autofmt_xdate()

# Labeling the axes
plt.xlabel('Date')
plt.ylabel('Hourky Trips')
plt.show()
```


    
![png](C%3A/Users/patri/Documents/bss_graph_venv/2024-08-01-Predicting_hourly_bikeshare_trips_with_a_Temporal_Graph_Convolutional_Network_49_0.png)
    


### Check the models error


```python
import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# For training data
train_y_true = np.array([snapshot.y for snapshot in train_bike_dataset])
train_rmse = calculate_rmse(train_y_true, y_pred_train_unscaled)

# For test data
test_y_true = np.array([snapshot.y for snapshot in test_bike_dataset])
test_rmse = calculate_rmse(test_y_true, y_pred_test_unscaled_piv)

print(f"Training RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
```

    Training RMSE: 1.010406732559204
    Test RMSE: 1.0090234279632568
    

## Conclusions
The model predictions, on average, deviate from the actual number of trips by approximately one trip per hour, which is pretty good! This suggests we haven't over-fit the model. Other architectures, such as GConvLSTM and GCLSTM, when tested, tended to over-fit and led to quite poor predictions. When plotted, the accuracy of the predictions are quite impressive, indicating that it has learned the temporal patterns of demand each day very well. If I were to go deeper into this analysis, it would have been nice to use some additional temporal features like weather, or spatial features based on the surrounding built environment of each station. That way, I could have potentially improved the node-level predictions, which could be stronger. Some additional benchmarking against a scheme wide prediction, simply using an LSTM or similar recurrent network would be interesting. Finally, I didn't utilise any hyper-parameter optimisation, besides manually testing, which I expect could lead to even greater accuracy!
