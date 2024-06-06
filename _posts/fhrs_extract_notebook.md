```python
import requests
import pandas as pd
import xml.etree.ElementTree as ET
```

The goal of this script is to gather data on restaurants and cafes to analyse their proximity to bikeshare stations. The thinking is that the closer these amenities are to bikeshare stations, the higher the likelihood that people will use bikeshare stations. These will be used as input to the statistical models. 

The script fetches data from several URLs that provide XML files containing information about food establishments in different regions. It parses this XML data to extract specific details like the type of business and their geocode information (latitude and longitude).


```python
# URL of the XML data
lec_url = 'https://ratings.food.gov.uk/api/open-data-files/FHRS878en-GB.xml'
C:/Users/patri/Documents/CDRC Bikeshare Station Data/station_optimisation/data_pre_process/tessellation/fhrs_extract_notebook.ipynb
```


```python

# Function to parse the XML and extract geocode information
def parse_geocode_and_business_type(url):
    response = requests.get(url)
    xml_data = response.content
    # Parse the XML data
    root = ET.fromstring(xml_data)
    
    data = []
    for establishment in root.find('EstablishmentCollection'):
        info = {}
        business_type_element = establishment.find('BusinessType')
        geocode_element = establishment.find('Geocode')
        
        # Extract BusinessType
        if business_type_element is not None:
            info['BusinessType'] = business_type_element.text
        
        # Extract Geocode
        if geocode_element is not None:
            longitude = geocode_element.find('Longitude')
            latitude = geocode_element.find('Latitude')
            info['Longitude'] = longitude.text if longitude is not None else None
            info['Latitude'] = latitude.text if latitude is not None else None
        
        if info:
            data.append(info)
            
    return pd.DataFrame(data)
```


```python
# Extract geocode and BusinessType information
lec_data = parse_geocode_and_business_type(lec_url)
```


```python
# Checking Leicester
lec_data = parse_geocode_and_business_type(lec_url).dropna(subset='Longitude')
lec_data
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
      <th>BusinessType</th>
      <th>Longitude</th>
      <th>Latitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Retailers - other</td>
      <td>-1.1020398</td>
      <td>52.6307377</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Restaurant/Cafe/Canteen</td>
      <td>-1.195841</td>
      <td>52.630674</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Restaurant/Cafe/Canteen</td>
      <td>-1.138173</td>
      <td>52.63595</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Takeaway/sandwich shop</td>
      <td>-1.098696</td>
      <td>52.667985</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Restaurant/Cafe/Canteen</td>
      <td>-1.133267</td>
      <td>52.632974</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3094</th>
      <td>Restaurant/Cafe/Canteen</td>
      <td>-1.132681</td>
      <td>52.622281</td>
    </tr>
    <tr>
      <th>3095</th>
      <td>Other catering premises</td>
      <td>-1.131804</td>
      <td>52.5982627</td>
    </tr>
    <tr>
      <th>3096</th>
      <td>Retailers - other</td>
      <td>-1.1540655</td>
      <td>52.6283514</td>
    </tr>
    <tr>
      <th>3099</th>
      <td>Restaurant/Cafe/Canteen</td>
      <td>-1.138505</td>
      <td>52.636357</td>
    </tr>
    <tr>
      <th>3100</th>
      <td>Takeaway/sandwich shop</td>
      <td>-1.132277</td>
      <td>52.631502</td>
    </tr>
  </tbody>
</table>
<p>2834 rows × 3 columns</p>
</div>


