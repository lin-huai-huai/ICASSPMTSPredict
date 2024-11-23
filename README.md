# Leveraging Heterophily in Spatial-Temporal Graphs for Multivariate Time-Series Forecasting


## Structure:

* PEMS03, PEMS04, PEMS07 and PEMS08 datasets are released by and available at [STSGCN](https://github.com/Davidham3/STSGCN). \\ KnowAir dataset is released by and available at [PM2.5-GNN](https://github.com/shuowang-ai/PM2.5-GNN?tab=readme-ov-file#pm25-gnn).


* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our DPSTGNN model


## Requirements

Python 3.8.11, Pytorch 1.9.0, Numpy 1.12.2, tslearn, argparse and configparser


<table>
  <tr>
    <th rowspan="2">Windows Size</th>
    <th colspan="3">1</th>
    <th colspan="3">3</th>
    <th colspan="3">5</th>
  </tr>
  <tr>
    <th>MAE</th>
    <th>MAPE</th>
    <th>RMSE</th>
    <th>MAE</th>
    <th>MAPE</th>
    <th>RMSE</th>
    <th>MAE</th>
    <th>MAPE</th>
    <th>RMSE</th>
  </tr>
  <tr>
    <td>PEMS04</td>
    <td>19.11</td>
    <td>12.68%</td>
    <td>30.99</td>
    <td>18.71</td>
    <td>12.23%</td>
    <td>30.79</td>
    <td>19.18</td>
    <td>12.70%</td>
    <td>18.83</td>
  </tr>
  <tr>
    <td>PEMS08</td>
    <td>14.83</td>
    <td>9.98%</td>
    <td>24.34</td>
    <td>14.42</td>
    <td>9.31%</td>
    <td>23.87</td>
    <td>14.71</td>
    <td>9.88%</td>
    <td>24.13</td>
  </tr>
</table>
Here, we present the results of different sliding window size parameters on the PEMS04 and PEMS08 datasets.
