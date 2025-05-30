#### A股商誉市场概况

接口: stock_sy_profile_em

目标地址:  https://data.eastmoney.com/sy/scgk.html

描述: 东方财富网-数据中心-特色数据-商誉-A股商誉市场概况

限量: 单次所有历史数据

输入参数

| 名称  | 类型  | 描述  |
|-----|-----|-----|
| -   | -   | -   |

输出参数

| 名称         | 类型      | 描述      |
|------------|---------|---------|
| 报告期        | object  | -       |
| 商誉         | float64 | 注意单位: 元 |
| 商誉减值       | float64 | 注意单位: 元 |
| 净资产        | float64 | 注意单位: 元 |
| 商誉占净资产比例   | float64 | -       |
| 商誉减值占净资产比例 | float64 | -       |
| 净利润规模      | float64 | 注意单位: 元 |
| 商誉减值占净利润比例 | float64 | -       |

接口示例

```python
import akshare as ak

stock_sy_profile_em_df = ak.stock_sy_profile_em()
print(stock_sy_profile_em_df)
```

数据示例

```
           报告期            商誉  ...         净利润规模  商誉减值占净利润比例
0   2010-12-31  9.305439e+10  ...  8.646720e+11   -0.008547
1   2011-12-31  1.334065e+11  ...  1.030624e+12   -0.001177
2   2012-12-31  1.639409e+11  ...  1.224410e+12   -0.000985
3   2013-12-31  2.051656e+11  ...  1.441550e+12   -0.001916
4   2014-12-31  3.246068e+11  ...  1.604929e+12   -0.001784
5   2015-12-31  5.985505e+11  ...  1.757708e+12   -0.005302
6   2016-12-31  9.667876e+11  ...  1.873202e+12   -0.008425
7   2017-12-31  1.195754e+12  ...  2.244254e+12   -0.011668
8   2018-12-31  1.219315e+12  ...  2.327512e+12   -0.049232
9   2019-12-31  1.190470e+12  ...  2.506131e+12   -0.045300
10  2020-12-31  1.157767e+12  ...  2.606449e+12   -0.028383
11  2021-12-31  1.156240e+12  ...  3.307810e+12   -0.015126
12  2022-12-31  1.198481e+12  ...  3.773819e+12   -0.015187
13  2023-12-31  1.226307e+12  ...  3.745003e+12         NaN
14  2024-06-30  1.246240e+12  ...  2.049446e+12         NaN
15  2024-09-30  1.254842e+12  ...  3.089467e+12         NaN
[16 rows x 8 columns]
```
