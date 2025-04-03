#### A 股等权重与中位数市盈率

接口: stock_a_ttm_lyr

目标地址: https://www.legulegu.com/stockdata/a-ttm-lyr

描述: 乐咕乐股-A 股等权重市盈率与中位数市盈率

限量: 单次返回所有数据

输入参数

| 名称  | 类型  | 描述  |
|-----|-----|-----|
| -   | -   | -   |

输出参数

| 名称                                  | 类型      | 描述                               |
|-------------------------------------|---------|----------------------------------|
| date                                | object  | 日期                               |
| middlePETTM                         | float64 | 全A股滚动市盈率(TTM)中位数                 |
| averagePETTM                        | float64 | 全A股滚动市盈率(TTM)等权平均                |
| middlePELYR                         | float64 | 全A股静态市盈率(LYR)中位数                 |
| averagePELYR                        | float64 | 全A股静态市盈率(LYR)等权平均                |
| quantileInAllHistoryMiddlePeTtm     | float64 | 当前"TTM(滚动市盈率)中位数"在历史数据上的分位数      |
| quantileInRecent10YearsMiddlePeTtm  | float64 | 当前"TTM(滚动市盈率)中位数"在最近10年数据上的分位数   |
| quantileInAllHistoryAveragePeTtm    | float64 | 当前"TTM(滚动市盈率)等权平均"在历史数据上的分位数     |
| quantileInRecent10YearsAveragePeTtm | float64 | 当前"TTM(滚动市盈率)等权平均"在在最近10年数据上的分位数 |
| quantileInAllHistoryMiddlePeLyr     | float64 | 当前"LYR(静态市盈率)中位数"在历史数据上的分位数      |
| quantileInRecent10YearsMiddlePeLyr  | float64 | 当前"LYR(静态市盈率)中位数"在最近10年数据上的分位数   |
| quantileInAllHistoryAveragePeLyr    | float64 | 当前"LYR(静态市盈率)等权平均"在历史数据上的分位数     |
| quantileInRecent10YearsAveragePeLyr | float64 | 当前"LYR(静态市盈率)等权平均"在最近10年数据上的分位数  |
| close                               | float64 | 沪深300指数                          |

接口示例

```python
import akshare as ak

stock_a_ttm_lyr_df = ak.stock_a_ttm_lyr()
print(stock_a_ttm_lyr_df)
```

数据示例

```
            date  middlePETTM  ...  quantileInRecent10YearsAveragePeLyr    close
0     2005-01-05        28.79  ...                              1.00000     0.00
1     2005-01-06        29.18  ...                              1.00000     0.00
2     2005-01-07        28.73  ...                              0.66667     0.00
3     2005-01-10        28.84  ...                              0.50000     0.00
4     2005-01-11        29.09  ...                              1.00000     0.00
...          ...          ...  ...                                  ...      ...
4795  2024-10-13        29.58  ...                              0.15534  3887.17
4796  2024-10-14        30.34  ...                              0.20725  3961.34
4797  2024-10-15        29.75  ...                              0.18995  3855.99
4798  2024-10-16        29.67  ...                              0.20643  3831.59
4799  2024-10-17        29.56  ...                              0.20107  3788.22
[4800 rows x 14 columns]
```
