#### A 股等权重与中位数市净率

接口: stock_a_all_pb

目标地址: https://www.legulegu.com/stockdata/all-pb

描述: 乐咕乐股-A 股等权重与中位数市净率

限量: 单次返回所有数据

输入参数

| 名称  | 类型  | 描述 |
|-----|-----|----|
| -   | -   | -  |

输出参数

| 名称                                          | 类型      | 描述                     |
|---------------------------------------------|---------|------------------------|
| date                                        | object  | 日期                     |
| middlePB                                    | float64 | 全部A股市净率中位数             |
| equalWeightAveragePB                        | float64 | 全部A股市净率等权平均            |
| close                                       | float64 | 上证指数                   |
| quantileInAllHistoryMiddlePB                | float64 | 当前市净率中位数在历史数据上的分位数     |
| quantileInRecent10YearsMiddlePB             | float64 | 当前市净率中位数在最近10年数据上的分位数  |
| quantileInAllHistoryEqualWeightAveragePB    | float64 | 当前市净率等权平均在历史数据上的分位数    |
| quantileInRecent10YearsEqualWeightAveragePB | float64 | 当前市净率等权平均在最近10年数据上的分位数 |

接口示例

```python
import akshare as ak

stock_a_all_pb_df = ak.stock_a_all_pb()
print(stock_a_all_pb_df)
```

数据示例

```
            date  ...  quantileInRecent10YearsEqualWeightAveragePB
0     2005-01-04  ...                                      1.00000
1     2005-01-05  ...                                      1.00000
2     2005-01-06  ...                                      0.66667
3     2005-01-07  ...                                      0.75000
4     2005-01-10  ...                                      1.00000
...          ...  ...                                          ...
4793  2024-10-11  ...                                      0.10763
4794  2024-10-14  ...                                      0.12170
4795  2024-10-15  ...                                      0.11551
4796  2024-10-16  ...                                      0.11840
4797  2024-10-17  ...                                      0.12129
[4798 rows x 8 columns]
```
