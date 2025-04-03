#### A 股估值指标

接口: stock_zh_valuation_baidu

目标地址: https://gushitong.baidu.com/stock/ab-002044

描述: 百度股市通-A 股-财务报表-估值数据

限量: 单次获取指定 symbol 和 indicator 的所有历史数据

输入参数

| 名称        | 类型  | 描述                                                                     |
|-----------|-----|------------------------------------------------------------------------|
| symbol    | str | symbol="002044"; A 股代码                                                 |
| indicator | str | indicator="总市值"; choice of {"总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"} |
| period    | str | period="近一年"; choice of {"近一年", "近三年", "近五年", "近十年", "全部"}             |

输出参数

| 名称    | 类型      | 描述  |
|-------|---------|-----|
| date  | object  | -   |
| value | float64 | -   |

接口示例

```python
import akshare as ak

stock_zh_valuation_baidu_df = ak.stock_zh_valuation_baidu(symbol="002044", indicator="总市值", period="近一年")
print(stock_zh_valuation_baidu_df)
```

数据示例

```
           date   value
0    2023-05-29  245.42
1    2023-05-30  246.60
2    2023-05-31  249.73
3    2023-06-01  253.64
4    2023-06-02  259.52
..          ...     ...
362  2024-05-25  167.92
363  2024-05-26  167.92
364  2024-05-27  165.96
365  2024-05-28  164.40
366  2024-05-29  167.14
[367 rows x 2 columns]
```
