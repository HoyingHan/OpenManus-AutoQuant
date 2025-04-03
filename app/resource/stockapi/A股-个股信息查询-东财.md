#### 个股信息查询-东财

接口: stock_individual_info_em

目标地址: http://quote.eastmoney.com/concept/sh603777.html?from=classic

描述: 东方财富-个股-股票信息

限量: 单次返回指定 symbol 的个股信息

输入参数

| 名称      | 类型    | 描述                      |
|---------|-------|-------------------------|
| symbol  | str   | symbol="603777"; 股票代码   |
| timeout | float | timeout=None; 默认不设置超时参数 |

输出参数

| 名称    | 类型     | 描述  |
|-------|--------|-----|
| item  | object | -   |
| value | object | -   |

接口示例

```python
import akshare as ak

stock_individual_info_em_df = ak.stock_individual_info_em(symbol="000001")
print(stock_individual_info_em_df)
```

数据示例

```
   item                value
0   总市值  337468917463.220032
1  流通市值      337466070320.25
2    行业                   银行
3  上市时间             19910403
4  股票代码               000001
5  股票简称                 平安银行
6   总股本        19405918198.0
7   流通股        19405754475.0
```
