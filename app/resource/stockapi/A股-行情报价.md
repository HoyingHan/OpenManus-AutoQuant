#### 行情报价

接口: stock_bid_ask_em

目标地址: https://quote.eastmoney.com/sz000001.html

描述: 东方财富-行情报价

限量: 单次返回指定股票的行情报价数据

输入参数

| 名称     | 类型  | 描述                    |
|--------|-----|-----------------------|
| symbol | str | symbol="000001"; 股票代码 |

输出参数

| 名称    | 类型      | 描述 |
|-------|---------|----|
| item  | object  | -  |
| value | float64 | -  |

接口示例

```python
import akshare as ak

stock_bid_ask_em_df = ak.stock_bid_ask_em(symbol="000001")
print(stock_bid_ask_em_df)
```

数据示例

```
          item         value
0       sell_5  1.049000e+01
1   sell_5_vol  1.147100e+06
2       sell_4  1.048000e+01
3   sell_4_vol  1.035700e+06
4       sell_3  1.047000e+01
5   sell_3_vol  1.489100e+06
6       sell_2  1.046000e+01
7   sell_2_vol  1.608400e+06
8       sell_1  1.045000e+01
9   sell_1_vol  2.339000e+05
10       buy_1  1.044000e+01
11   buy_1_vol  3.690000e+05
12       buy_2  1.043000e+01
13   buy_2_vol  8.359000e+05
14       buy_3  1.042000e+01
15   buy_3_vol  6.016000e+05
16       buy_4  1.041000e+01
17   buy_4_vol  7.381000e+05
18       buy_5  1.040000e+01
19   buy_5_vol  1.301900e+06
20          最新  1.045000e+01
21          均价  1.043000e+01
22          涨幅  4.800000e-01
23          涨跌  5.000000e-02
24          总手  8.726630e+05
25          金额  9.102786e+08
26          换手  4.500000e-01
27          量比  4.400000e-01
28          最高  1.047000e+01
29          最低  1.037000e+01
30          今开  1.038000e+01
31          昨收  1.040000e+01
32          涨停  1.144000e+01
33          跌停  9.360000e+00
34          外盘  4.715810e+05
35          内盘  4.010820e+05
```
