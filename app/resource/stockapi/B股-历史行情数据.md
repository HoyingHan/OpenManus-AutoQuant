#### 历史行情数据

接口: stock_zh_a_cdr_daily

目标地址: https://finance.sina.com.cn/realstock/company/sh689009/nc.shtml

描述: 上海证券交易所-科创板-CDR

限量: 单次返回指定 CDR 的日频率数据, 分钟历史行情数据可以通过 stock_zh_a_minute 获取

名词解释:

1. [Investopedia-CDR](https://www.investopedia.com/terms/c/cdr.asp)
2. [百度百科-中国存托凭证](https://baike.baidu.com/item/%E4%B8%AD%E5%9B%BD%E5%AD%98%E6%89%98%E5%87%AD%E8%AF%81/2489906?fr=aladdin)

输入参数

| 名称         | 类型  | 描述                          |
|------------|-----|-----------------------------|
| symbol     | str | symbol='sh689009'; CDR 股票代码 |
| start_date | str | start_date='20201103'       |
| end_date   | str | end_date='20201116'         |

输出参数

| 名称     | 类型      | 描述      |
|--------|---------|---------|
| date   | object  | 交易日     |
| open   | float64 | -       |
| high   | float64 | -       |
| low    | float64 | -       |
| close  | float64 | -       |
| volume | float64 | 注意单位: 手 |

接口示例

```python
import akshare as ak

stock_zh_a_cdr_daily_df = ak.stock_zh_a_cdr_daily(symbol='sh689009', start_date='20201103', end_date='20201116')
print(stock_zh_a_cdr_daily_df)
```

数据示例

```
           date   open   high    low  close      volume
0    2020-10-29  33.00  49.80  33.00  38.50  40954922.0
1    2020-10-30  40.02  51.56  40.02  47.60  33600551.0
2    2020-11-02  50.20  56.78  48.81  56.77  27193402.0
3    2020-11-03  56.50  59.55  53.36  57.39  25121445.0
4    2020-11-04  57.45  57.80  51.90  54.40  20846450.0
..          ...    ...    ...    ...    ...         ...
265  2021-11-30  60.50  61.20  60.36  60.84   1535244.0
266  2021-12-01  60.99  61.00  59.78  59.78   2116728.0
267  2021-12-02  59.67  59.77  57.66  57.73   2421344.0
268  2021-12-03  57.55  58.78  57.50  58.60   1709167.0
269  2021-12-06  58.99  58.99  56.30  56.75   1719351.0
```

