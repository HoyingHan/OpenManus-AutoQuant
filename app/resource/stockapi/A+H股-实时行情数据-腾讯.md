#### 实时行情数据-腾讯

接口: stock_zh_ah_spot

目标地址: https://stockapp.finance.qq.com/mstats/#mod=list&id=hk_ah&module=HK&type=AH

描述: A+H 股数据是从腾讯财经获取的数据, 延迟 15 分钟更新

限量: 单次返回所有 A+H 上市公司的实时行情数据

输入参数

| 名称  | 类型  | 描述  |
|-----|-----|-----|
| -   | -   | -   |

输出参数

| 名称  | 类型      | 描述      |
|-----|---------|---------|
| 代码  | object  | -       |
| 名称  | object  | -       |
| 最新价 | float64 | -       |
| 涨跌幅 | float64 | 注意单位: % |
| 涨跌额 | float64 | -       |
| 买入  | float64 | -       |
| 卖出  | float64 | -       |
| 成交量 | float64 | -       |
| 成交额 | float64 | -       |
| 今开  | float64 | -       |
| 昨收  | float64 | -       |
| 最高  | float64 | -       |
| 最低  | float64 | -       |

接口示例

```python
import akshare as ak

stock_zh_ah_spot_df = ak.stock_zh_ah_spot()
print(stock_zh_ah_spot_df)
```

数据示例

```
        代码       名称   最新价   涨跌幅   涨跌额  ...          成交额    今开    昨收    最高    最低
0    00525   广深铁路股份  1.66 -0.60 -0.01  ...  18427959.54  1.67  1.67  1.69  1.63
1    01618     中国中冶  1.59 -1.85 -0.03  ...  11468956.78  1.62  1.62  1.63  1.59
2    02727     上海电气  1.54  0.00  0.00  ...  12801574.00  1.54  1.54  1.57  1.53
3    03369     秦港股份  1.49  0.00  0.00  ...   6051885.00  1.49  1.49  1.58  1.48
4    03678     弘业期货  1.41 -0.70 -0.01  ...   1871710.00  1.43  1.42  1.44  1.41
..     ...      ...   ...   ...   ...  ...          ...   ...   ...   ...   ...
155  06806     申万宏源  1.39 -1.42 -0.02  ...   4312447.20  1.41  1.41  1.42  1.39
156  00991     大唐发电  1.24 -2.36 -0.03  ...  29927046.00  1.26  1.27  1.27  1.23
157  00323  马鞍山钢铁股份  1.22 -0.81 -0.01  ...   2345447.26  1.23  1.23  1.25  1.22
158  01635     大众公用  1.20  0.84  0.01  ...    371140.00  1.20  1.19  1.20  1.19
159  01375     中州证券  1.11 -1.77 -0.02  ...   4867240.00  1.13  1.13  1.13  1.09
[160 rows x 13 columns]
```
