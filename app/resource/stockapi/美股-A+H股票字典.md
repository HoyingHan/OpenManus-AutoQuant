#### A+H股票字典

接口: stock_zh_ah_name

目标地址: https://stockapp.finance.qq.com/mstats/#mod=list&id=hk_ah&module=HK&type=AH

描述: A+H 股数据是从腾讯财经获取的数据, 历史数据按日频率更新

限量: 单次返回所有 A+H 上市公司的代码和名称

输入参数

| 名称  | 类型  | 描述  |
|-----|-----|-----|
| -   | -   | -   |

输出参数

| 名称  | 类型     | 描述  |
|-----|--------|-----|
| 代码  | object | -   |
| 名称  | object | -   |

接口示例

```python
import akshare as ak

stock_zh_ah_name_df = ak.stock_zh_ah_name()
print(stock_zh_ah_name_df)
```

数据示例

```
     代码        名称
0    01211     比亚迪股份
1    06160      百济神州
2    01880      中国中免
3    00941      中国移动
4    06821       凯莱英
..     ...       ...
144  01053    重庆钢铁股份
145  00588  北京北辰实业股份
146  02009      金隅集团
147  02880      辽港股份
148  01033     中石化油服
[149 rows x 2 columns]
```

