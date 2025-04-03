#### MSCI

接口: stock_esg_msci_sina

目标地址: https://finance.sina.com.cn/esg/grade.shtml

描述: 新浪财经-ESG评级中心-ESG评级-MSCI

限量: 单次返回所有数据

输入参数

| 名称  | 类型  | 描述  |
|-----|-----|-----|
| -   | -   | -   |

输出参数

| 名称     | 类型      | 描述 |
|--------|---------|----|
| 股票代码   | object  | -  |
| ESG评分  | object  | -  |
| 环境总评   | float64 | -  |
| 社会责任总评 | float64 | -  |
| 治理总评   | float64 | -  |
| 评级日期   | object  | -  |
| 交易市场   | object  | -  |

接口示例

```python
import akshare as ak

stock_esg_msci_sina_df = ak.stock_esg_msci_sina()
print(stock_esg_msci_sina_df)
```

数据示例

```
        股票代码 ESG评分 环境总评 社会责任总评 治理总评 评级日期 交易市场
0      00019.HK   AAA   7.0     7.3   6.3  2024-04-24   HK
1     000513.SZ   AAA   6.8     6.4   6.1  2024-06-25   CN
2      00066.HK   AAA   7.2     5.6   6.5  2024-07-11   HK
3      00087.HK   AAA   7.0     7.3   6.3  2024-04-24   HK
4      00992.HK   AAA   5.1     6.2   5.5  2024-07-08   HK
...         ...   ...   ...     ...   ...         ...  ...
4619   UONEK.US   CCC   7.0     3.2   1.0  2024-06-24   US
4620     UVE.US   CCC   1.9     2.0   5.4  2024-07-08   US
4621     VTS.US   CCC   1.1     2.4   6.8  2024-04-24   US
4622    WULF.US   CCC   0.3     1.3   3.9  2024-05-20   US
4623    ZETA.US   CCC   6.7     3.1   2.0  2024-06-24   US
[4624 rows x 7 columns]
```
