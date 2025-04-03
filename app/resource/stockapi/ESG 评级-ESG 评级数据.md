#### ESG 评级数据

接口: stock_esg_rate_sina

目标地址: https://finance.sina.com.cn/esg/grade.shtml

描述: 新浪财经-ESG评级中心-ESG评级-ESG评级数据

限量: 单次返回所有数据

输入参数

| 名称  | 类型  | 描述  |
|-----|-----|-----|
| -   | -   | -   |

输出参数

| 名称    | 类型     | 描述 |
|-------|--------|----|
| 成分股代码 | object | -  |
| 评级机构  | object | -  |
| 评级    | object | -  |
| 评级季度  | object | -  |
| 标识    | object | -  |
| 交易市场  | object | -  |

接口示例

```python
import akshare as ak

stock_esg_rate_sina_df = ak.stock_esg_rate_sina()
print(stock_esg_rate_sina_df)
```

数据示例

```
    成分股代码              评级机构     评级    评级季度             标识 交易市场
0      SZ000001             中财绿金院     A-  2022Q4            NaN   cn
1      SZ000001              商道融绿     B+  2022Q4            NaN   cn
2      SZ000001                盟浪      A  2022Q2            NaN   cn
3      SZ000001               中诚信    AA-  2023Q3            NaN   cn
4      SZ000001  晨星Sustainalytics  24.96  2022Q4  Comprehensive   cn
         ...               ...    ...     ...            ...  ...
46888   HK02361                盟浪      -  2022Q2            NaN   hk
46889   HK02361               中诚信     BB  2023Q3            NaN   hk
46890   HK02361  晨星Sustainalytics      -  2022Q4                  hk
46891   HK02361                妙盈      -  2022Q2            NaN   hk
46892   HK02361             华测CTI      -  2022Q1            NaN   hk
[46893 rows x 6 columns]
```
