#### 股票列表-A股

接口: stock_info_a_code_name

目标地址: 沪深京三个交易所

描述: 沪深京 A 股股票代码和股票简称数据

限量: 单次获取所有 A 股股票代码和简称数据

输入参数

| 名称  | 类型  | 描述  |
|-----|-----|-----|
| -   | -   | -   |

输出参数

| 名称   | 类型     | 描述  |
|------|--------|-----|
| code | object | -   |
| name | object | -   |

接口示例

```python
import akshare as ak

stock_info_a_code_name_df = ak.stock_info_a_code_name()
print(stock_info_a_code_name_df)
```

数据示例

```
        code   name
0     000001   平安银行
1     000002  万  科Ａ
2     000004   国华网安
3     000005   ST星源
4     000006   深振业Ａ
      ...    ...
4623  871396   常辅股份
4624  871553   凯腾精工
4625  871642   通易航天
4626  871981   晶赛科技
4627  872925   锦好医疗
```
