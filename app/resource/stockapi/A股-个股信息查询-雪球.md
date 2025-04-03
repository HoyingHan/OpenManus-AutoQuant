#### 个股信息查询-雪球

接口: stock_individual_basic_info_xq

目标地址: https://xueqiu.com/snowman/S/SH601127/detail#/GSJJ

描述: 雪球财经-个股-公司概况-公司简介

限量: 单次返回指定 symbol 的个股信息

输入参数

| 名称      | 类型    | 描述                      |
|---------|-------|-------------------------|
| symbol  | str   | symbol="SH601127"; 股票代码 |
| token   | str   | token=None;             |
| timeout | float | timeout=None; 默认不设置超时参数 |

输出参数

| 名称    | 类型     | 描述  |
|-------|--------|-----|
| item  | object | -   |
| value | object | -   |

接口示例

```python
import akshare as ak

stock_individual_basic_info_xq_df = ak.stock_individual_basic_info_xq(symbol="SH601127")
print(stock_individual_basic_info_xq_df)
```

数据示例

```
                            item                                              value
0                         org_id                                         T000071215
1                    org_name_cn                                        赛力斯集团股份有限公司
2              org_short_name_cn                                                赛力斯
3                    org_name_en                               Seres Group Co.,Ltd.
4              org_short_name_en                                              SERES
5        main_operation_business      新能源汽车及核心三电(电池、电驱、电控)、传统汽车及核心部件总成的研发、制造、销售及服务。
6                operating_scope  　　一般项目：制造、销售：汽车零部件、机动车辆零部件、普通机械、电器机械、电器、电子产品（不...
7                district_encode                                             500106
8            org_cn_introduction  赛力斯始创于1986年，是以新能源汽车为核心业务的技术科技型汽车企业。现有员工1.6万人，A...
9           legal_representative                                                张正萍
10               general_manager                                                张正萍
11                     secretary                                                 申薇
12              established_date                                      1178812800000
13                     reg_asset                                       1509782193.0
14                     staff_num                                              16102
15                     telephone                                     86-23-65179666
16                      postcode                                             401335
17                           fax                                     86-23-65179777
18                         email                                    601127@seres.cn
19                   org_website                                   www.seres.com.cn
20                reg_address_cn                                      重庆市沙坪坝区五云湖路7号
21                reg_address_en                                               None
22             office_address_cn                                      重庆市沙坪坝区五云湖路7号
23             office_address_en                                               None
24               currency_encode                                             019001
25                      currency                                                CNY
26                   listed_date                                      1465920000000
27               provincial_name                                                重庆市
28             actual_controller                                       张兴海 (13.79%)
29                   classi_name                                               民营企业
30                   pre_name_cn                                     重庆小康工业集团股份有限公司
31                      chairman                                                张正萍
32               executives_nums                                                 20
33              actual_issue_vol                                        142500000.0
34                   issue_price                                               5.81
35             actual_rc_net_amt                                        738451000.0
36              pe_after_issuing                                              18.19
37  online_success_rate_of_issue                                           0.110176
38            affiliate_industry         {'ind_code': 'BK0025', 'ind_name': '汽车整车'}
```
