# API 接口概览

## A+H股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| A+H股-实时行情数据-东财 | stock_zh_ah_spot_em | 东方财富网-行情中心-沪深港通-AH股比价-实时行情, 延迟 15 分钟更新 |
| A+H股-实时行情数据-腾讯 | stock_zh_ah_spot | A+H 股数据是从腾讯财经获取的数据, 延迟 15 分钟更新 |
| A+H股-历史行情数据 | stock_zh_ah_daily | 腾讯财经-A+H 股数据 |
| A+H股-A+H股票字典 | stock_zh_ah_name | A+H 股数据是从腾讯财经获取的数据, 历史数据按日频率更新 |

## A股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| A股-股票市场总貌 | stock_sse_summary | 上海证券交易所-股票数据总貌 |
| A股-股票市场总貌 | stock_szse_summary | 深圳证券交易所-市场总貌-证券类别统计 |
| A股-股票市场总貌 | stock_szse_area_summary | 深圳证券交易所-市场总貌-地区交易排序 |
| A股-股票市场总貌 | stock_szse_sector_summary | 深圳证券交易所-统计资料-股票行业成交数据 |
| A股-股票市场总貌 | stock_sse_deal_daily | 上海证券交易所-数据-股票数据-成交概况-股票成交概况-每日股票情况 |
| A股-个股信息查询-东财 | stock_individual_info_em | 东方财富-个股-股票信息 |
| A股-个股信息查询-雪球 | stock_individual_basic_info_xq | 雪球财经-个股-公司概况-公司简介 |
| A股-行情报价 | stock_bid_ask_em | 东方财富-行情报价 |
| A股-实时行情数据 | stock_zh_a_spot_em | 东方财富网-沪深京 A 股-实时行情数据 |
| A股-实时行情数据 | stock_sh_a_spot_em | 东方财富网-沪 A 股-实时行情数据 |
| A股-实时行情数据 | stock_sz_a_spot_em | 东方财富网-深 A 股-实时行情数据 |
| A股-实时行情数据 | stock_bj_a_spot_em | 东方财富网-京 A 股-实时行情数据 |
| A股-实时行情数据 | stock_new_a_spot_em | 东方财富网-新股-实时行情数据 |
| A股-实时行情数据 | stock_cy_a_spot_em | 东方财富网-创业板-实时行情 |
| A股-实时行情数据 | stock_kc_a_spot_em | 东方财富网-科创板-实时行情 |
| A股-实时行情数据 | stock_zh_a_spot | 新浪财经-沪深京 A 股数据, 重复运行本函数会被新浪暂时封 IP, 建议增加时间间隔 |
| A股-实时行情数据 | stock_individual_spot_xq | 雪球-行情中心-个股 |
| A股-历史行情数据 | stock_zh_a_hist | 东方财富-沪深京 A 股日频率数据; 历史数据按日频率更新, 当日收盘价请在收盘后获取 |
| A股-历史行情数据 | stock_zh_a_daily | 新浪财经-沪深京 A 股的数据, 历史数据按日频率更新; 注意其中的 **sh689009** 为 CDR, 请 通过 **ak.stock_zh_a_cdr_daily** 接口获取 |
| A股-历史行情数据 | stock_zh_a_hist_tx | 腾讯证券-日频-股票历史数据; 历史数据按日频率更新, 当日收盘价请在收盘后获取 |
| A股-历史行情数据 | stock_zh_a_minute | 新浪财经-沪深京 A 股股票或者指数的分时数据，目前可以获取 1, 5, 15, 30, 60 分钟的数据频率, 可以指定是否复权 |
| A股-历史行情数据 | stock_zh_a_hist_min_em | 东方财富网-行情首页-沪深京 A 股-每日分时行情; 该接口只能获取近期的分时数据，注意时间周期的设置 |
| A股-历史行情数据 | stock_intraday_em | 东方财富-分时数据 |
| A股-历史行情数据 | stock_intraday_sina | 新浪财经-日内分时数据 |
| A股-历史行情数据 | stock_zh_a_hist_pre_min_em | 东方财富-股票行情-盘前数据 |
| A股-历史分笔数据 | stock_zh_a_tick_tx | 每个交易日 16:00 提供当日数据; 如遇到数据缺失, 请使用 **ak.stock_zh_a_tick_163()** 接口(注意数据会有一定差异) |
| A股-CDR-历史行情数据 | stock_zh_a_cdr_daily | 上海证券交易所-科创板-CDR |

## B股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| B股-实时行情数据 | stock_zh_b_spot_em | 东方财富网-实时行情数据 |
| B股-实时行情数据 | stock_zh_b_spot | B 股数据是从新浪财经获取的数据, 重复运行本函数会被新浪暂时封 IP, 建议增加时间间隔 |
| B股-历史行情数据 | stock_zh_b_daily | B 股数据是从新浪财经获取的数据, 历史数据按日频率更新 |
| B股-历史行情数据 | stock_zh_b_minute | 新浪财经 B 股股票或者指数的分时数据，目前可以获取 1, 5, 15, 30, 60 分钟的数据频率, 可以指定是否复权 |

## ESG 评级

| 分类 | 接口名 | 描述 |
|------|--------|------|
| ESG 评级-ESG 评级数据 | stock_esg_rate_sina | 新浪财经-ESG评级中心-ESG评级-ESG评级数据 |
| ESG 评级-MSCI | stock_esg_msci_sina | 新浪财经-ESG评级中心-ESG评级-MSCI |
| ESG 评级-路孚特 | stock_esg_rft_sina | 新浪财经-ESG评级中心-ESG评级-路孚特 |
| ESG 评级-秩鼎 | stock_esg_zd_sina | 新浪财经-ESG评级中心-ESG评级-秩鼎 |
| ESG 评级-华证指数 | stock_esg_hz_sina | 新浪财经-ESG评级中心-ESG评级-华证指数 |

## IPO 受益股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| IPO 受益股-公司动态 | stock_ipo_benefit_ths | 同花顺-数据中心-新股数据-IPO受益股 |

## 一致行动人

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 一致行动人-营业部排行 | stock_yzxdr_em | 东方财富网-数据中心-特色数据-一致行动人 |

## 两网及退市

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 两网及退市-公司动态 | stock_zh_a_stop_em | 东方财富网-行情中心-沪深个股-两网及退市 |

## 个股新闻

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 个股新闻-沪深港通持股-个股详情 | stock_news_em | 东方财富指定个股的新闻资讯数据 |

## 主营介绍

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 主营介绍-同花顺-机构调研-详细 | stock_zyjs_ths | 同花顺-主营介绍 |

## 主营构成

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 主营构成-东财-机构调研-详细 | stock_zygc_em | 东方财富网-个股-主营构成 |
| 主营构成-益盟-机构调研-详细 | stock_zygc_ym | 益盟-F10-主营构成 |

## 停复牌

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 停复牌-沪深港通持股-个股详情 | news_trade_notify_suspend_baidu | 百度股市通-交易提醒-停复牌 |

## 停复牌信息

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 停复牌信息-沪深港通持股-个股详情 | stock_tfp_em | 东方财富网-数据中心-特色数据-停复牌信息 |

## 分析师指数

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 分析师指数-分析师指数排行 | stock_analyst_rank_em | 东方财富网-数据中心-研究报告-东方财富分析师指数 |
| 分析师指数-分析师详情 | stock_analyst_detail_em | 东方财富网-数据中心-研究报告-东方财富分析师指数-分析师详情 |

## 分红派息

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 分红派息-沪深港通持股-个股详情 | news_trade_notify_dividend_baidu | 百度股市通-交易提醒-分红派息 |

## 分红配送

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 分红配送-分红配送-东财 | stock_fhps_em | 东方财富-数据中心-年报季报-分红配送 |
| 分红配送-分红配送详情-东财 | stock_fhps_detail_em | 东方财富网-数据中心-分红送配-分红送配详情 |
| 分红配送-分红情况-同花顺 | stock_fhps_detail_ths | 同花顺-分红情况 |
| 分红配送-分红配送详情-港股-同花顺 | stock_hk_fhpx_detail_ths | 同花顺-港股-分红派息 |

## 千股千评

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 千股千评-分析师详情 | stock_comment_em | 东方财富网-数据中心-特色数据-千股千评 |

## 千股千评详情

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 千股千评详情-主力控盘 | stock_comment_detail_zlkp_jgcyd_em | 东方财富网-数据中心-特色数据-千股千评-主力控盘-机构参与度 |
| 千股千评详情-综合评价 | stock_comment_detail_zhpj_lspf_em | 东方财富网-数据中心-特色数据-千股千评-综合评价-历史评分 |
| 千股千评详情-市场热度 | stock_comment_detail_scrd_focus_em | 东方财富网-数据中心-特色数据-千股千评-市场热度-用户关注指数 |
| 千股千评详情-市场热度 | stock_comment_detail_scrd_desire_em | 东方财富网-数据中心-特色数据-千股千评-市场热度-市场参与意愿 |
| 千股千评详情-市场热度 | stock_comment_detail_scrd_desire_daily_em | 东方财富网-数据中心-特色数据-千股千评-市场热度-日度市场参与意愿 |
| 千股千评详情-市场热度 | stock_comment_detail_scrd_cost_em | 东方财富网-数据中心-特色数据-千股千评-市场热度-市场成本 |

## 商誉专题

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 商誉专题-A股商誉市场概况 | stock_sy_profile_em | 东方财富网-数据中心-特色数据-商誉-A股商誉市场概况 |
| 商誉专题-商誉减值预期明细 | stock_sy_yq_em | 东方财富网-数据中心-特色数据-商誉-商誉减值预期明细 |
| 商誉专题-个股商誉减值明细 | stock_sy_jz_em | 东方财富网-数据中心-特色数据-商誉-个股商誉减值明细 |
| 商誉专题-个股商誉明细 | stock_sy_em | 东方财富网-数据中心-特色数据-商誉-个股商誉明细 |
| 商誉专题-行业商誉 | stock_sy_hy_em | 东方财富网-数据中心-特色数据-商誉-行业商誉 |

## 基本面数据

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 基本面数据-股东大会 | stock_gddh_em | 东方财富网-数据中心-股东大会 |
| 基本面数据-重大合同 | stock_zdhtmx_em | 东方财富网-数据中心-重大合同-重大合同明细 |
| 基本面数据-个股研报 | stock_research_report_em | 东方财富网-数据中心-研究报告-个股研报 |
| 基本面数据-沪深京 A 股公告 | stock_notice_report | 东方财富网-数据中心-公告大全-沪深京 A 股公告 |
| 基本面数据-财务报表-新浪 | stock_financial_report_sina | 新浪财经-财务报表-三大报表 |
| 基本面数据-财务报表-东财 | stock_balance_sheet_by_report_em | 东方财富-股票-财务分析-资产负债表-按报告期 |
| 基本面数据-财务报表-东财 | stock_balance_sheet_by_yearly_em | 东方财富-股票-财务分析-资产负债表-按年度 |
| 基本面数据-财务报表-东财 | stock_profit_sheet_by_report_em | 东方财富-股票-财务分析-利润表-报告期 |
| 基本面数据-财务报表-东财 | stock_profit_sheet_by_yearly_em | 东方财富-股票-财务分析-利润表-按年度 |
| 基本面数据-财务报表-东财 | stock_profit_sheet_by_quarterly_em | 东方财富-股票-财务分析-利润表-按单季度 |
| 基本面数据-财务报表-东财 | stock_cash_flow_sheet_by_report_em | 东方财富-股票-财务分析-现金流量表-按报告期 |
| 基本面数据-财务报表-东财 | stock_cash_flow_sheet_by_yearly_em | 东方财富-股票-财务分析-现金流量表-按年度 |
| 基本面数据-财务报表-东财 | stock_cash_flow_sheet_by_quarterly_em | 东方财富-股票-财务分析-现金流量表-按单季度 |
| 基本面数据-财务报表-同花顺 | stock_financial_debt_ths | 同花顺-财务指标-资产负债表 |
| 基本面数据-财务报表-同花顺 | stock_financial_benefit_ths | 同花顺-财务指标-利润表 |
| 基本面数据-财务报表-同花顺 | stock_financial_cash_ths | 同花顺-财务指标-现金流量表 |
| 基本面数据-财务报表-东财-已退市股票 | stock_balance_sheet_by_report_delisted_em | 东方财富-股票-财务分析-资产负债表-已退市股票-按报告期 |
| 基本面数据-财务报表-东财-已退市股票 | stock_profit_sheet_by_report_delisted_em | 东方财富-股票-财务分析-利润表-已退市股票-按报告期 |
| 基本面数据-财务报表-东财-已退市股票 | stock_cash_flow_sheet_by_report_delisted_em | 东方财富-股票-财务分析-现金流量表-已退市股票-按报告期 |
| 基本面数据-港股财务报表 | stock_financial_hk_report_em | 东方财富-港股-财务报表-三大报表 |
| 基本面数据-美股财务报表 | stock_financial_us_report_em | 东方财富-美股-财务分析-三大报表 |
| 基本面数据-关键指标-新浪 | stock_financial_abstract | 新浪财经-财务报表-关键指标 |
| 基本面数据-关键指标-同花顺 | stock_financial_abstract_ths | 同花顺-财务指标-主要指标 |
| 基本面数据-财务指标 | stock_financial_analysis_indicator | 新浪财经-财务分析-财务指标 |
| 基本面数据-港股财务指标 | stock_financial_hk_analysis_indicator_em | 东方财富-港股-财务分析-主要指标 |
| 基本面数据-美股财务指标 | stock_financial_us_analysis_indicator_em | 东方财富-美股-财务分析-主要指标 |
| 基本面数据-历史分红 | stock_history_dividend | 新浪财经-发行与分配-历史分红 |
| 基本面数据-十大流通股东(个股) | stock_gdfx_free_top_10_em | 东方财富网-个股-十大流通股东 |
| 基本面数据-十大股东(个股) | stock_gdfx_top_10_em | 东方财富网-个股-十大股东 |
| 基本面数据-股东持股变动统计-十大流通股东 | stock_gdfx_free_holding_change_em | 东方财富网-数据中心-股东分析-股东持股变动统计-十大流通股东 |
| 基本面数据-股东持股变动统计-十大股东 | stock_gdfx_holding_change_em | 东方财富网-数据中心-股东分析-股东持股变动统计-十大股东 |
| 基本面数据-高管持股变动统计 | stock_management_change_ths | 同花顺-公司大事-高管持股变动 |
| 基本面数据-股东持股变动统计 | stock_shareholder_change_ths | 同花顺-公司大事-股东持股变动 |
| 基本面数据-股东持股分析-十大流通股东 | stock_gdfx_free_holding_analyse_em | 东方财富网-数据中心-股东分析-股东持股分析-十大流通股东 |
| 基本面数据-股东持股分析-十大股东 | stock_gdfx_holding_analyse_em | 东方财富网-数据中心-股东分析-股东持股分析-十大股东 |
| 基本面数据-股东持股明细-十大流通股东 | stock_gdfx_free_holding_detail_em | 东方财富网-数据中心-股东分析-股东持股明细-十大流通股东 |
| 基本面数据-股东持股明细-十大股东 | stock_gdfx_holding_detail_em | 东方财富网-数据中心-股东分析-股东持股明细-十大股东 |
| 基本面数据-股东持股统计-十大流通股东 | stock_gdfx_free_holding_statistics_em | 东方财富网-数据中心-股东分析-股东持股统计-十大股东 |
| 基本面数据-股东持股统计-十大股东 | stock_gdfx_holding_statistics_em | 东方财富网-数据中心-股东分析-股东持股统计-十大股东 |
| 基本面数据-股东协同-十大流通股东 | stock_gdfx_free_holding_teamwork_em | 东方财富网-数据中心-股东分析-股东协同-十大流通股东 |
| 基本面数据-股东协同-十大股东 | stock_gdfx_holding_teamwork_em | 东方财富网-数据中心-股东分析-股东协同-十大股东 |
| 基本面数据-股东户数 | stock_zh_a_gdhs | 东方财富网-数据中心-特色数据-股东户数数据 |
| 基本面数据-股东户数详情 | stock_zh_a_gdhs_detail_em | 东方财富网-数据中心-特色数据-股东户数详情 |
| 基本面数据-分红配股 | stock_history_dividend_detail | 新浪财经-发行与分配-分红配股 |
| 基本面数据-历史分红 | stock_dividend_cninfo | 巨潮资讯-个股-历史分红 |
| 基本面数据-新股发行 | stock_ipo_info | 新浪财经-发行与分配-新股发行 |
| 基本面数据-股票增发 | stock_add_stock | 新浪财经-发行与分配-增发 |
| 基本面数据-限售解禁 | stock_restricted_release_queue_sina | 新浪财经-发行分配-限售解禁 |
| 基本面数据-限售解禁 | stock_restricted_release_summary_em | 东方财富网-数据中心-特色数据-限售股解禁 |
| 基本面数据-限售解禁 | stock_restricted_release_detail_em | 东方财富网-数据中心-限售股解禁-解禁详情一览 |
| 基本面数据-限售解禁 | stock_restricted_release_queue_em | 东方财富网-数据中心-个股限售解禁-解禁批次 |
| 基本面数据-限售解禁 | stock_restricted_release_stockholder_em | 东方财富网-数据中心-个股限售解禁-解禁股东 |
| 基本面数据-流通股东 | stock_circulate_stock_holder | 新浪财经-股东股本-流通股东 |
| 基本面数据-板块行情 | stock_sector_spot | 新浪行业-板块行情 |
| 基本面数据-板块详情 | stock_sector_detail | 新浪行业-板块行情-成份详情, 由于新浪网页提供的统计数据有误, 部分行业数量大于统计数 |
| 基本面数据-股票列表-A股 | stock_info_a_code_name | 沪深京 A 股股票代码和股票简称数据 |
| 基本面数据-股票列表-上证 | stock_info_sh_name_code | 上海证券交易所股票代码和简称数据 |
| 基本面数据-股票列表-深证 | stock_info_sz_name_code | 深证证券交易所股票代码和股票简称数据 |
| 基本面数据-股票列表-北证 | stock_info_bj_name_code | 北京证券交易所股票代码和简称数据 |
| 基本面数据-终止/暂停上市-深证 | stock_info_sz_delist | 深证证券交易所终止/暂停上市股票 |
| 基本面数据-两网及退市 | stock_staq_net_stop | 东方财富网-行情中心-沪深个股-两网及退市 |
| 基本面数据-暂停/终止上市-上证 | stock_info_sh_delist | 上海证券交易所暂停/终止上市股票 |
| 基本面数据-股票更名 | stock_info_change_name | 新浪财经-股票曾用名 |
| 基本面数据-名称变更-深证 | stock_info_sz_change_name | 深证证券交易所-市场数据-股票数据-名称变更 |
| 基本面数据-基金持股 | stock_fund_stock_holder | 新浪财经-股本股东-基金持股 |
| 基本面数据-主要股东 | stock_main_stock_holder | 新浪财经-股本股东-主要股东 |
| 基本面数据-机构持股 | stock_institute_hold | 新浪财经-机构持股-机构持股一览表 |
| 基本面数据-机构持股 | stock_institute_hold_detail | 新浪财经-机构持股-机构持股详情 |
| 基本面数据-机构推荐 | stock_institute_recommend | 新浪财经-机构推荐池-具体指标的数据 |
| 基本面数据-机构推荐 | stock_institute_recommend_detail | 新浪财经-机构推荐池-股票评级记录 |
| 基本面数据-机构推荐 | stock_rank_forecast_cninfo | 巨潮资讯-数据中心-评级预测-投资评级 |
| 基本面数据-机构推荐 | stock_industry_clf_hist_sw | 申万宏源研究-行业分类-全部行业分类 |
| 基本面数据-机构推荐 | stock_industry_pe_ratio_cninfo | 巨潮资讯-数据中心-行业分析-行业市盈率 |
| 基本面数据-机构推荐 | stock_new_gh_cninfo | 巨潮资讯-数据中心-新股数据-新股过会 |
| 基本面数据-机构推荐 | stock_new_ipo_cninfo | 巨潮资讯-数据中心-新股数据-新股发行 |
| 基本面数据-机构推荐 | stock_share_hold_change_sse | 上海证券交易所-披露-监管信息公开-公司监管-董董监高人员股份变动 |
| 基本面数据-机构推荐 | stock_share_hold_change_szse | 深圳证券交易所-信息披露-监管信息公开-董监高人员股份变动 |
| 基本面数据-机构推荐 | stock_share_hold_change_bse | 北京证券交易所-信息披露-监管信息-董监高及相关人员持股变动 |
| 基本面数据-机构推荐 | stock_hold_num_cninfo | 巨潮资讯-数据中心-专题统计-股东股本-股东人数及持股集中度 |
| 基本面数据-机构推荐 | stock_hold_change_cninfo | 巨潮资讯-数据中心-专题统计-股东股本-股本变动 |
| 基本面数据-机构推荐 | stock_hold_control_cninfo | 巨潮资讯-数据中心-专题统计-股东股本-实际控制人持股变动 |
| 基本面数据-机构推荐 | stock_hold_management_detail_cninfo | 巨潮资讯-数据中心-专题统计-股东股本-高管持股变动明细 |
| 基本面数据-机构推荐 | stock_hold_management_detail_em | 东方财富网-数据中心-特色数据-高管持股-董监高及相关人员持股变动明细 |
| 基本面数据-机构推荐 | stock_hold_management_person_em | 东方财富网-数据中心-特色数据-高管持股-人员增减持股变动明细 |
| 基本面数据-机构推荐 | stock_cg_guarantee_cninfo | 巨潮资讯-数据中心-专题统计-公司治理-对外担保 |
| 基本面数据-机构推荐 | stock_cg_lawsuit_cninfo | 巨潮资讯-数据中心-专题统计-公司治理-公司诉讼 |
| 基本面数据-机构推荐 | stock_cg_equity_mortgage_cninfo | 巨潮资讯-数据中心-专题统计-公司治理-股权质押 |
| 基本面数据-美港目标价 | stock_price_js | 美港电讯-美港目标价数据 |
| 基本面数据-券商业绩月报 | stock_qsjy_em | 东方财富网-数据中心-特色数据-券商业绩月报 |
| 基本面数据-A 股个股指标 | stock_a_indicator_lg | 乐咕乐股-A 股个股指标: 市盈率, 市净率, 股息率 |
| 基本面数据-A 股股息率 | stock_a_gxl_lg | 乐咕乐股-股息率-A 股股息率 |
| 基本面数据-恒生指数股息率 | stock_hk_gxl_lg | 乐咕乐股-股息率-恒生指数股息率 |
| 基本面数据-大盘拥挤度 | stock_a_congestion_lg | 乐咕乐股-大盘拥挤度 |
| 基本面数据-股债利差 | stock_ebs_lg | 乐咕乐股-股债利差 |
| 基本面数据-巴菲特指标 | stock_buffett_index_lg | 乐估乐股-底部研究-巴菲特指标 |
| 基本面数据-A 股等权重与中位数市盈率 | stock_a_ttm_lyr | 乐咕乐股-A 股等权重市盈率与中位数市盈率 |
| 基本面数据-A 股等权重与中位数市净率 | stock_a_all_pb | 乐咕乐股-A 股等权重与中位数市净率 |
| 基本面数据-主板市盈率 | stock_market_pe_lg | 乐咕乐股-主板市盈率 |
| 基本面数据-指数市盈率 | stock_index_pe_lg | 乐咕乐股-指数市盈率 |
| 基本面数据-主板市净率 | stock_market_pb_lg | 乐咕乐股-主板市净率 |
| 基本面数据-指数市净率 | stock_index_pb_lg | 乐咕乐股-指数市净率 |
| 基本面数据-A 股估值指标 | stock_zh_valuation_baidu | 百度股市通-A 股-财务报表-估值数据 |
| 基本面数据-个股估值 | stock_value_em | 东方财富网-数据中心-估值分析-每日互动-每日互动-估值分析 |
| 基本面数据-涨跌投票 | stock_zh_vote_baidu | 百度股市通- A 股或指数-股评-投票 |
| 基本面数据-港股个股指标 | stock_hk_indicator_eniu | 亿牛网-港股个股指标: 市盈率, 市净率, 股息率, ROE, 市值 |
| 基本面数据-港股估值指标 | stock_hk_valuation_baidu | 百度股市通-港股-财务报表-估值数据 |
| 基本面数据-创新高和新低的股票数量 | stock_a_high_low_statistics | 不同市场的创新高和新低的股票数量 |
| 基本面数据-破净股统计 | stock_a_below_net_asset_statistics | 乐咕乐股-A 股破净股统计数据 |
| 基本面数据-基金持股 | stock_report_fund_hold | 东方财富网-数据中心-主力数据-基金持仓 |
| 基本面数据-基金持股明细 | stock_report_fund_hold_detail | 东方财富网-数据中心-主力数据-基金持仓-基金持仓明细表 |
| 基本面数据-龙虎榜 | stock_lhb_detail_em | 东方财富网-数据中心-龙虎榜单-龙虎榜详情 |
| 基本面数据-龙虎榜 | stock_lhb_stock_statistic_em | 东方财富网-数据中心-龙虎榜单-个股上榜统计 |
| 基本面数据-龙虎榜 | stock_lhb_jgmmtj_em | 东方财富网-数据中心-龙虎榜单-机构买卖每日统计 |
| 基本面数据-龙虎榜 | stock_lhb_jgstatistic_em | 东方财富网-数据中心-龙虎榜单-机构席位追踪 |
| 基本面数据-龙虎榜 | stock_lhb_hyyyb_em | 东方财富网-数据中心-龙虎榜单-每日活跃营业部 |
| 基本面数据-龙虎榜 | stock_lhb_yybph_em | 东方财富网-数据中心-龙虎榜单-营业部排行 |
| 基本面数据-龙虎榜 | stock_lhb_traderstatistic_em | 东方财富网-数据中心-龙虎榜单-营业部统计 |
| 基本面数据-龙虎榜 | stock_lhb_stock_detail_em | 东方财富网-数据中心-龙虎榜单-个股龙虎榜详情 |
| 基本面数据-龙虎榜 | stock_lh_yyb_most | 龙虎榜-营业部排行-上榜次数最多 |
| 基本面数据-龙虎榜 | stock_lh_yyb_capital | 龙虎榜-营业部排行-资金实力最强 |
| 基本面数据-龙虎榜 | stock_lh_yyb_control | 龙虎榜-营业部排行-抱团操作实力 |
| 基本面数据-龙虎榜 | stock_lhb_detail_daily_sina | 新浪财经-龙虎榜-每日详情 |
| 基本面数据-龙虎榜 | stock_lhb_ggtj_sina | 新浪财经-龙虎榜-个股上榜统计 |
| 基本面数据-龙虎榜 | stock_lhb_yytj_sina | 新浪财经-龙虎榜-营业上榜统计 |
| 基本面数据-龙虎榜 | stock_lhb_jgzz_sina | 新浪财经-龙虎榜-机构席位追踪 |
| 基本面数据-龙虎榜 | stock_lhb_jgmx_sina | 新浪财经-龙虎榜-机构席位成交明细 |
| 基本面数据-首发申报信息 | stock_ipo_declare | 东方财富网-数据中心-新股申购-首发申报信息-首发申报企业信息 |
| 基本面数据-IPO审核信息 | stock_register_kcb | 东方财富网-数据中心-新股数据-IPO审核信息-科创板 |
| 基本面数据-IPO审核信息 | stock_register_cyb | 东方财富网-数据中心-新股数据-IPO审核信息-创业板 |
| 基本面数据-IPO审核信息 | stock_register_sh | 东方财富网-数据中心-新股数据-IPO审核信息-上海主板 |
| 基本面数据-IPO审核信息 | stock_register_sz | 东方财富网-数据中心-新股数据-IPO审核信息-深圳主板 |
| 基本面数据-IPO审核信息 | stock_register_bj | 东方财富网-数据中心-新股数据-IPO审核信息-北交所 |
| 基本面数据-IPO审核信息 | stock_register_db | 东方财富网-数据中心-新股数据-注册制审核-达标企业 |
| 基本面数据-增发 | stock_qbzf_em | 东方财富网-数据中心-新股数据-增发-全部增发 |
| 基本面数据-配股 | stock_pg_em | 东方财富网-数据中心-新股数据-配股 |
| 基本面数据-股票回购数据 | stock_repurchase_em | 东方财富网-数据中心-股票回购-股票回购数据 |

## 大宗交易

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 大宗交易-市场统计 | stock_dzjy_sctj | 东方财富网-数据中心-大宗交易-市场统计 |
| 大宗交易-每日明细 | stock_dzjy_mrmx | 东方财富网-数据中心-大宗交易-每日明细 |
| 大宗交易-每日统计 | stock_dzjy_mrtj | 东方财富网-数据中心-大宗交易-每日统计 |
| 大宗交易-活跃 A 股统计 | stock_dzjy_hygtj | 东方财富网-数据中心-大宗交易-活跃 A 股统计 |
| 大宗交易-活跃营业部统计 | stock_dzjy_hyyybtj | 东方财富网-数据中心-大宗交易-活跃营业部统计 |
| 大宗交易-营业部排行 | stock_dzjy_yybph | 东方财富网-数据中心-大宗交易-营业部排行 |

## 年报季报

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 年报季报-业绩报表 | stock_yjbb_em | 东方财富-数据中心-年报季报-业绩报表 |
| 年报季报-业绩快报 | stock_yjkb_em | 东方财富-数据中心-年报季报-业绩快报 |
| 年报季报-业绩预告 | stock_yjyg_em | 东方财富-数据中心-年报季报-业绩预告 |
| 年报季报-预约披露时间-东方财富 | stock_yysj_em | 东方财富-数据中心-年报季报-预约披露时间 |
| 年报季报-预约披露时间-巨潮资讯 | stock_report_disclosure | 巨潮资讯-数据-预约披露的数据 |
| 年报季报-信息披露公告-巨潮资讯 | stock_zh_a_disclosure_report_cninfo | 巨潮资讯-首页-公告查询-信息披露公告-沪深京 |
| 年报季报-信息披露调研-巨潮资讯 | stock_zh_a_disclosure_relation_cninfo | 巨潮资讯-首页-公告查询-信息披露调研-沪深京 |
| 年报季报-行业分类数据-巨潮资讯 | stock_industry_category_cninfo | 巨潮资讯-数据-行业分类数据 |
| 年报季报-上市公司行业归属的变动情况-巨潮资讯 | stock_industry_change_cninfo | 巨潮资讯-数据-上市公司行业归属的变动情况 |
| 年报季报-公司股本变动-巨潮资讯 | stock_share_change_cninfo | 巨潮资讯-数据-公司股本变动 |
| 年报季报-配股实施方案-巨潮资讯 | stock_allotment_cninfo | 巨潮资讯-个股-配股实施方案 |
| 年报季报-公司概况-巨潮资讯 | stock_profile_cninfo | 巨潮资讯-个股-公司概况 |
| 年报季报-上市相关-巨潮资讯 | stock_ipo_summary_cninfo | 巨潮资讯-个股-上市相关 |
| 年报季报-资产负债表-沪深 | stock_zcfz_em | 东方财富-数据中心-年报季报-业绩快报-资产负债表 |
| 年报季报-资产负债表-北交所 | stock_zcfz_bj_em | 东方财富-数据中心-年报季报-业绩快报-资产负债表 |
| 年报季报-利润表 | stock_lrb_em | 东方财富-数据中心-年报季报-业绩快报-利润表 |
| 年报季报-现金流量表 | stock_xjll_em | 东方财富-数据中心-年报季报-业绩快报-现金流量表 |

## 技术指标

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 技术指标-持续放量 | stock_rank_cxfl_ths | 同花顺-数据中心-技术选股-持续放量 |
| 技术指标-持续缩量 | stock_rank_cxsl_ths | 同花顺-数据中心-技术选股-持续缩量 |
| 技术指标-向上突破 | stock_rank_xstp_ths | 同花顺-数据中心-技术选股-向上突破 |
| 技术指标-向下突破 | stock_rank_xxtp_ths | 同花顺-数据中心-技术选股-向下突破 |
| 技术指标-量价齐升 | stock_rank_ljqs_ths | 同花顺-数据中心-技术选股-量价齐升 |
| 技术指标-量价齐跌 | stock_rank_ljqd_ths | 同花顺-数据中心-技术选股-量价齐跌 |
| 技术指标-险资举牌 | stock_rank_xzjp_ths | 同花顺-数据中心-技术选股-险资举牌 |

## 新股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 新股-公司动态 | stock_zh_a_new_em | 东方财富网-行情中心-沪深个股-新股 |

## 新股上市首日

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 新股上市首日-公司动态 | stock_xgsr_ths | 同花顺-数据中心-新股数据-新股上市首日 |

## 新股数据

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 新股数据-打新收益率 | stock_dxsyl_em | 东方财富网-数据中心-新股申购-打新收益率 |
| 新股数据-新股申购与中签 | stock_xgsglb_em | 东方财富网-数据中心-新股数据-新股申购-新股申购与中签查询 |

## 机构调研

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 机构调研-机构调研-统计 | stock_jgdy_tj_em | 东方财富网-数据中心-特色数据-机构调研-机构调研统计 |
| 机构调研-机构调研-详细 | stock_jgdy_detail_em | 东方财富网-数据中心-特色数据-机构调研-机构调研详细 |

## 板块异动详情

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 板块异动详情-相关股票 | stock_board_change_em | 东方财富-行情中心-当日板块异动详情 |

## 概念板块

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 概念板块-同花顺-概念板块指数 | stock_board_concept_index_ths | 同花顺-板块-概念板块-指数日频率数据 |
| 概念板块-同花顺-概念板块简介 | stock_board_concept_info_ths | 同花顺-板块-概念板块-板块简介 |
| 概念板块-东方财富-概念板块 | stock_board_concept_name_em | 东方财富网-行情中心-沪深京板块-概念板块 |
| 概念板块-东方财富-概念板块-实时行情 | stock_board_concept_spot_em | 东方财富网-行情中心-沪深京板块-概念板块-实时行情 |
| 概念板块-东方财富-成份股 | stock_board_concept_cons_em | 东方财富-沪深板块-概念板块-板块成份 |
| 概念板块-东方财富-指数 | stock_board_concept_hist_em | 东方财富-沪深板块-概念板块-历史行情数据 |
| 概念板块-东方财富-指数-分时 | stock_board_concept_hist_min_em | 东方财富-沪深板块-概念板块-分时历史行情数据 |
| 概念板块-富途牛牛-美股概念-成分股 | stock_concept_cons_futu | 富途牛牛-主题投资-概念板块-成分股 |

## 次新股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 次新股-历史行情数据 | stock_zh_a_new | 新浪财经-行情中心-沪深股市-次新股 |

## 沪深港通持股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 沪深港通持股-结算汇率-深港通 | stock_sgt_settlement_exchange_rate_szse | 深港通-港股通业务信息-结算汇率 |
| 沪深港通持股-结算汇率-沪港通 | stock_sgt_settlement_exchange_rate_sse | 沪港通-港股通信息披露-结算汇兑 |
| 沪深港通持股-参考汇率-深港通 | stock_sgt_reference_exchange_rate_szse | 深港通-港股通业务信息-参考汇率 |
| 沪深港通持股-参考汇率-沪港通 | stock_sgt_reference_exchange_rate_sse | 沪港通-港股通信息披露-参考汇率 |
| 沪深港通持股-港股通成份股 | stock_hk_ggt_components_em | 东方财富网-行情中心-港股市场-港股通成份股 |
| 沪深港通持股-沪深港通分时数据 | stock_hsgt_fund_min_em | 东方财富-数据中心-沪深港通-市场概括-分时数据 |
| 沪深港通持股-板块排行 | stock_hsgt_board_rank_em | 东方财富网-数据中心-沪深港通持股-板块排行 |
| 沪深港通持股-个股排行 | stock_hsgt_hold_stock_em | 东方财富网-数据中心-沪深港通持股-个股排行 |
| 沪深港通持股-每日个股统计 | stock_hsgt_stock_statistics_em | 东方财富网-数据中心-沪深港通-沪深港通持股-每日个股统计 |
| 沪深港通持股-机构排行 | stock_hsgt_institution_statistics_em | 东方财富网-数据中心-沪深港通-沪深港通持股-机构排行 |
| 沪深港通持股-沪深港通-港股通(沪>港)实时行情 | stock_hsgt_sh_hk_spot_em | 东方财富网-行情中心-沪深港通-港股通(沪>港)-股票；按股票代码排序 |
| 沪深港通持股-沪深港通历史数据 | stock_hsgt_hist_em | 东方财富网-数据中心-资金流向-沪深港通资金流向-沪深港通历史数据 |
| 沪深港通持股-沪深港通持股-个股 | stock_hsgt_individual_em | 东方财富网-数据中心-沪深港通-沪深港通持股-具体股票 |
| 沪深港通持股-沪深港通持股-个股详情 | stock_hsgt_individual_detail_em | 东方财富网-数据中心-沪深港通-沪深港通持股-具体股票-个股详情 |

## 沪深港通资金流向

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 沪深港通资金流向-市场热度 | stock_hsgt_fund_flow_summary_em | 东方财富网-数据中心-资金流向-沪深港通资金流向 |

## 涨停板行情

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 涨停板行情-涨停股池 | stock_zt_pool_em | 东方财富网-行情中心-涨停板行情-涨停股池 |
| 涨停板行情-昨日涨停股池 | stock_zt_pool_previous_em | 东方财富网-行情中心-涨停板行情-昨日涨停股池 |
| 涨停板行情-强势股池 | stock_zt_pool_strong_em | 东方财富网-行情中心-涨停板行情-强势股池 |
| 涨停板行情-次新股池 | stock_zt_pool_sub_new_em | 东方财富网-行情中心-涨停板行情-次新股池 |
| 涨停板行情-炸板股池 | stock_zt_pool_zbgc_em | 东方财富网-行情中心-涨停板行情-炸板股池 |
| 涨停板行情-跌停股池 | stock_zt_pool_dtgc_em | 东方财富网-行情中心-涨停板行情-跌停股池 |

## 港股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 港股-实时行情数据-东财 | stock_hk_spot_em | 所有港股的实时行情数据; 该数据有 15 分钟延时 |
| 港股-港股主板实时行情数据-东财 | stock_hk_main_board_spot_em | 港股主板的实时行情数据; 该数据有 15 分钟延时 |
| 港股-实时行情数据-新浪 | stock_hk_spot | 获取所有港股的实时行情数据 15 分钟延时 |
| 港股-个股信息查询-雪球 | stock_individual_basic_info_hk_xq | 雪球-个股-公司概况-公司简介 |
| 港股-分时数据-东财 | stock_hk_hist_min_em | 东方财富网-行情首页-港股-每日分时行情 |
| 港股-历史行情数据-东财 | stock_hk_hist | 港股-历史行情数据, 可以选择返回复权后数据, 更新频率为日频 |
| 港股-历史行情数据-新浪 | stock_hk_daily |  |
| 港股-知名港股 | stock_hk_famous_spot_em | 东方财富网-行情中心-港股市场-知名港股实时行情数据 |

## 港股盈利预测

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 港股盈利预测-经济通-深圳证券交易所 | stock_hk_profit_forecast_et | 经济通-公司资料-盈利预测 |

## 盈利预测

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 盈利预测-东方财富-深圳证券交易所 | stock_profit_forecast_em | 东方财富网-数据中心-研究报告-盈利预测; 该数据源网页端返回数据有异常, 本接口已修复该异常 |
| 盈利预测-同花顺-深圳证券交易所 | stock_profit_forecast_ths | 同花顺-盈利预测 |

## 盘口异动

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 盘口异动-相关股票 | stock_changes_em | 东方财富-行情中心-盘口异动数据 |

## 科创板

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 科创板-实时行情数据 | stock_zh_kcb_spot | 新浪财经-科创板股票实时行情数据 |
| 科创板-历史行情数据 | stock_zh_kcb_daily | 新浪财经-科创板股票历史行情数据 |
| 科创板-科创板公告 | stock_zh_kcb_report_em | 东方财富-科创板报告数据 |

## 筹码分布

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 筹码分布-东方财富 | stock_cyq_em | 东方财富网-概念板-行情中心-日K-筹码分布 |

## 管理层讨论与分析

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 管理层讨论与分析-机构调研-详细 | stock_mda_ym | 益盟-F10-管理层讨论与分析 |

## 美股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 美股-实时行情数据-东财 | stock_us_spot_em | 东方财富网-美股-实时行情 |
| 美股-实时行情数据-新浪 | stock_us_spot | 新浪财经-美股; 获取的数据有 15 分钟延迟; 建议使用 ak.stock_us_spot_em() 来获取数据 |
| 美股-历史行情数据-东财 | stock_us_hist | 东方财富网-行情-美股-每日行情 |
| 美股-个股信息查询-雪球 | stock_individual_basic_info_us_xq | 雪球-个股-公司概况-公司简介 |
| 美股-分时数据-东财 | stock_us_hist_min_em | 东方财富网-行情首页-美股-每日分时行情 |
| 美股-历史行情数据-新浪 | stock_us_daily | 美股历史行情数据，设定 adjust="qfq" 则返回前复权后的数据，默认 adjust="", 则返回未复权的数据，历史数据按日频率更新 |
| 美股-粉单市场 | stock_us_pink_spot_em | 美股粉单市场的实时行情数据 |
| 美股-知名美股 | stock_us_famous_spot_em | 美股-知名美股的实时行情数据 |

## 股市日历

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 股市日历-公司动态 | stock_gsrl_gsdt_em | 东方财富网-数据中心-股市日历-公司动态 |

## 股票热度

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 股票热度-股票热度-雪球 | stock_hot_follow_xq | 雪球-沪深股市-热度排行榜-关注排行榜 |
| 股票热度-股票热度-雪球 | stock_hot_tweet_xq | 雪球-沪深股市-热度排行榜-讨论排行榜 |
| 股票热度-股票热度-雪球 | stock_hot_deal_xq | 雪球-沪深股市-热度排行榜-交易排行榜 |
| 股票热度-股票热度-问财 | stock_hot_rank_wc | 问财-热门股票排名数据; 请注意访问的频率 |
| 股票热度-股票热度-东财 | stock_hot_rank_em | 东方财富网站-股票热度 |
| 股票热度-股票热度-东财 | stock_hot_up_em | 东方财富-个股人气榜-飙升榜 |
| 股票热度-股票热度-东财 | stock_hk_hot_rank_em | 东方财富-个股人气榜-人气榜-港股市场 |
| 股票热度-历史趋势及粉丝特征 | stock_hot_rank_detail_em | 东方财富网-股票热度-历史趋势及粉丝特征 |
| 股票热度-历史趋势及粉丝特征 | stock_hk_hot_rank_detail_em | 东方财富网-股票热度-历史趋势 |
| 股票热度-互动平台 | stock_irm_cninfo | 互动易-提问 |
| 股票热度-互动平台 | stock_irm_ans_cninfo | 互动易-回答 |
| 股票热度-互动平台 | stock_sns_sseinfo | 上证e互动-提问与回答 |
| 股票热度-个股人气榜-实时变动 | stock_hot_rank_detail_realtime_em | 东方财富网-个股人气榜-实时变动 |
| 股票热度-个股人气榜-实时变动 | stock_hk_hot_rank_detail_realtime_em | 东方财富网-个股人气榜-实时变动 |
| 股票热度-热门关键词 | stock_hot_keyword_em | 东方财富-个股人气榜-热门关键词 |
| 股票热度-内部交易 | stock_inner_trade_xq | 雪球-行情中心-沪深股市-内部交易 |
| 股票热度-个股人气榜-最新排名 | stock_hot_rank_latest_em | 东方财富-个股人气榜-最新排名 |
| 股票热度-个股人气榜-最新排名 | stock_hk_hot_rank_latest_em | 东方财富-个股人气榜-最新排名 |
| 股票热度-热搜股票 | stock_hot_search_baidu | 百度股市通-热搜股票 |
| 股票热度-相关股票 | stock_hot_rank_relate_em | 东方财富-个股人气榜-相关股票 |

## 股票账户统计

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 股票账户统计-股票账户统计月度 | stock_account_statistics_em | 东方财富网-数据中心-特色数据-股票账户统计 |

## 股票质押

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 股票质押-股权质押市场概况 | stock_gpzy_profile_em | 东方财富网-数据中心-特色数据-股权质押-股权质押市场概况 |
| 股票质押-上市公司质押比例 | stock_gpzy_pledge_ratio_em | 东方财富网-数据中心-特色数据-股权质押-上市公司质押比例 |
| 股票质押-重要股东股权质押明细 | stock_gpzy_pledge_ratio_detail_em | 东方财富网-数据中心-特色数据-股权质押-重要股东股权质押明细 |
| 股票质押-质押机构分布统计-证券公司 | stock_gpzy_distribute_statistics_company_em | 东方财富网-数据中心-特色数据-股权质押-质押机构分布统计-证券公司 |
| 股票质押-质押机构分布统计-银行 | stock_gpzy_distribute_statistics_bank_em | 东方财富网-数据中心-特色数据-股权质押-质押机构分布统计-银行 |
| 股票质押-上市公司质押比例 | stock_gpzy_industry_data_em | 东方财富网-数据中心-特色数据-股权质押-上市公司质押比例-行业数据 |

## 融资融券

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 融资融券-标的证券名单及保证金比例查询 | stock_margin_ratio_pa | 融资融券-标的证券名单及保证金比例查询 |
| 融资融券-两融账户信息 | stock_margin_account_info | 东方财富网-数据中心-融资融券-融资融券账户统计-两融账户信息 |
| 融资融券-上海证券交易所 | stock_margin_sse | 上海证券交易所-融资融券数据-融资融券汇总数据 |
| 融资融券-上海证券交易所 | stock_margin_detail_sse | 上海证券交易所-融资融券数据-融资融券明细数据 |
| 融资融券-深圳证券交易所 | stock_margin_szse | 深圳证券交易所-融资融券数据-融资融券汇总数据 |
| 融资融券-深圳证券交易所 | stock_margin_detail_szse | 深证证券交易所-融资融券数据-融资融券交易明细数据 |
| 融资融券-深圳证券交易所 | stock_margin_underlying_info_szse | 深圳证券交易所-融资融券数据-标的证券信息 |

## 行业板块

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 行业板块-同花顺-同花顺行业一览表 | stock_board_industry_summary_ths | 同花顺-同花顺行业一览表 |
| 行业板块-同花顺-指数 | stock_board_industry_index_ths | 同花顺-板块-行业板块-指数日频率数据 |
| 行业板块-东方财富-行业板块 | stock_board_industry_name_em | 东方财富-沪深京板块-行业板块 |
| 行业板块-东方财富-行业板块-实时行情 | stock_board_industry_spot_em | 东方财富网-沪深板块-行业板块-实时行情 |
| 行业板块-东方财富-成份股 | stock_board_industry_cons_em | 东方财富-沪深板块-行业板块-板块成份 |
| 行业板块-东方财富-指数-日频 | stock_board_industry_hist_em | 东方财富-沪深板块-行业板块-历史行情数据 |
| 行业板块-东方财富-指数-分时 | stock_board_industry_hist_min_em | 东方财富-沪深板块-行业板块-分时历史行情数据 |

## 财报发行

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 财报发行-沪深港通持股-个股详情 | news_report_time_baidu | 百度股市通-财报发行 |

## 财经内容精选

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 财经内容精选-沪深港通持股-个股详情 | stock_news_main_cx | 财新网-财新数据通-内容精选 |

## 资金流向

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 资金流向-同花顺 | stock_fund_flow_individual | 同花顺-数据中心-资金流向-个股资金流 |
| 资金流向-同花顺 | stock_fund_flow_concept | 同花顺-数据中心-资金流向-概念资金流 |
| 资金流向-同花顺 | stock_fund_flow_industry | 同花顺-数据中心-资金流向-行业资金流 |
| 资金流向-同花顺 | stock_fund_flow_big_deal | 同花顺-数据中心-资金流向-大单追踪 |
| 资金流向-东方财富 | stock_individual_fund_flow | 东方财富网-数据中心-个股资金流向 |
| 资金流向-东方财富 | stock_individual_fund_flow_rank | 东方财富网-数据中心-资金流向-排名 |
| 资金流向-东方财富 | stock_market_fund_flow | 东方财富网-数据中心-资金流向-大盘 |
| 资金流向-东方财富 | stock_sector_fund_flow_rank | 东方财富网-数据中心-资金流向-板块资金流-排名 |
| 资金流向-东方财富 | stock_main_fund_flow | 东方财富网-数据中心-资金流向-主力净流入排名 |
| 资金流向-东方财富 | stock_sector_fund_flow_summary | 东方财富网-数据中心-资金流向-行业资金流-xx行业个股资金流 |
| 资金流向-东方财富 | stock_sector_fund_flow_hist | 东方财富网-数据中心-资金流向-行业资金流-行业历史资金流 |
| 资金流向-东方财富 | stock_concept_fund_flow_hist | 东方财富网-数据中心-资金流向-概念资金流-概念历史资金流 |

## 赚钱效应分析

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 赚钱效应分析-跌停股池 | stock_market_activity_legu | 乐咕乐股网-赚钱效应分析数据 |

## 风险警示板

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 风险警示板-公司动态 | stock_zh_a_st_em | 东方财富网-行情中心-沪深个股-风险警示板 |

## 高管持股

| 分类 | 接口名 | 描述 |
|------|--------|------|
| 高管持股-股东增减持 | stock_ggcg_em | 东方财富网-数据中心-特色数据-高管持股 |

