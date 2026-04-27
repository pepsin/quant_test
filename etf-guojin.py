#coding:gbk
"""
ETF轮动策略 - 国金QMT版本
基于原聚宽策略转换

【QMT使用说明】
1. 在QMT中新建Python策略，将本代码粘贴进去
2. 策略周期选择"1分钟"（实盘盯盘用），回测时同样选1分钟周期
3. 回测参数：在QMT回测面板设置初始资金、手续费率、滑点
4. 实盘前务必先在模拟盘测试
5. 如使用miniQMT，需要在外部Python环境中安装xtquant，并做额外适配

【与聚宽版本的主要差异】
- 股票代码格式: .XSHG/.XSHE 改为 .SH/.SZ
- 数据接口: 使用 ContextInfo.get_market_data / get_full_tick
- 交易接口: 使用 order_target_value（回测）或 passorder（实盘）
- 持仓查询: 使用 get_trade_detail_data
- 定时调度: handlebar中配合时间判断（1分钟周期驱动）
- 移除溢价率过滤（QMT不支持基金净值查询）
- 成交量过滤简化为日线级别估算
- 日志输出: 使用print()（QMT内置策略控制台可见）
"""

import math
import datetime
import time
import numpy as np
import pandas as pd


# ======================== 代码格式转换 ========================
def jq_to_qmt(code):
    """聚宽代码转QMT代码"""
    return code.replace('.XSHG', '.SH').replace('.XSHE', '.SZ')


def qmt_to_jq(code):
    """QMT代码转聚宽代码"""
    return code.replace('.SH', '.XSHG').replace('.SZ', '.XSHE')


# ======================== QMT数据适配层 ========================
def attribute_history(C, security, count, period, fields):
    """
    模拟聚宽 attribute_history
    返回: DataFrame(index=时间字符串, columns=fields)
    兼容QMT不同版本的返回格式
    """
    try:
        data = C.get_market_data(fields, security, period=period, count=count)
        if data is None or len(data) == 0:
            return pd.DataFrame()

        # 格式1: 某些版本直接返回 DataFrame(index=times, columns=fields)
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            # 确保包含所需字段
            for f in fields:
                if f not in df.columns:
                    df[f] = None
            df = df.sort_index()
            return df

        # 格式2: 标准dict格式 {field: DataFrame(index=[code], columns=[times])}
        if isinstance(data, dict):
            times = None
            for f in fields:
                if f in data and data[f] is not None and len(data[f]) > 0:
                    times = list(data[f].columns)
                    break
            if not times:
                return pd.DataFrame()

            rows = []
            for t in times:
                row = {}
                for f in fields:
                    if f in data and data[f] is not None:
                        row[f] = data[f].iloc[0][t]
                    else:
                        row[f] = None
                rows.append(row)

            df = pd.DataFrame(rows, index=times)
            df = df.sort_index()
            return df

        return pd.DataFrame()
    except Exception as e:
        print(f"[attribute_history] 失败 {security}: {e}")
        return pd.DataFrame()


def get_price(C, security, start_date, end_date, frequency, fields):
    """
    模拟聚宽 get_price
    start_date/end_date 格式: '20230101' 或 '20230101103000'，空字符串表示不限
    """
    try:
        data = C.get_market_data(fields, security, period=frequency,
                                  start_time=start_date, end_time=end_date)
        if data is None or len(data) == 0:
            return pd.DataFrame()

        # 格式1: 某些版本直接返回 DataFrame
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            for f in fields:
                if f not in df.columns:
                    df[f] = None
            df = df.sort_index()
            return df

        # 格式2: 标准dict格式
        if isinstance(data, dict):
            times = None
            for f in fields:
                if f in data and data[f] is not None and len(data[f]) > 0:
                    times = list(data[f].columns)
                    break

            if not times:
                return pd.DataFrame()

            rows = []
            for t in times:
                row = {}
                for f in fields:
                    if f in data and data[f] is not None:
                        row[f] = data[f].iloc[0][t]
                    else:
                        row[f] = None
                rows.append(row)

            df = pd.DataFrame(rows, index=times)
            df = df.sort_index()
            return df

        return pd.DataFrame()
    except Exception as e:
        print(f"[get_price] 失败 {security}: {e}")
        return pd.DataFrame()


def get_current_data_price(C, security):
    """获取当前最新价格
    优先用K线收盘价（回测安全），再用get_full_tick（实盘）
    """
    # 方案1: K线收盘价（回测中安全，不会引入未来函数）
    try:
        data = C.get_market_data(['close'], security, period='1d', count=1)
        if data and 'close' in data and len(data['close']) > 0:
            price = float(data['close'].iloc[0].iloc[-1])
            if price > 0:
                return price
    except Exception:
        pass

    # 方案2: 获取当前周期的最新价（适用于分钟级别回测/实盘）
    try:
        # 尝试获取当前周期的close（如1分钟周期的当前K线收盘价）
        data = C.get_market_data(['close'], security, count=1)
        if data and 'close' in data and len(data['close']) > 0:
            price = float(data['close'].iloc[0].iloc[-1])
            if price > 0:
                return price
    except Exception:
        pass

    # 方案3: 全推tick（实盘中最新价，回测中可能不可用或返回最新快照）
    try:
        tick = C.get_full_tick([security])
        if tick and security in tick:
            price = tick[security].get('lastPrice', 0)
            if price and price > 0:
                return float(price)
    except Exception:
        pass
    return 0.0


def get_stock_name(C, security):
    """获取证券名称"""
    try:
        return C.get_stock_name(security)
    except Exception:
        return security


def get_trade_days_list(C, end_date_str, count):
    """
    获取交易日列表（往前count个交易日，包含end_date当天）
    end_date_str 格式: '20230101'
    返回 datetime.date 列表
    """
    try:
        dates = C.get_trading_dates('510300.SH', '', end_date_str, count, '1d')
        result = []
        for d in dates:
            d_str = str(d).strip()
            # 兼容多种格式: '20230101', '2023-01-01', datetime对象等
            if len(d_str) >= 8 and d_str[4] == '-' and d_str[7] == '-':
                result.append(datetime.datetime.strptime(d_str[:10], '%Y-%m-%d').date())
            elif len(d_str) == 8 and d_str.isdigit():
                result.append(datetime.datetime.strptime(d_str, '%Y%m%d').date())
            else:
                # 尝试通用解析
                result.append(pd.to_datetime(d_str).date())
        return result
    except Exception as e:
        print(f"[get_trade_days_list] 失败: {e}")
        return []


# ======================== QMT交易适配层 ========================
def get_positions(C):
    """
    获取持仓字典
    返回: {code: {'volume':总数量, 'available':可用数量, 'cost':成本价, 'market_value':市值}}
    """
    if not hasattr(C, 'accountid') or not C.accountid:
        return {}
    try:
        pos_list = get_trade_detail_data(C.accountid, 'stock', 'position')
        positions = {}
        for pos in pos_list:
            code = pos.m_strInstrumentID + '.' + pos.m_strExchangeID
            positions[code] = {
                'volume': int(pos.m_nVolume),
                'available': int(pos.m_nCanUseVolume),
                'cost': float(pos.m_dOpenPrice),
                'market_value': float(pos.m_dMarketValue),
            }
        return positions
    except Exception as e:
        print(f"[get_positions] 失败: {e}")
        return {}


def get_account_info(C):
    """获取账户信息，返回可用资金、总资产等"""
    if not hasattr(C, 'accountid') or not C.accountid:
        return None
    try:
        acc_list = get_trade_detail_data(C.accountid, 'stock', 'account')
        if acc_list and len(acc_list) > 0:
            acc = acc_list[0]
            return {
                'available_cash': float(acc.m_dAvailable),
                'total_asset': float(acc.m_dBalance),
            }
    except Exception as e:
        print(f"[get_account_info] 失败: {e}")
    return None


def smart_order_target_value(C, security, value):
    """
    下单调整至目标金额
    回测优先用order_target_value（最可靠），实盘不支持时fallback到passorder
    value=0 表示清仓
    """
    # 方案1: order_target_value（回测支持，部分QMT实盘版本也支持）
    try:
        if value <= 0:
            order_target_value(security, 0, C)
        else:
            order_target_value(security, value, C)
        return True
    except Exception as e1:
        # 方案2: passorder（通用实盘下单）
        # 注意：在handlebar中调用passorder，quickTrade建议用1（只在最新K线下单）
        # 如果在定时器(run_time)回调中调用，需要用2（强制立即下单）
        try:
            price = get_current_data_price(C, security)
            if price <= 0:
                print(f"[下单失败] {security} 无法获取价格")
                return False

            positions = get_positions(C)
            account = get_account_info(C)
            current_volume = positions.get(security, {}).get('volume', 0)
            current_value = positions.get(security, {}).get('market_value', 0)

            if value <= 0:
                if current_volume > 0:
                    passorder(24, 1101, C.accountid, security, 5, -1, current_volume, 'ETF轮动', 1, C)
                return True

            diff_value = value - current_value
            if abs(diff_value) < C.min_money:
                return True

            if diff_value > 0:
                if account is None or diff_value > account['available_cash']:
                    print(f"[下单跳过] {security} 资金不足")
                    return False
                vol = int(diff_value / price / 100) * 100
                if vol <= 0:
                    return True
                passorder(23, 1101, C.accountid, security, 5, -1, vol, 'ETF轮动', 1, C)
            else:
                sell_value = abs(diff_value)
                vol = int(sell_value / price / 100) * 100
                vol = min(vol, current_volume)
                if vol <= 0:
                    return True
                passorder(24, 1101, C.accountid, security, 5, -1, vol, 'ETF轮动', 1, C)
            return True
        except Exception as e2:
            print(f"[下单失败] {security}: order_target_value({e1}), passorder({e2})")
            return False


# ======================== 初始化模块 ========================
def init(ContextInfo):
    """
    初始化函数，策略启动时调用一次
    """
    print("========== 策略初始化开始 ==========")

    # 设置账号（回测时可填任意字符串，实盘请填真实资金账号）
    if not hasattr(ContextInfo, 'accountid') or not ContextInfo.accountid:
        ContextInfo.accountid = "test"  # 回测默认账号
    ContextInfo.set_account(ContextInfo.accountid)

    # ---------- ETF池（聚宽格式保留一份，QMT格式一份） ----------
    ContextInfo.etf_pool_jq = [
        # 大宗商品ETF
        "518880.XSHG",  # 黄金ETF
        "159980.XSHE",  # 有色ETF
        "159985.XSHE",  # 豆粕ETF
        "501018.XSHG",  # 南方原油
        '161226.XSHE',  # 白银LOF
        "159981.XSHE",  # 能源化工ETF
        # 国际ETF
        "513100.XSHG",  # 纳指ETF
        "159509.XSHE",  # 纳指科技ETF
        "513290.XSHG",  # 纳指生物ETF
        "513500.XSHG",  # 标普500ETF
        "159529.XSHE",  # 标普消费
        "513400.XSHG",  # 道琼斯ETF
        "513520.XSHG",  # 日经225ETF
        "513030.XSHG",  # 德国30ETF
        "513080.XSHG",  # 法国ETF
        "513310.XSHG",  # 中韩半导体ETF
        "513730.XSHG",  # 东南亚ETF
        # 香港ETF
        "159792.XSHE",  # 港股互联ETF
        "513130.XSHG",  # 恒生科技
        "513050.XSHG",  # 中概互联网ETF
        "159920.XSHE",  # 恒生ETF
        "513690.XSHG",  # 港股红利
        # 指数ETF
        "510300.XSHG",  # 沪深300ETF
        "510500.XSHG",  # 中证500ETF
        "510050.XSHG",  # 上证50ETF
        "510210.XSHG",  # 上证ETF
        "159915.XSHE",  # 创业板ETF
        "588080.XSHG",  # 科创50
        "512100.XSHG",  # 中证1000ETF
        "563360.XSHG",  # A500-ETF
        "563300.XSHG",  # 中证2000ETF
        # 风格ETF
        "512890.XSHG",  # 红利低波ETF
        "159967.XSHE",  # 创业板成长ETF
        "512040.XSHG",  # 价值ETF
        "159201.XSHE",  # 自由现金流ETF
        # 债券ETF
        "511380.XSHG",  # 可转债ETF
        "511010.XSHG",  # 国债ETF
        "511220.XSHG",  # 城投债ETF
    ]
    ContextInfo.etf_pool = [jq_to_qmt(c) for c in ContextInfo.etf_pool_jq]

    # ---------- 核心参数 ----------
    ContextInfo.lookback_days = 25
    ContextInfo.holdings_num = 1
    ContextInfo.defensive_etf_jq = "511880.XSHG"
    ContextInfo.defensive_etf = jq_to_qmt(ContextInfo.defensive_etf_jq)
    ContextInfo.min_money = 5000

    # ---------- 盈利保护参数 ----------
    ContextInfo.enable_profit_protection = True
    ContextInfo.profit_protection_lookback = 1
    ContextInfo.profit_protection_threshold = 0.05
    ContextInfo.profit_protection_check_times = ['11:00']

    ContextInfo.loss = 0.97
    ContextInfo.min_score_threshold = 0
    ContextInfo.max_score_threshold = 100.0

    # ---------- 成交量过滤 ----------
    ContextInfo.enable_volume_check = True
    ContextInfo.volume_lookback = 5
    ContextInfo.volume_threshold = 2
    ContextInfo.volume_return_limit = 1

    # ---------- 短期动量过滤 ----------
    ContextInfo.use_short_momentum_filter = True
    ContextInfo.short_lookback_days = 10
    ContextInfo.short_momentum_threshold = 0.0

    # ---------- 溢价率过滤（QMT不支持基金净值查询，默认关闭） ----------
    ContextInfo.enable_premium_filter = False
    ContextInfo.premium_threshold = 0.20

    # ---------- 运行时变量 ----------
    ContextInfo.rankings_cache = {'date': None, 'data': None}

    # ---------- 震荡期参数 ----------
    ContextInfo.enable_range_bound_mode = True
    ContextInfo.current_filter = '正常期'
    ContextInfo.risk_state = '正常期'
    ContextInfo.lookback_high_low_days = 20
    ContextInfo.risk_benchmark_jq = '510300.XSHG'
    ContextInfo.risk_benchmark = jq_to_qmt(ContextInfo.risk_benchmark_jq)
    ContextInfo.laplace_s_param = 0.05
    ContextInfo.laplace_min_slope = 0.001
    ContextInfo.gaussian_sigma = 1.2
    ContextInfo.gaussian_min_slope = 0.002

    ContextInfo.enable_bias_trigger = True
    ContextInfo.bias_threshold = 0.10
    ContextInfo.ma_period = 20
    ContextInfo.enable_rsi_trigger = True
    ContextInfo.rsi_overbought = 75
    ContextInfo.rsi_pullback = 60
    ContextInfo.previous_rsi = None
    ContextInfo.enable_stop_loss_trigger = False
    ContextInfo.stop_loss_triggered_today = False
    ContextInfo.stop_loss_triggered_date = None

    ContextInfo.enable_low_point_rise_trigger = True
    ContextInfo.low_point_rise_threshold = 0.03
    ContextInfo.enable_stable_signal_trigger = True
    ContextInfo.drawdown_recovery = 0.03
    ContextInfo.max_range_bound_days = 15
    ContextInfo.stable_days = 0

    ContextInfo.filter_switch_cooldown = 2
    ContextInfo.last_switch_date = None
    ContextInfo.range_bound_start_date = None
    ContextInfo.range_bound_days_count = 0
    ContextInfo.previous_drawdown = None

    # ---------- 每日执行标记（用于在1分钟handlebar中控制每天只执行一次） ----------
    ContextInfo.last_run_date = None
    ContextInfo.has_sold_today = False
    ContextInfo.has_bought_today = False
    ContextInfo.has_profit_check_today = False
    ContextInfo.has_range_bound_check_today = False
    ContextInfo.has_checked_morning = False

    # 首次初始化标记（在第一个交易日初始化震荡期状态）
    ContextInfo.range_bound_initialized = False

    print(f"策略初始化完成：ETF池{len(ContextInfo.etf_pool)}只，动量周期{ContextInfo.lookback_days}天，持仓{ContextInfo.holdings_num}只")
    print(f"盈利保护开关：{'开启' if ContextInfo.enable_profit_protection else '关闭'}")
    print(f"震荡期模式：{'开启' if ContextInfo.enable_range_bound_mode else '关闭'}")
    print("========== 策略初始化完成 ==========")


# ======================== 时间工具 ========================
def get_current_datetime(C):
    """获取当前K线对应的日期时间"""
    try:
        timetag = C.get_bar_timetag(C.barpos)
        dt_str = timetag_to_datetime(timetag, '%Y-%m-%d %H:%M:%S')
        return datetime.datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
    except Exception:
        # 备选：使用系统时间（实盘中）
        return datetime.datetime.now()


def get_current_date(C):
    """获取当前日期（datetime.date）"""
    return get_current_datetime(C).date()


def get_current_time_str(C):
    """获取当前时间 HH:MM"""
    return get_current_datetime(C).strftime('%H:%M')


def is_new_trading_day(C):
    """判断是否进入新的交易日"""
    today = get_current_date(C)
    if C.last_run_date != today:
        return True
    return False


def reset_daily_flags(C):
    """重置每日执行标记"""
    C.has_sold_today = False
    C.has_bought_today = False
    C.has_profit_check_today = False
    C.has_range_bound_check_today = False
    C.has_checked_morning = False
    C.last_run_date = get_current_date(C)
    print(f"新的一天 {C.last_run_date}，重置执行标记")


# ======================== 盈利保护 ========================
def check_profit_protection(C, security, lookback=None, threshold=None):
    """检查是否触发盈利保护（从最近N日最高点回撤超过阈值）"""
    if not C.enable_profit_protection:
        return False

    lookback = lookback or C.profit_protection_lookback
    threshold = threshold or C.profit_protection_threshold

    hist = attribute_history(C, security, lookback, '1d', ['high'])
    if hist.empty or len(hist) < lookback:
        return False

    max_high = hist['high'].max()
    current_price = get_current_data_price(C, security)
    if current_price <= 0:
        return False

    if current_price <= max_high * (1 - threshold):
        print(f"[回撤] {security} {get_stock_name(C, security)} 触发盈利保护："
              f"当前价{current_price:.3f}，最近{lookback}日最高{max_high:.3f}，"
              f"回撤{(1 - current_price / max_high) * 100:.2f}%")
        return True
    return False


def profit_protection_check(C):
    """盈利保护独立检查函数"""
    if not C.enable_profit_protection:
        return

    print("========== 盈利保护检查开始 ==========")
    positions = get_positions(C)
    for sec in list(positions.keys()):
        if sec not in C.etf_pool and sec != C.defensive_etf:
            continue
        if positions[sec]['volume'] > 0:
            if check_profit_protection(C, sec):
                if smart_order_target_value(C, sec, 0):
                    print(f"[止盈] 盈利保护卖出：{sec} {get_stock_name(C, sec)}")
                    if getattr(C, 'enable_stop_loss_trigger', False):
                        C.stop_loss_triggered_today = True
                        C.stop_loss_triggered_date = get_current_date(C)
    print("========== 盈利保护检查完成 ==========")


# ======================== 震荡期机制 ========================
def calculate_rsi(close, period=14):
    """计算RSI值"""
    try:
        if len(close) < period + 1:
            return None
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception:
        return None


def laplace_filter(price, s=0.05):
    """拉普拉斯滤波器（正常期使用）"""
    alpha = 1 - np.exp(-s)
    L = np.zeros(len(price))
    L[0] = price[0]
    for t in range(1, len(price)):
        L[t] = alpha * price[t] + (1 - alpha) * L[t - 1]
    return L


def gaussian_filter_last_two(price, sigma=1.2):
    """仅计算高斯滤波最后两个点（震荡期使用，效率优化）"""
    n = len(price)
    if n < 2:
        return 0, 0
    idx_1 = np.arange(n)
    weights_1 = np.exp(-((idx_1 + 1) ** 2) / (2 * sigma ** 2))[::-1]
    weights_1 /= np.sum(weights_1)
    g1 = np.sum(price * weights_1)
    price_2 = price[:-1]
    idx_2 = np.arange(n - 1)
    weights_2 = np.exp(-((idx_2 + 1) ** 2) / (2 * sigma ** 2))[::-1]
    weights_2 /= np.sum(weights_2)
    g2 = np.sum(price_2 * weights_2)
    return g1, g2


def get_risk_benchmark_state(C):
    """获取风险基准的日线+盘中融合状态，用于震荡期判断"""
    required_days = max(C.ma_period, C.lookback_high_low_days)
    lookback = required_days + 30

    df = attribute_history(C, C.risk_benchmark, lookback, '1d', ['close', 'high', 'low'])
    if df is None or len(df) < required_days:
        return None

    daily_close = df['close'].values.astype(float)
    daily_high = df['high'].values.astype(float)
    daily_low = df['low'].values.astype(float)
    current_price = float(daily_close[-1])
    intraday_high = current_price
    intraday_low = current_price
    data_source = '昨日日线'

    # 尝试获取实时价格
    try:
        live_price = get_current_data_price(C, C.risk_benchmark)
        if live_price > 0 and live_price != current_price:
            current_price = live_price
            intraday_high = max(intraday_high, current_price)
            intraday_low = min(intraday_low, current_price)
            data_source = '实时快照'
    except Exception:
        pass

    close_series = np.append(daily_close, current_price)
    high_series = np.append(daily_high, max(intraday_high, current_price))
    low_series = np.append(daily_low, min(intraday_low, current_price))

    recent_high = np.max(high_series[-C.lookback_high_low_days:])
    recent_low = np.min(low_series[-C.lookback_high_low_days:])
    ma = np.mean(close_series[-C.ma_period:])
    current_rsi = calculate_rsi(close_series, period=14)
    previous_rsi = calculate_rsi(daily_close, period=14)

    return {
        'close_series': close_series,
        'high_series': high_series,
        'low_series': low_series,
        'current_price': current_price,
        'recent_high': recent_high,
        'recent_low': recent_low,
        'ma': ma,
        'current_rsi': current_rsi,
        'previous_rsi': previous_rsi,
        'data_source': data_source,
    }


def is_fresh_stop_loss_signal(C):
    """判断止损信号是否仍在有效期内"""
    signal_date = getattr(C, 'stop_loss_triggered_date', None)
    if signal_date is None:
        return False
    today = get_current_date(C)
    if signal_date == today:
        return True
    C.stop_loss_triggered_today = False
    C.stop_loss_triggered_date = None
    return False


def init_range_bound_status(C):
    """首次运行时，根据历史数据判断当前是否处于震荡期"""
    if not C.enable_range_bound_mode:
        return
    print("【首次运行】初始化震荡期状态...")
    try:
        lookback = max(C.ma_period, C.lookback_high_low_days) + 30

        df = attribute_history(C, C.risk_benchmark, lookback, '1d', ['close', 'high', 'low'])
        if df is None or len(df) < max(C.ma_period, C.lookback_high_low_days):
            print("【首次运行】数据不足，保持正常期")
            return

        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        current_price = close[-1]

        if len(close) >= C.lookback_high_low_days:
            recent_high = np.max(high[-C.lookback_high_low_days:])
            recent_low = np.min(low[-C.lookback_high_low_days:])
        else:
            recent_high = np.max(high)
            recent_low = np.min(low)

        ma = np.mean(close[-C.ma_period:])
        bias = (current_price - ma) / ma if ma > 0 else 0
        current_rsi = calculate_rsi(close, period=14)

        should_enter = False
        signals = []
        if C.enable_bias_trigger and bias > C.bias_threshold:
            should_enter = True
            signals.append(f"乖离率{bias:.2%}>{C.bias_threshold:.0%}")
        if C.enable_rsi_trigger and current_rsi is not None and len(close) >= 15:
            prev_rsi = calculate_rsi(close[:-1], period=14)
            if prev_rsi is not None and prev_rsi > C.rsi_overbought and current_rsi < C.rsi_pullback:
                should_enter = True
                signals.append(f"RSI超买回落{prev_rsi:.1f}->{current_rsi:.1f}")

        if should_enter:
            C.current_filter = '震荡期'
            C.risk_state = '震荡期'
            C.range_bound_start_date = get_current_date(C)
            C.range_bound_days_count = 0
            print(f"【首次运行】初始化进入震荡期: {'; '.join(signals)}")
        else:
            C.current_filter = '正常期'
            C.risk_state = '正常期'
            if len(close) >= C.lookback_high_low_days:
                C.previous_drawdown = (recent_high - current_price) / recent_high if recent_high > 0 else 0
            else:
                C.previous_drawdown = 0
            C.previous_rsi = current_rsi
            rsi_str = f"{current_rsi:.1f}" if current_rsi is not None else "N/A"
            print(f"【首次运行】初始状态: 正常期, 乖离率: {bias:.2%}, RSI: {rsi_str}")
    except Exception as e:
        print(f"【首次运行】初始化震荡期状态异常: {e}，保持正常期")


def check_and_exit_range_bound_mode(C):
    """检查是否需要退出震荡期"""
    if not C.enable_range_bound_mode or C.current_filter != '震荡期':
        return

    print("【震荡期退出检查】开始...")
    try:
        benchmark_state = get_risk_benchmark_state(C)
        if benchmark_state is None:
            print("【震荡期退出检查】数据不足，跳过")
            return

        close = benchmark_state['close_series']
        current_price = benchmark_state['current_price']
        recent_high = benchmark_state['recent_high']
        recent_low = benchmark_state['recent_low']

        current_drawdown = (recent_high - current_price) / recent_high if recent_high > 0 else 0
        rise_from_low = (current_price - recent_low) / recent_low if recent_low > 0 else 0
        recovery_signals = []
        ma = benchmark_state['ma']
        current_rsi = benchmark_state['current_rsi']

        print(f"【震荡期数据】当前价: {current_price:.3f}, 回撤: {current_drawdown:.2%}, 从低点涨幅: {rise_from_low:.2%}")

        if C.enable_low_point_rise_trigger and rise_from_low >= C.low_point_rise_threshold:
            recovery_signals.append(f"从低点上涨{rise_from_low:.2%}>={C.low_point_rise_threshold:.0%}")

        if C.enable_stable_signal_trigger:
            if current_price > ma:
                recovery_signals.append("价格站上均线")
            if len(close) >= 2 and close[-1] > close[-2]:
                recovery_signals.append("价格回升")
            if C.previous_drawdown is not None and current_drawdown < C.previous_drawdown:
                recovery_signals.append("回撤收窄")
            if current_rsi is not None and C.previous_rsi is not None and current_rsi > C.previous_rsi:
                recovery_signals.append("RSI回升")

            drawdown_safe = current_drawdown < C.drawdown_recovery
            if drawdown_safe:
                C.stable_days += 1
                print(f"【企稳计数】连续企稳天数: {C.stable_days}")
            else:
                C.stable_days = 0

        C.previous_drawdown = current_drawdown
        C.previous_rsi = current_rsi

        range_bound_days = 0
        if C.range_bound_start_date is not None:
            trade_days = get_trade_days_list(C, get_current_datetime(C).strftime('%Y%m%d'),
                                             C.max_range_bound_days + 5)
            range_bound_days = sum(1 for d in trade_days
                                   if d >= C.range_bound_start_date and d < get_current_date(C))
            if range_bound_days >= C.max_range_bound_days:
                recovery_signals.append(f"震荡期满({range_bound_days}天)")

        low_point_condition = C.enable_low_point_rise_trigger and rise_from_low >= C.low_point_rise_threshold
        stable_condition = False
        if C.enable_stable_signal_trigger:
            drawdown_safe = current_drawdown < C.drawdown_recovery
            stable_condition = drawdown_safe and len(recovery_signals) >= 2 and C.stable_days >= 2
        force_condition = range_bound_days >= C.max_range_bound_days

        should_recover = low_point_condition or stable_condition or force_condition
        if should_recover:
            can_switch = True
            if C.last_switch_date is not None:
                trade_days = get_trade_days_list(C, get_current_datetime(C).strftime('%Y%m%d'), 10)
                days_since = sum(1 for d in trade_days if d >= C.last_switch_date and d < get_current_date(C))
                if days_since < C.filter_switch_cooldown:
                    can_switch = False
                    print(f"【震荡期退出】冷却期中，距上次切换{days_since}天")
            if can_switch:
                C.current_filter = '正常期'
                C.risk_state = '正常期'
                C.last_switch_date = get_current_date(C)
                C.range_bound_start_date = None
                C.range_bound_days_count = 0
                C.stable_days = 0
                print(f"【退出震荡期】切换回拉普拉斯滤波器: {'; '.join(recovery_signals)}")
        else:
            print("【震荡期退出检查】未满足退出条件，保持震荡期")
    except Exception as e:
        print(f"【震荡期退出检查】判断出错: {e}")


def check_and_enter_range_bound_mode(C):
    """检查是否需要进入震荡期"""
    if not C.enable_range_bound_mode:
        return
    if C.current_filter == '震荡期':
        return

    print("【震荡期进入检查】开始...")
    stop_loss_signal_active = is_fresh_stop_loss_signal(C)

    can_switch = True
    if C.last_switch_date is not None:
        trade_days = get_trade_days_list(C, get_current_datetime(C).strftime('%Y%m%d'), 10)
        days_since = sum(1 for d in trade_days if d >= C.last_switch_date and d < get_current_date(C))
        if days_since < C.filter_switch_cooldown:
            can_switch = False
            print(f"【震荡期检查】冷却期中，距上次切换{days_since}天")

    if not can_switch:
        return

    risk_signals = []
    try:
        benchmark_state = get_risk_benchmark_state(C)
        if benchmark_state is not None:
            close = benchmark_state['close_series']
            current_price = benchmark_state['current_price']

            if C.enable_bias_trigger:
                ma = benchmark_state['ma']
                bias = (current_price - ma) / ma if ma > 0 else 0
                if bias > C.bias_threshold:
                    risk_signals.append(f"乖离率过大({bias:.2%}>{C.bias_threshold:.0%})")

            if C.enable_rsi_trigger:
                current_rsi = benchmark_state['current_rsi']
                if len(close) >= 15 and current_rsi is not None:
                    prev_rsi = benchmark_state['previous_rsi']
                    if prev_rsi is not None:
                        if prev_rsi > C.rsi_overbought and current_rsi < C.rsi_pullback and current_rsi < prev_rsi:
                            risk_signals.append(f"RSI超买回落({prev_rsi:.1f}->{current_rsi:.1f})")
    except Exception as e:
        print(f"【震荡期检查】获取基准数据异常: {e}")

    if C.enable_stop_loss_trigger and stop_loss_signal_active:
        risk_signals.append("盈利保护触发止损")

    if len(risk_signals) > 0:
        C.current_filter = '震荡期'
        C.risk_state = '震荡期'
        C.last_switch_date = get_current_date(C)
        C.range_bound_start_date = get_current_date(C)
        C.range_bound_days_count = 0
        C.stable_days = 0
        C.stop_loss_triggered_today = False
        C.stop_loss_triggered_date = None
        print(f"【进入震荡期】切换到高斯滤波器: {'; '.join(risk_signals)}")
    else:
        print("【震荡期检查】未满足进入条件，保持正常期")


def check_range_bound(C):
    """震荡期检查入口"""
    if not C.enable_range_bound_mode:
        return
    print("========== 震荡期检查开始 ==========")
    print(f"当前状态: {C.current_filter}")
    check_and_exit_range_bound_mode(C)
    check_and_enter_range_bound_mode(C)
    print(f"检查后状态: {C.current_filter}")
    C.rankings_cache = {'date': None, 'data': None}
    print("========== 震荡期检查完成 ==========")


# ======================== 核心计算模块 ========================
def get_cached_rankings(C):
    """获取缓存的ETF排名"""
    today = get_current_date(C)
    if C.rankings_cache['date'] != today:
        print("重新计算ETF排名...")
        ranked = get_ranked_etfs(C)
        C.rankings_cache = {'date': today, 'data': ranked}
    else:
        print("使用缓存的ETF排名")
    return C.rankings_cache['data']


def get_ranked_etfs(C):
    """计算所有ETF的动量得分，应用过滤条件，返回按得分降序的列表"""
    etf_metrics = []
    for etf in C.etf_pool:
        # 价格获取失败视为停牌，跳过
        current_price = get_current_data_price(C, etf)
        if current_price <= 0:
            continue

        metrics = calculate_momentum_metrics(C, etf)
        if metrics is not None:
            if C.min_score_threshold < metrics['score'] < C.max_score_threshold:
                etf_metrics.append(metrics)
            else:
                print(f"{etf} {metrics['etf_name']} 得分{metrics['score']:.2f}超出阈值，过滤")

    etf_metrics.sort(key=lambda x: x['score'], reverse=True)
    return etf_metrics


def calculate_momentum_metrics(C, etf):
    """计算单只ETF的动量指标，应用所有过滤条件"""
    try:
        name = get_stock_name(C, etf)
        lookback = max(C.lookback_days, C.short_lookback_days) + 20
        prices = attribute_history(C, etf, lookback, '1d', ['close', 'high'])
        if len(prices) < C.lookback_days:
            return None

        current_price = get_current_data_price(C, etf)
        if current_price <= 0:
            return None
        price_series = np.append(prices["close"].values, current_price)

        # ===== 1. 盈利保护检查（排除） =====
        if check_profit_protection(C, etf):
            print(f"[排除] {etf} {name} 触发盈利保护，从排名中排除")
            return None

        # ===== 2. 成交量过滤（排除）【QMT简化版：仅用日线估算】 =====
        if C.enable_volume_check:
            vol_ratio = get_volume_ratio(C, etf)
            if vol_ratio is not None:
                annualized = get_annualized_returns(price_series, C.lookback_days)
                if annualized > C.volume_return_limit:
                    print(f"[放量] {etf} {name} 成交量放量{vol_ratio:.1f}倍，年化{annualized*100:.1f}% > 阈值，过滤")
                    return None

        # ===== 3. 短期动量过滤（排除） =====
        if len(price_series) >= C.short_lookback_days + 1:
            short_return = price_series[-1] / price_series[-(C.short_lookback_days + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / C.short_lookback_days) - 1
        else:
            short_annualized = 0

        if C.use_short_momentum_filter and short_annualized < C.short_momentum_threshold:
            return None

        # ===== 4. 长期动量计算（得分） =====
        recent = price_series[-(C.lookback_days + 1):]
        y = np.log(recent)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1

        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        score = annualized_returns * r_squared

        # ===== 5. 近3日单日跌幅过滤（排除） =====
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            if min(day1, day2, day3) < C.loss:
                print(f"[暴跌] {etf} {name} 近3日有单日跌幅超{(1-C.loss)*100:.1f}%，直接排除")
                return None

        # ===== 6. 动态滤波器过滤（震荡期机制） =====
        if C.enable_range_bound_mode and len(price_series) >= 10:
            try:
                laplace_values = laplace_filter(price_series, s=C.laplace_s_param)
                laplace_slope = laplace_values[-1] - laplace_values[-2] if len(laplace_values) >= 2 else 0
                passed_laplace = (current_price > laplace_values[-1] and laplace_slope > C.laplace_min_slope)

                g1_val, g2_val = gaussian_filter_last_two(price_series, sigma=C.gaussian_sigma)
                gaussian_slope = g1_val - g2_val
                passed_gaussian = (current_price > g1_val and gaussian_slope > C.gaussian_min_slope)

                if C.current_filter == '正常期':
                    passed_filter = passed_laplace
                    filter_name = '拉普拉斯'
                else:
                    passed_filter = passed_gaussian
                    filter_name = '高斯'

                if not passed_filter:
                    print(f"{etf} {name} 未通过{filter_name}滤波器({C.current_filter})，过滤")
                    return None
            except Exception as e:
                print(f"{etf} {name} 滤波器计算异常: {e}")

        return {
            'etf': etf,
            'etf_name': name,
            'annualized_returns': annualized_returns,
            'r_squared': r_squared,
            'score': score,
            'current_price': current_price,
            'short_annualized': short_annualized,
        }
    except Exception as e:
        print(f"计算{etf} {get_stock_name(C, etf)}时出错: {e}")
        return None


def get_annualized_returns(price_series, lookback_days):
    """计算加权年化收益率"""
    recent = price_series[-(lookback_days + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, _ = np.polyfit(x, y, 1, w=weights)
    return math.exp(slope * 250) - 1


def get_volume_ratio(C, security, lookback=None, threshold=None):
    """
    计算当日成交量与过去N日均量的比值
    【QMT简化版】仅用日线数据估算，当日volume取截至当前时刻的累计值
    """
    lookback = lookback or C.volume_lookback
    threshold = threshold or C.volume_threshold
    try:
        hist = attribute_history(C, security, lookback, '1d', ['volume'])
        if hist.empty or len(hist) < lookback:
            return None
        avg_vol = hist['volume'].mean()

        # 获取当天截至目前的成交量（日线中当天数据即为累计值）
        today_str = get_current_date(C).strftime('%Y%m%d')
        today_data = get_price(C, security, today_str, today_str, '1d', ['volume'])
        if today_data.empty or 'volume' not in today_data.columns:
            return None

        current_vol = float(today_data['volume'].iloc[0])
        ratio = current_vol / avg_vol if avg_vol > 0 else 0

        if ratio > threshold:
            return ratio
        return None
    except Exception as e:
        print(f"成交量计算失败 {security}: {e}")
        return None


def check_defensive_etf_available(C):
    """检查防御ETF是否可交易"""
    price = get_current_data_price(C, C.defensive_etf)
    return price > 0



# ======================== 卖出模块 ========================
def etf_sell_trade(C):
    """卖出不符合条件的持仓（排名变化）"""
    print("========== 卖出操作开始 ==========")

    ranked = get_cached_rankings(C)
    target_etfs = []
    for m in ranked[:C.holdings_num]:
        if m['score'] >= C.min_score_threshold:
            target_etfs.append(m['etf'])

    defensive_available = check_defensive_etf_available(C)
    if not target_etfs and defensive_available:
        target_etfs = [C.defensive_etf]

    target_set = set(target_etfs)
    positions = get_positions(C)

    for sec in list(positions.keys()):
        if sec not in C.etf_pool and sec != C.defensive_etf:
            continue
        if sec not in target_set:
            if positions[sec]['volume'] > 0:
                if smart_order_target_value(C, sec, 0):
                    print(f"[卖出] 卖出不在目标的持仓：{sec} {get_stock_name(C, sec)}")

    print("========== 卖出操作完成 ==========")


# ======================== 买入模块 ========================
def etf_buy_trade(C):
    """买入符合条件的ETF，等权分配"""
    print("========== 买入操作开始 ==========")

    ranked = get_cached_rankings(C)
    print("=== ETF排名前5 ===")
    for i, m in enumerate(ranked[:5]):
        print(f"排名{i+1}: {m['etf']} {m['etf_name']} 得分{m['score']:.4f} "
              f"年化{m['annualized_returns']*100:.2f}% R^2={m['r_squared']:.4f}")

    target_etfs = []
    for m in ranked:
        if len(target_etfs) >= C.holdings_num:
            break
        target_etfs.append(m['etf'])
        print(f"[目标] 目标ETF {len(target_etfs)}: {m['etf']} {m['etf_name']} 得分{m['score']:.4f}")

    if not target_etfs:
        if check_defensive_etf_available(C):
            target_etfs = [C.defensive_etf]
            print(f"[止盈] 进入防御模式，选择防御ETF：{C.defensive_etf} {get_stock_name(C, C.defensive_etf)}")
        else:
            print("[空仓] 无目标ETF且防御不可用，保持空仓")
            return

    # 检查是否有持仓需要先卖出
    positions = get_positions(C)
    current_etf_pos = [s for s in positions if s in C.etf_pool or s == C.defensive_etf]
    to_sell = [s for s in current_etf_pos if s not in target_etfs]
    if to_sell:
        to_sell_names = [get_stock_name(C, s) for s in to_sell]
        print(f"尚有持仓需要卖出：{list(zip(to_sell, to_sell_names))}，等待卖出完成再买入")
        return

    # 等权分配
    acc = get_account_info(C)
    if acc is None:
        print("无法获取账户信息，跳过买入")
        return

    total_value = acc['total_asset']
    cash = acc['available_cash']
    per_etf_value = total_value / max(len(target_etfs), 1)

    if per_etf_value < C.min_money:
        print(f"单标的目标金额{per_etf_value:.0f}小于最小交易金额{C.min_money}，不买入")
        return

    for etf in target_etfs:
        price = get_current_data_price(C, etf)
        if price <= 0:
            continue
        needed = per_etf_value
        current_value = positions.get(etf, {}).get('market_value', 0)
        diff = needed - current_value

        if abs(diff) < C.min_money:
            print(f"{etf} {get_stock_name(C, etf)} 市值{current_value:.0f}与目标{needed:.0f}差异过小，不调仓")
            continue

        if diff > 0 and diff > cash:
            print(f"{etf} 目标差额{diff:.0f}大于可用资金{cash:.0f}，跳过")
            continue

        if smart_order_target_value(C, etf, needed):
            print(f"[买入] 买入/调仓：{etf} {get_stock_name(C, etf)} 目标金额{needed:.0f}")
            if diff > 0:
                cash -= diff

    print("========== 买入操作完成 ==========")


# ======================== QMT主循环 ========================
def handlebar(ContextInfo):
    """
    QMT核心处理函数。
    策略周期建议设为1分钟，这样handlebar每分钟触发一次，
    我们在其中判断当前时间，模拟聚宽的run_daily定时调度。
    """
    # 只在最后一根K线执行（实盘）或每根K线都判断（回测兼容）
    # 由于时间窗口判断（如'13:10' <= current_time <= '13:15'）已经天然限制了执行频率，
    # 无需额外判断is_last_bar()

    # ---------- 判断是否进入新的交易日，重置每日标记 ----------
    if is_new_trading_day(ContextInfo):
        reset_daily_flags(ContextInfo)
        # 首次运行时初始化震荡期状态
        if not ContextInfo.range_bound_initialized:
            init_range_bound_status(ContextInfo)
            ContextInfo.range_bound_initialized = True

    current_time = get_current_time_str(ContextInfo)

    # ---------- 09:10 检查持仓（日志） ----------
    if '09:10' <= current_time <= '09:15' and not ContextInfo.has_checked_morning:
        positions = get_positions(ContextInfo)
        for sec, pos in positions.items():
            print(f"[持仓] 持仓：{sec} {get_stock_name(ContextInfo, sec)} "
                  f"数量{pos['volume']} 成本{pos['cost']:.3f} 市值{pos['market_value']:.0f}")
        ContextInfo.has_checked_morning = True

    # ---------- 11:00 盈利保护检查 ----------
    if '11:00' <= current_time <= '11:05' and not ContextInfo.has_profit_check_today:
        if ContextInfo.enable_profit_protection:
            profit_protection_check(ContextInfo)
        ContextInfo.has_profit_check_today = True

    # ---------- 13:10 卖出 ----------
    if '13:10' <= current_time <= '13:15' and not ContextInfo.has_sold_today:
        etf_sell_trade(ContextInfo)
        ContextInfo.has_sold_today = True

    # ---------- 13:11 买入 ----------
    # 注意：实际执行会在13:11-13:16窗口，确保在卖出之后
    if '13:11' <= current_time <= '13:16' and not ContextInfo.has_bought_today:
        etf_buy_trade(ContextInfo)
        ContextInfo.has_bought_today = True

    # ---------- 兼容日线周期回测 ----------
    # 如果策略运行在日线周期，handlebar在15:00触发，上述时间窗口不会命中
    # 因此在15:00-15:30增加一次性调仓逻辑
    if '15:00' <= current_time <= '15:30' and not ContextInfo.has_sold_today:
        if not ContextInfo.has_range_bound_check_today:
            check_range_bound(ContextInfo)
            ContextInfo.has_range_bound_check_today = True
        etf_sell_trade(ContextInfo)
        ContextInfo.has_sold_today = True
        etf_buy_trade(ContextInfo)
        ContextInfo.has_bought_today = True

    # ---------- 13:55 震荡期检查 ----------
    if '13:55' <= current_time <= '14:00' and not ContextInfo.has_range_bound_check_today:
        check_range_bound(ContextInfo)
        ContextInfo.has_range_bound_check_today = True

    # ---------- 15:10 收盘重置 ----------
    if '15:10' <= current_time <= '15:15':
        if ContextInfo.current_filter == '震荡期' and ContextInfo.range_bound_start_date is not None:
            try:
                trade_days = get_trade_days_list(ContextInfo,
                                                  get_current_datetime(ContextInfo).strftime('%Y%m%d'),
                                                  ContextInfo.max_range_bound_days + 5)
                count = sum(1 for d in trade_days
                            if d >= ContextInfo.range_bound_start_date
                            and d < get_current_date(ContextInfo))
                ContextInfo.range_bound_days_count = count
                print(f"震荡期已持续 {ContextInfo.range_bound_days_count} 个交易日")
            except Exception as e:
                print(f"收盘重置出错: {e}")
        # 重置早盘标记，为下一交易日做准备
        ContextInfo.has_checked_morning = False
