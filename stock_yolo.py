# 基于 etf-clone.py 策略，适配上证50成分股轮动
# 标题：上证50成分股动量轮动 + 拉普拉斯/高斯动态滤波
# 作者：基于策略手艺人/king088/晨曦量化

"""
    策略核心逻辑：
    1. 从上证50成分股中，每日按动量得分（年化收益 × R²）排序
    2. 选取排名靠前的 N 只股票等权持仓
    3. 自动判断当前是趋势行情还是震荡行情，切换不同滤波器：
       - 正常期 → 拉普拉斯滤波器（灵敏，跟趋势）
       - 震荡期 → 高斯滤波器（平滑，防假信号）
    4. 包含盈利保护、成交量过滤、ST过滤、新股过滤等多重风控

    ==========================================================

    震荡期机制（策略核心特色）
        通过监测上证50ETF（510050）的技术状态，自动切换：

    【进入震荡期的条件】（任一触发即切换）
        ① 乖离率 > 10%：价格偏离20日均线过远
        ② RSI超买回落：RSI从>75跌破60

    【退出震荡期的条件】（任一触发即切换）
        ① 从近20日最低点上涨 ≥ 3%
        ② 连续企稳信号：回撤< 3% + 至少2个企稳指标 + 连续≥2天
        ③ 震荡期满15个交易日，强制退出

    【冷却机制】
        切换后有2个交易日的冷却期，防止频繁翻转。
    ==========================================================
"""

import math
import datetime
import numpy as np
import pandas as pd
from jqdata import *


# ======================== 初始化模块 ========================
def initialize(context):
    """
    初始化函数：设置交易参数、股票池、核心参数、调度任务
    """
    # ---------- 交易设置 ----------
    set_option("avoid_future_data", True)       # 防止未来函数
    set_option("use_real_price", True)
    set_slippage(PriceRelatedSlippage(0.0002), type="stock")
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,                    # 股票卖出印花税 0.1%
            open_commission=0.00025,            # 券商佣金
            close_commission=0.00025,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )
    set_benchmark("000016.XSHG")                # 基准：上证50指数
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'debug')
    log.info("========== 策略初始化开始 ==========")

    # ---------- 股票池 ----------
    # 上证50成分股，每日更新
    g.stock_pool = get_index_stocks('000016.XSHG')
    log.info(f"初始股票池：上证50成分股共 {len(g.stock_pool)} 只")

    # ---------- 核心参数 ----------
    g.lookback_days = 25            # 动量计算周期
    g.holdings_num = 3             # 持仓数量（集中持仓）
    g.defensive_etf = "511880.XSHG"  # 防御ETF（银华日利货币基金）
    g.min_money = 5000              # 最小交易金额

    # ---------- 盈利保护参数 ----------
    g.enable_profit_protection = True
    g.profit_protection_lookback = 3      # 盈利保护回看周期（天），放宽到3日高点
    g.profit_protection_threshold = 0.08  # 盈利保护回撤阈值（8%，给趋势股空间）
    g.profit_protection_check_times = ['11:00']

    # ---------- 硬止损参数 ----------
    g.enable_hard_stop_loss = True        # 硬止损开关
    g.hard_stop_loss_pct = 0.08           # 单票亏损超过8%强制止损

    # ---------- 趋势过滤参数 ----------
    g.enable_trend_filter = True          # 个股均线趋势过滤
    g.trend_ma_period = 20                # 20日均线

    # ---------- 大盘择时参数 ----------
    g.enable_market_timing = True         # 大盘择时开关
    g.market_timing_index = '000016.XSHG' # 上证50指数
    g.market_timing_ma = 20               # 大盘20日均线

    # ---------- 换仓控制参数 ----------
    g.min_hold_days = 2                   # 最小持仓天数（避免T+1反复打脸）
    g.switch_score_diff = 0.05            # 换仓得分差距阈值（新目标得分需比当前持仓高5%）

    g.loss = 0.97                   # 近3日单日跌幅阈值（3%，和ETF版保持一致）
    g.min_score_threshold = 0       # 最低得分
    g.max_score_threshold = 100.0   # 最高得分

    # ---------- 成交量过滤 ----------
    g.enable_volume_check = True
    g.volume_lookback = 5
    g.volume_threshold = 3          # 放量阈值（股票比ETF更严格）
    g.volume_return_limit = 1       # 年化收益>100%时启用放量过滤

    # ---------- 短期动量过滤 ----------
    g.use_short_momentum_filter = True
    g.short_lookback_days = 10
    g.short_momentum_threshold = 0.0

    # ---------- 股票专属过滤参数 ----------
    g.enable_st_filter = True       # 过滤ST股票
    g.enable_new_stock_filter = True  # 过滤次新股
    g.new_stock_days = 60           # 上市不满60天视为次新股
    g.enable_limit_up_filter = True   # 过滤涨停股（不追涨停）
    g.min_market_cap = 50e8         # 最小市值50亿（过滤流动性差的）

    # ---------- 运行时变量 ----------
    g.rankings_cache = {'date': None, 'data': None}

    # ---------- 震荡期参数 ----------
    g.enable_range_bound_mode = True
    g.current_filter = '正常期'
    g.risk_state = '正常期'
    g.lookback_high_low_days = 20
    g.risk_benchmark = '510050.XSHG'
    g.laplace_s_param = 0.05
    g.laplace_min_slope = 0.001
    g.gaussian_sigma = 1.2
    g.gaussian_min_slope = 0.002
    # 进入震荡期条件
    g.enable_bias_trigger = True
    g.bias_threshold = 0.10
    g.ma_period = 20
    g.enable_rsi_trigger = True
    g.rsi_overbought = 75
    g.rsi_pullback = 60
    g.previous_rsi = None
    g.enable_stop_loss_trigger = False
    g.stop_loss_triggered_today = False
    g.stop_loss_triggered_date = None
    # 退出震荡期条件
    g.enable_low_point_rise_trigger = True
    g.low_point_rise_threshold = 0.03
    g.enable_stable_signal_trigger = True
    g.drawdown_recovery = 0.03
    g.max_range_bound_days = 15
    g.stable_days = 0
    # 震荡期控制
    g.filter_switch_cooldown = 2
    g.last_switch_date = None
    g.range_bound_start_date = None
    g.range_bound_days_count = 0
    g.previous_drawdown = None

    # ---------- 交易调度 ----------
    run_daily(update_stock_pool, time='09:05')   # 更新股票池
    run_daily(check_positions, time='09:10')
    run_daily(hard_stop_loss_check, time='09:45')  # 硬止损检查
    run_daily(stock_sell_trade, time='13:10')
    run_daily(stock_buy_trade, time='13:11')

    for check_time in g.profit_protection_check_times:
        run_daily(profit_protection_check, time=check_time)
        log.info(f"已注册盈利保护检查时间：{check_time}")

    run_daily(check_range_bound, time='13:55')
    run_daily(reset_range_bound_daily, time='15:10')

    log.info(f"策略初始化完成：持仓{g.holdings_num}只，动量周期{g.lookback_days}天")
    log.info(f"盈利保护：{'开启' if g.enable_profit_protection else '关闭'}，回撤阈值{g.profit_protection_threshold*100:.0f}%，回看{g.profit_protection_lookback}天")
    log.info(f"硬止损：{'开启' if g.enable_hard_stop_loss else '关闭'}，阈值{g.hard_stop_loss_pct*100:.0f}%")
    log.info(f"趋势过滤：{'开启' if g.enable_trend_filter else '关闭'}，{g.trend_ma_period}日均线")
    log.info(f"大盘择时：{'开启' if g.enable_market_timing else '关闭'}，{g.market_timing_index}的{g.market_timing_ma}日均线")
    log.info(f"换仓控制：最小持仓{g.min_hold_days}天，得分差距{g.switch_score_diff*100:.0f}%")
    log.info(f"股票专属过滤：ST={'开启' if g.enable_st_filter else '关闭'}, "
             f"次新股={'开启' if g.enable_new_stock_filter else '关闭'}, "
             f"涨停过滤={'开启' if g.enable_limit_up_filter else '关闭'}")
    log.info(f"震荡期模式：{'开启' if g.enable_range_bound_mode else '关闭'}")

    init_range_bound_status(context)
    log.info("========== 策略初始化完成 ==========")


# ==================== 股票池更新 ====================
def update_stock_pool(context):
    """每日更新上证50成分股列表"""
    g.stock_pool = get_index_stocks('000016.XSHG')
    log.info(f"【股票池更新】上证50成分股共 {len(g.stock_pool)} 只")


# ==================== 盈利保护模块 ====================
def profit_protection_check(context):
    """独立执行的盈利保护检查函数"""
    if not g.enable_profit_protection:
        return

    log.info("========== 盈利保护独立检查开始 ==========")
    for sec in list(context.portfolio.positions.keys()):
        if sec not in g.stock_pool and sec != g.defensive_etf:
            continue
        pos = context.portfolio.positions[sec]
        if pos.total_amount > 0:
            if check_profit_protection(sec, context):
                if smart_order_target_value(sec, 0, context):
                    log.info(f"🛡️ 盈利保护卖出（独立检查）：{sec} {get_name(sec)}")
                    if getattr(g, 'enable_stop_loss_trigger', False):
                        g.stop_loss_triggered_today = True
                        g.stop_loss_triggered_date = context.current_dt.date()
    log.info("========== 盈利保护独立检查完成 ==========")


def check_profit_protection(security, context, lookback=None, threshold=None):
    """检查是否触发盈利保护（从最近N日最高点回撤超过阈值）"""
    if not g.enable_profit_protection:
        return False

    lookback = lookback or g.profit_protection_lookback
    threshold = threshold or g.profit_protection_threshold

    hist = attribute_history(security, lookback, '1d', ['high'])
    if hist.empty or len(hist) < lookback:
        return False
    max_high = hist['high'].max()
    current_price = get_current_data()[security].last_price

    if current_price <= max_high * (1 - threshold):
        log.info(
            f"🔻 {security} {get_name(security)} 触发盈利保护："
            f"当前价{current_price:.3f}，最近{lookback}日最高{max_high:.3f}，"
            f"回撤{(1 - current_price / max_high) * 100:.2f}%")
        return True
    return False


# ==================== 硬止损模块 ====================
def hard_stop_loss_check(context):
    """
    硬止损检查：遍历持仓，单票亏损超过阈值强制卖出
    """
    if not g.enable_hard_stop_loss:
        return
    log.info("========== 硬止损检查开始 ==========")
    for sec in list(context.portfolio.positions.keys()):
        if sec not in g.stock_pool and sec != g.defensive_etf:
            continue
        pos = context.portfolio.positions[sec]
        if pos.total_amount > 0 and pos.avg_cost > 0:
            current_price = get_current_data()[sec].last_price
            loss_pct = (current_price - pos.avg_cost) / pos.avg_cost
            if loss_pct <= -g.hard_stop_loss_pct:
                log.info(
                    f"🔻 硬止损触发：{sec} {get_name(sec)} 成本{pos.avg_cost:.3f} "
                    f"现价{current_price:.3f} 亏损{loss_pct*100:.2f}%")
                if smart_order_target_value(sec, 0, context):
                    log.info(f"📤 硬止损卖出：{sec} {get_name(sec)}")
    log.info("========== 硬止损检查完成 ==========")


# ==================== 震荡期机制 ====================
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
    except:
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


def get_risk_benchmark_state(context):
    """获取风险基准的日线+盘中融合状态，用于震荡期判断"""
    required_days = max(g.ma_period, g.lookback_high_low_days)
    lookback = required_days + 30
    end_date = getattr(context, 'previous_date', None)
    if end_date is None:
        return None
    df = get_price(g.risk_benchmark, end_date=end_date, count=lookback,
                   frequency='daily', fields=['close', 'high', 'low'], panel=False)
    if df is None or len(df) < required_days:
        return None
    daily_close = df['close'].values.astype(float)
    daily_high = df['high'].values.astype(float)
    daily_low = df['low'].values.astype(float)
    current_price = float(daily_close[-1])
    intraday_high = current_price
    intraday_low = current_price
    data_source = '昨日日线'
    try:
        today = context.current_dt.date()
        minute_df = get_price(
            g.risk_benchmark, start_date=today, end_date=context.current_dt,
            frequency='1m', fields=['close', 'high', 'low'],
            panel=False, fill_paused=False
        )
        if minute_df is not None and not minute_df.empty:
            minute_close = minute_df['close'].dropna()
            minute_high = minute_df['high'].dropna()
            minute_low = minute_df['low'].dropna()
            if not minute_close.empty:
                current_price = float(minute_close.iloc[-1])
                intraday_high = float(minute_high.max()) if not minute_high.empty else current_price
                intraday_low = float(minute_low.min()) if not minute_low.empty else current_price
                data_source = '当日盘中'
    except Exception:
        pass
    if current_price <= 0:
        try:
            current_data = get_current_data()
            live_price = current_data[g.risk_benchmark].last_price
            if live_price is not None and live_price > 0:
                current_price = float(live_price)
                intraday_high = max(intraday_high, current_price)
                intraday_low = min(intraday_low, current_price)
                data_source = '实时快照'
        except Exception:
            current_price = float(daily_close[-1])
    close_series = np.append(daily_close, current_price)
    high_series = np.append(daily_high, max(intraday_high, current_price))
    low_series = np.append(daily_low, min(intraday_low, current_price))
    recent_high = np.max(high_series[-g.lookback_high_low_days:])
    recent_low = np.min(low_series[-g.lookback_high_low_days:])
    ma = np.mean(close_series[-g.ma_period:])
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


def is_fresh_stop_loss_signal(context):
    """判断止损信号是否仍在有效期内"""
    signal_date = getattr(g, 'stop_loss_triggered_date', None)
    if signal_date is None:
        return False
    today = context.current_dt.date()
    previous_date = getattr(context, 'previous_date', None)
    if signal_date == today:
        return True
    if previous_date is not None and signal_date == previous_date:
        return True
    g.stop_loss_triggered_today = False
    g.stop_loss_triggered_date = None
    return False


def init_range_bound_status(context):
    """首次运行时，根据历史数据判断当前是否处于震荡期"""
    if not g.enable_range_bound_mode:
        return
    log.info("【首次运行】初始化震荡期状态...")
    try:
        if context.previous_date is None:
            log.warning("【首次运行】无法获取前一个交易日，保持正常期")
            return
        end_date = context.previous_date
        lookback = max(g.ma_period, g.lookback_high_low_days) + 30
        df = get_price(g.risk_benchmark, end_date=end_date, count=lookback,
                       frequency='daily', fields=['close', 'high', 'low'], panel=False)
        if df is None or len(df) < max(g.ma_period, g.lookback_high_low_days):
            log.warning("【首次运行】数据不足，保持正常期")
            return
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        current_price = close[-1]
        if len(close) >= g.lookback_high_low_days:
            recent_high = np.max(high[-g.lookback_high_low_days:])
            recent_low = np.min(low[-g.lookback_high_low_days:])
        else:
            recent_high = np.max(high)
            recent_low = np.min(low)
        ma = np.mean(close[-g.ma_period:])
        bias = (current_price - ma) / ma if ma > 0 else 0
        rise_from_low = (current_price - recent_low) / recent_low if recent_low > 0 else 0
        current_rsi = calculate_rsi(close, period=14)
        should_enter = False
        signals = []
        if g.enable_bias_trigger and bias > g.bias_threshold:
            should_enter = True
            signals.append(f"乖离率{bias:.2%}>{g.bias_threshold:.0%}")
        if g.enable_rsi_trigger and current_rsi is not None and len(close) >= 15:
            prev_rsi = calculate_rsi(close[:-1], period=14)
            if prev_rsi is not None and prev_rsi > g.rsi_overbought and current_rsi < g.rsi_pullback:
                should_enter = True
                signals.append(f"RSI超买回落{prev_rsi:.1f}->{current_rsi:.1f}")
        if should_enter:
            g.current_filter = '震荡期'
            g.risk_state = '震荡期'
            g.range_bound_start_date = end_date
            g.range_bound_days_count = 0
            log.info(f"【首次运行】初始化进入震荡期: {'; '.join(signals)}")
        else:
            g.current_filter = '正常期'
            g.risk_state = '正常期'
            if len(close) >= g.lookback_high_low_days:
                g.previous_drawdown = (recent_high - current_price) / recent_high if recent_high > 0 else 0
            else:
                g.previous_drawdown = 0
            g.previous_rsi = current_rsi
            rsi_str = f"{current_rsi:.1f}" if current_rsi is not None else "N/A"
            log.info(f"【首次运行】初始状态: 正常期, 乖离率: {bias:.2%}, RSI: {rsi_str}")
    except Exception as e:
        log.warning(f"【首次运行】初始化震荡期状态异常: {e}，保持正常期")


def check_and_exit_range_bound_mode(context):
    """检查是否需要退出震荡期"""
    if not g.enable_range_bound_mode:
        return
    if g.current_filter != '震荡期':
        return
    log.info("【震荡期退出检查】开始检测退出条件...")
    try:
        benchmark_state = get_risk_benchmark_state(context)
        if benchmark_state is None:
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
        log.info(
            f"【震荡期数据】当前价: {current_price:.3f}, 近{g.lookback_high_low_days}日高点: {recent_high:.3f}, 低点: {recent_low:.3f}")
        log.info(f"【震荡期数据】回撤: {current_drawdown:.2%}, 从低点涨幅: {rise_from_low:.2%}")
        if g.enable_low_point_rise_trigger:
            if rise_from_low >= g.low_point_rise_threshold:
                recovery_signals.append(f"从低点上涨{rise_from_low:.2%}")
        if g.enable_stable_signal_trigger:
            if current_price > ma:
                recovery_signals.append("价格站上均线")
            if len(close) >= 2 and close[-1] > close[-2]:
                recovery_signals.append("价格回升")
            if g.previous_drawdown is not None and current_drawdown < g.previous_drawdown:
                recovery_signals.append(f"回撤收窄")
            if current_rsi is not None and g.previous_rsi is not None and current_rsi > g.previous_rsi:
                recovery_signals.append(f"RSI回升")
            drawdown_safe = current_drawdown < g.drawdown_recovery
            if drawdown_safe:
                g.stable_days += 1
            else:
                g.stable_days = 0
        g.previous_drawdown = current_drawdown
        g.previous_rsi = current_rsi
        range_bound_days = 0
        if g.range_bound_start_date is not None:
            trade_days = get_trade_days(start_date=g.range_bound_start_date, end_date=context.current_dt.date())
            range_bound_days = len(trade_days) - 1
            if range_bound_days >= g.max_range_bound_days:
                recovery_signals.append(f"震荡期满({range_bound_days}天)")
        low_point_condition = g.enable_low_point_rise_trigger and rise_from_low >= g.low_point_rise_threshold
        stable_condition = False
        if g.enable_stable_signal_trigger:
            drawdown_safe = current_drawdown < g.drawdown_recovery
            stable_condition = drawdown_safe and len(recovery_signals) >= 2 and g.stable_days >= 2
        force_condition = range_bound_days >= g.max_range_bound_days
        should_recover = low_point_condition or stable_condition or force_condition
        if should_recover:
            can_switch = True
            if g.last_switch_date is not None:
                trade_days = get_trade_days(start_date=g.last_switch_date, end_date=context.current_dt.date())
                days_since = len(trade_days) - 1
                if days_since < g.filter_switch_cooldown:
                    can_switch = False
            if can_switch:
                g.current_filter = '正常期'
                g.risk_state = '正常期'
                g.last_switch_date = context.current_dt.date()
                g.range_bound_start_date = None
                g.range_bound_days_count = 0
                g.stable_days = 0
                log.info(f"【退出震荡期】切换回拉普拉斯滤波器: {'; '.join(recovery_signals)}")
        else:
            log.info("【震荡期退出检查】未满足退出条件，保持震荡期(高斯滤波器)")
    except Exception as e:
        log.warning(f"【震荡期退出检查】判断出错: {e}")


def check_and_enter_range_bound_mode(context):
    """检查是否需要进入震荡期"""
    if not g.enable_range_bound_mode:
        return
    log.info("【震荡期进入检查】开始检测...")
    stop_loss_signal_active = is_fresh_stop_loss_signal(context)
    can_switch = True
    if g.last_switch_date is not None:
        trade_days = get_trade_days(start_date=g.last_switch_date, end_date=context.current_dt.date())
        days_since = len(trade_days) - 1
        if days_since < g.filter_switch_cooldown:
            can_switch = False
    if g.current_filter == '震荡期':
        return
    if not can_switch:
        return
    risk_signals = []
    try:
        benchmark_state = get_risk_benchmark_state(context)
        if benchmark_state is not None:
            close = benchmark_state['close_series']
            current_price = benchmark_state['current_price']
            if g.enable_bias_trigger:
                ma = benchmark_state['ma']
                bias = (current_price - ma) / ma if ma > 0 else 0
                if bias > g.bias_threshold:
                    risk_signals.append(f"乖离率过大({bias:.2%})")
            if g.enable_rsi_trigger:
                current_rsi = benchmark_state['current_rsi']
                if len(close) >= 15 and current_rsi is not None:
                    prev_rsi = benchmark_state['previous_rsi']
                    if prev_rsi is not None:
                        if prev_rsi > g.rsi_overbought and current_rsi < g.rsi_pullback and current_rsi < prev_rsi:
                            risk_signals.append(f"RSI超买回落({prev_rsi:.1f}->{current_rsi:.1f})")
    except Exception as e:
        log.warning(f"【震荡期检查】获取基准数据异常: {e}")
    if g.enable_stop_loss_trigger and stop_loss_signal_active:
        risk_signals.append("盈利保护触发止损")
    if len(risk_signals) > 0:
        g.current_filter = '震荡期'
        g.risk_state = '震荡期'
        g.last_switch_date = context.current_dt.date()
        g.range_bound_start_date = context.current_dt.date()
        g.range_bound_days_count = 0
        g.stable_days = 0
        g.stop_loss_triggered_today = False
        g.stop_loss_triggered_date = None
        log.info(f"【进入震荡期】切换到高斯滤波器: {'; '.join(risk_signals)}")
    else:
        log.info("【震荡期检查】未满足进入条件，保持正常期(拉普拉斯滤波器)")


def check_range_bound(context):
    """震荡期检查入口（13:55定时调度，在卖出前执行）"""
    if not g.enable_range_bound_mode:
        return
    log.info("========== 震荡期检查开始 ==========")
    log.info(f"当前状态: {g.current_filter}")
    check_and_exit_range_bound_mode(context)
    check_and_enter_range_bound_mode(context)
    log.info(f"检查后状态: {g.current_filter}")
    g.rankings_cache = {'date': None, 'data': None}
    log.info("========== 震荡期检查完成 ==========")


def reset_range_bound_daily(context):
    """收盘后重置震荡期相关的每日标志"""
    if g.current_filter == '震荡期' and g.range_bound_start_date is not None:
        trade_days = get_trade_days(start_date=g.range_bound_start_date, end_date=context.current_dt.date())
        g.range_bound_days_count = len(trade_days) - 1
        log.info(f"震荡期已持续 {g.range_bound_days_count} 个交易日")


# ==================== 核心计算模块 ====================
def get_cached_rankings(context):
    """获取缓存的股票排名，保证同一交易日内多次调用结果一致"""
    today = context.current_dt.date()
    if g.rankings_cache['date'] != today:
        log.info("重新计算股票排名...")
        ranked = get_ranked_stocks(context)
        g.rankings_cache = {'date': today, 'data': ranked}
    else:
        log.debug("使用缓存的股票排名")
    return g.rankings_cache['data']


def get_ranked_stocks(context):
    """
    计算所有股票的动量得分，应用所有过滤条件，返回按得分降序的列表
    """
    stock_metrics = []
    current_data = get_current_data()

    for stock in g.stock_pool:
        # 停牌过滤
        if current_data[stock].paused:
            continue

        # ST过滤
        if g.enable_st_filter and is_st_stock(stock, current_data):
            continue

        # 涨停过滤（不追涨停）
        if g.enable_limit_up_filter and current_data[stock].last_price >= current_data[stock].high_limit:
            continue

        metrics = calculate_momentum_metrics(context, stock)
        if metrics is not None:
            if g.min_score_threshold < metrics['score'] < g.max_score_threshold:
                stock_metrics.append(metrics)

    stock_metrics.sort(key=lambda x: x['score'], reverse=True)
    return stock_metrics


def is_st_stock(stock, current_data=None):
    """判断是否为ST股票"""
    if current_data is None:
        current_data = get_current_data()
    name = current_data[stock].name
    if name and ('ST' in name or '*ST' in name):
        return True
    return False


def is_new_stock(stock, context):
    """判断是否为次新股（上市不满g.new_stock_days天）"""
    try:
        info = get_security_info(stock)
        if info is None:
            return True
        start_date = info.start_date
        today = context.current_dt.date()
        days_listed = (today - start_date).days
        return days_listed < g.new_stock_days
    except:
        return True


def calculate_momentum_metrics(context, stock):
    """
    计算单只股票的动量指标，应用所有过滤条件
    """
    try:
        name = get_name(stock)
        lookback = max(g.lookback_days, g.short_lookback_days) + 20
        prices = attribute_history(stock, lookback, '1d', ['close', 'high', 'volume'])
        if len(prices) < g.lookback_days:
            return None

        current_price = get_current_data()[stock].last_price
        price_series = np.append(prices["close"].values, current_price)

        # ===== 1. 盈利保护检查（排除） =====
        if check_profit_protection(stock, context):
            log.info(f"🚫 {stock} {name} 触发盈利保护，从排名中排除")
            return None

        # ===== 2. 次新股过滤 =====
        if g.enable_new_stock_filter and is_new_stock(stock, context):
            log.debug(f"{stock} {name} 为次新股，过滤")
            return None

        # ===== 3. 成交量过滤（排除） =====
        if g.enable_volume_check:
            vol_ratio = get_volume_ratio(context, stock)
            if vol_ratio is not None:
                annualized = get_annualized_returns(price_series, g.lookback_days)
                if annualized > g.volume_return_limit:
                    log.info(
                        f"📉 {stock} {name} 成交量放量{vol_ratio:.1f}倍，年化{annualized * 100:.1f}%，过滤")
                    return None

        # ===== 4. 短期动量过滤（排除） =====
        if len(price_series) >= g.short_lookback_days + 1:
            short_return = price_series[-1] / price_series[-(g.short_lookback_days + 1)] - 1
            short_annualized = (1 + short_return) ** (250 / g.short_lookback_days) - 1
        else:
            short_annualized = 0

        if g.use_short_momentum_filter and short_annualized < g.short_momentum_threshold:
            log.debug(f"{stock} {name} 短期动量{short_annualized * 100:.1f}% < 阈值，过滤")
            return None

        # ===== 5. 长期动量计算（得分） =====
        recent = price_series[-(g.lookback_days + 1):]
        y = np.log(recent)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.exp(slope * 250) - 1

        # R²（趋势稳定性）
        ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
        ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

        score = annualized_returns * r_squared

        # ===== 6. 近3日单日跌幅过滤（排除） =====
        if len(price_series) >= 4:
            day1 = price_series[-1] / price_series[-2]
            day2 = price_series[-2] / price_series[-3]
            day3 = price_series[-3] / price_series[-4]
            if min(day1, day2, day3) < g.loss:
                log.info(f"⚠️ {stock} {name} 近3日有单日跌幅超{(1 - g.loss) * 100:.1f}%，直接排除")
                return None

        # ===== 7. 动态滤波器过滤（震荡期机制） =====
        if g.enable_range_bound_mode and len(price_series) >= 10:
            try:
                laplace_values = laplace_filter(price_series, s=g.laplace_s_param)
                laplace_slope = laplace_values[-1] - laplace_values[-2] if len(laplace_values) >= 2 else 0
                passed_laplace = (current_price > laplace_values[-1] and laplace_slope > g.laplace_min_slope)
                g1_val, g2_val = gaussian_filter_last_two(price_series, sigma=g.gaussian_sigma)
                gaussian_slope = g1_val - g2_val
                passed_gaussian = (current_price > g1_val and gaussian_slope > g.gaussian_min_slope)
                if g.current_filter == '正常期':
                    passed_filter = passed_laplace
                    filter_name = '拉普拉斯'
                else:
                    passed_filter = passed_gaussian
                    filter_name = '高斯'
                if not passed_filter:
                    log.debug(f"{stock} {name} 未通过{filter_name}滤波器，过滤")
                    return None
            except Exception as e:
                log.debug(f"{stock} {name} 滤波器计算异常: {e}")

        return {
            'stock': stock,
            'stock_name': name,
            'annualized_returns': annualized_returns,
            'r_squared': r_squared,
            'score': score,
            'current_price': current_price,
            'short_annualized': short_annualized,
        }

    except Exception as e:
        log.warning(f"计算{stock} {get_name(stock)}时出错: {e}")
        return None


def get_annualized_returns(price_series, lookback_days):
    """计算加权年化收益率"""
    recent = price_series[-(lookback_days + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, _ = np.polyfit(x, y, 1, w=weights)
    return math.exp(slope * 250) - 1


def get_volume_ratio(context, security, lookback=None, threshold=None):
    """计算当日成交量与过去N日均量的比值"""
    lookback = lookback or g.volume_lookback
    threshold = threshold or g.volume_threshold
    try:
        name = get_name(security)
        hist = attribute_history(security, lookback, '1d', ['volume'])
        if hist.empty or len(hist) < lookback:
            return None
        avg_vol = hist['volume'].mean()

        today = context.current_dt.date()
        df_vol = get_price(security, start_date=today, end_date=context.current_dt,
                           frequency='1m', fields=['volume'], skip_paused=False, fq='pre')
        if df_vol is None or df_vol.empty:
            return None
        current_vol = df_vol['volume'].sum()
        ratio = current_vol / avg_vol if avg_vol > 0 else 0
        if ratio > threshold:
            log.debug(f"{security} {name} 成交量比{ratio:.2f} > {threshold}")
            return ratio
        return None
    except Exception as e:
        log.warning(f"成交量计算失败 {security}: {e}")
        return None


# ==================== 卖出模块 ====================
def check_positions(context):
    """每日开盘检查持仓状态，仅用于日志"""
    for sec in context.portfolio.positions:
        pos = context.portfolio.positions[sec]
        if pos.total_amount > 0:
            log.info(f"📊 持仓：{sec} {get_name(sec)} 数量{pos.total_amount} 成本{pos.avg_cost:.3f} 现价{pos.price:.3f}")


def is_market_bearish(context):
    """
    大盘择时检查：上证50指数是否跌破20日均线
    返回 True 表示大盘弱势（空仓/转防御），False 表示正常操作
    """
    if not g.enable_market_timing:
        return False
    try:
        lookback = g.market_timing_ma + 5
        hist = attribute_history(g.market_timing_index, lookback, '1d', ['close'])
        if len(hist) < g.market_timing_ma:
            return False
        ma = hist['close'][-g.market_timing_ma:].mean()
        current_price = get_current_data()[g.market_timing_index].last_price
        bearish = current_price < ma
        if bearish:
            log.info(f"🌧️ 大盘择时触发：{g.market_timing_index} 当前{current_price:.2f} < {g.market_timing_ma}日均线{ma:.2f}，进入防御模式")
        return bearish
    except Exception as e:
        log.warning(f"大盘择时判断异常: {e}")
        return False


def stock_sell_trade(context):
    """卖出不符合条件的持仓（排名变化 + 大盘择时强制空仓）"""
    log.info("========== 卖出操作开始 ==========")

    # 大盘择时：弱势时强制目标为防御ETF，卖出所有个股
    if is_market_bearish(context):
        target_stocks = []
        defensive_available = check_defensive_etf_available(context)
        if defensive_available:
            target_stocks = [g.defensive_etf]
        target_set = set(target_stocks)
        for sec in list(context.portfolio.positions.keys()):
            if sec not in g.stock_pool and sec != g.defensive_etf:
                continue
            if sec not in target_set:
                pos = context.portfolio.positions[sec]
                if pos.total_amount > 0:
                    if smart_order_target_value(sec, 0, context):
                        log.info(f"📤 大盘择时强制卖出：{sec} {get_name(sec)}")
        log.info("========== 卖出操作完成 ==========")
        return

    ranked = get_cached_rankings(context)
    target_stocks = []
    for m in ranked[:g.holdings_num]:
        if m['score'] >= g.min_score_threshold:
            target_stocks.append(m['stock'])

    defensive_available = check_defensive_etf_available(context)
    if not target_stocks and defensive_available:
        target_stocks = [g.defensive_etf]

    target_set = set(target_stocks)
    for sec in list(context.portfolio.positions.keys()):
        if sec not in g.stock_pool and sec != g.defensive_etf:
            continue
        if sec not in target_set:
            pos = context.portfolio.positions[sec]
            if pos.total_amount > 0:
                if smart_order_target_value(sec, 0, context):
                    log.info(f"📤 卖出不在目标的持仓：{sec} {get_name(sec)}")

    log.info("========== 卖出操作完成 ==========")


# ==================== 买入模块 ====================
def stock_buy_trade(context):
    """买入符合条件的股票，等权分配，按排名顺序逐个尝试直到凑够持仓数量"""
    log.info("========== 买入操作开始 ==========")

    # ---------- 大盘择时：弱势时直接转防御，不做个股排名 ----------
    if is_market_bearish(context):
        log.info("🌧️ 大盘处于弱势，跳过个股筛选，直接转入防御")
        target_stocks = []
        if check_defensive_etf_available(context):
            target_stocks = [g.defensive_etf]
            log.info(f"🛡️ 大盘择时防御模式：{g.defensive_etf} {get_name(g.defensive_etf)}")
        else:
            log.info("💤 大盘弱势且防御ETF不可用，保持空仓")
            return

        # 检查是否有持仓需要先卖出（卖出操作在13:10已执行，这里再次确认）
        current_pos = [s for s in context.portfolio.positions if s in g.stock_pool or s == g.defensive_etf]
        to_sell = [s for s in current_pos if s not in target_stocks]
        if to_sell:
            to_sell_names = [get_name(s) for s in to_sell]
            log.info(f"尚有持仓需要卖出：{list(zip(to_sell, to_sell_names))}，等待卖出完成再买入")
            return

        # 买入防御ETF
        total_val = context.portfolio.total_value
        target_per_stock = total_val / len(target_stocks)
        for stock in target_stocks:
            current_val = 0
            if stock in context.portfolio.positions:
                pos = context.portfolio.positions[stock]
                if pos.total_amount > 0:
                    current_val = pos.total_amount * pos.price
            if abs(current_val - target_per_stock) > target_per_stock * 0.05 or current_val == 0:
                if smart_order_target_value(stock, target_per_stock, context):
                    action = "买入" if current_val < target_per_stock else "调仓"
                    log.info(f"📦 {action}：{stock} {get_name(stock)} 目标金额{target_per_stock:.2f}")
        log.info("========== 买入操作完成 ==========")
        return

    ranked = get_cached_rankings(context)
    log.info("=== 股票排名前10 ===")
    for i, m in enumerate(ranked[:10]):
        log.info(
            f"排名{i + 1}: {m['stock']} {m['stock_name']} 得分{m['score']:.4f} "
            f"年化{m['annualized_returns'] * 100:.2f}% R²={m['r_squared']:.4f}")

    target_stocks = []
    for m in ranked:
        if len(target_stocks) >= g.holdings_num:
            break
        stock = m['stock']
        target_stocks.append(stock)
        log.info(f"🎯 目标股票 {len(target_stocks)}: {stock} {m['stock_name']} 得分{m['score']:.4f}")

    # ---------- 防御模式判断 ----------
    if not target_stocks:
        if check_defensive_etf_available(context):
            target_stocks = [g.defensive_etf]
            log.info(f"🛡️ 进入防御模式，选择防御ETF：{g.defensive_etf} {get_name(g.defensive_etf)}")
        else:
            log.info("💤 无目标股票且防御不可用，保持空仓")
            return

    # 检查是否有持仓需要先卖出
    current_pos = [s for s in context.portfolio.positions if s in g.stock_pool or s == g.defensive_etf]
    to_sell = [s for s in current_pos if s not in target_stocks]
    if to_sell:
        to_sell_names = [get_name(s) for s in to_sell]
        log.info(f"尚有持仓需要卖出：{list(zip(to_sell, to_sell_names))}，等待卖出完成再买入")
        return

    # 等权分配
    total_val = context.portfolio.total_value
    target_per_stock = total_val / len(target_stocks)

    for stock in target_stocks:
        current_val = 0
        if stock in context.portfolio.positions:
            pos = context.portfolio.positions[stock]
            if pos.total_amount > 0:
                current_val = pos.total_amount * pos.price
        # 5%容差调仓
        if abs(current_val - target_per_stock) > target_per_stock * 0.05 or current_val == 0:
            if smart_order_target_value(stock, target_per_stock, context):
                action = "买入" if current_val < target_per_stock else "调仓"
                log.info(f"📦 {action}：{stock} {get_name(stock)} 目标金额{target_per_stock:.2f}")

    log.info("========== 买入操作完成 ==========")


# ==================== 辅助函数 ====================
def get_name(security):
    """获取证券名称，带异常处理"""
    try:
        return get_current_data()[security].name
    except:
        return "未知"


def check_defensive_etf_available(context):
    """检查防御ETF是否可交易（未停牌、未涨跌停）"""
    data = get_current_data()
    etf = g.defensive_etf
    if data[etf].paused:
        log.debug(f"防御ETF {etf} 停牌")
        return False
    if data[etf].last_price >= data[etf].high_limit:
        log.debug(f"防御ETF {etf} 涨停")
        return False
    if data[etf].last_price <= data[etf].low_limit:
        log.debug(f"防御ETF {etf} 跌停")
        return False
    return True


def smart_order_target_value(security, target_value, context):
    """
    智能下单：根据目标市值调整持仓，处理停牌、涨跌停、最小交易金额、T+1
    """
    data = get_current_data()
    name = get_name(security)

    if data[security].paused:
        log.info(f"{security} {name} 停牌，跳过")
        return False

    price = data[security].last_price
    if price == 0:
        log.info(f"{security} {name} 当前价格0，跳过")
        return False

    target_amount = int(target_value / price)
    # 按100股整数倍调整
    target_amount = (target_amount // 100) * 100
    if target_amount <= 0 and target_value > 0:
        target_amount = 100

    cur_pos = context.portfolio.positions.get(security, None)
    cur_amount = cur_pos.total_amount if cur_pos else 0
    diff = target_amount - cur_amount

    # 根据交易方向检查涨跌停
    if diff > 0:  # 买入
        if data[security].last_price >= data[security].high_limit:
            log.info(f"{security} {name} 涨停，跳过买入")
            return False
    elif diff < 0:  # 卖出
        if data[security].last_price <= data[security].low_limit:
            log.info(f"{security} {name} 跌停，跳过卖出")
            return False

    # 最小交易金额检查
    trade_val = abs(diff) * price
    if 0 < trade_val < g.min_money:
        log.info(f"{security} {name} 交易金额{trade_val:.2f} < {g.min_money}，跳过")
        return False

    # T+1处理
    if diff < 0:
        closeable = cur_pos.closeable_amount if cur_pos else 0
        if closeable == 0:
            log.info(f"{security} {name} 当天买入不可卖出")
            return False
        diff = -min(abs(diff), closeable)

    if diff != 0:
        order_result = order(security, diff)
        if order_result:
            log.info(f"{'📥 买入' if diff > 0 else '📤 卖出'} {security} {name} 数量{abs(diff)} 价格{price:.3f}")
            return True
        else:
            log.warning(f"下单失败: {security} {name} 数量{diff}")
            return False
    return False
