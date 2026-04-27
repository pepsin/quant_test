"""
ETF-Clone 策略 Mock 测试框架

用于离线验证策略的核心过滤/排名逻辑，无需聚宽平台。
可生成各类 mock 价格序列（趋势、震荡、慢牛、暴跌等），
观察参数调整对过滤结果的影响。
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

# ============ 策略配置 ============
@dataclass
class StrategyConfig:
    """纯数据版策略参数，与原策略 g 对象对应"""
    # 长期动量
    lookback_days: int = 25

    # 短期动量（旧默认值，作为数据不足时的回退）
    short_lookback_days: int = 10
    short_momentum_threshold: float = 0.0

    # 波动率自适应（新增）
    short_volatility_high: float = 0.30      # 高于此值视为高波动
    short_volatility_low: float = 0.15       # 低于此值视为低波动
    short_lookback_high_vol: int = 10        # 高波动品种回看天数
    short_lookback_mid_vol: int = 15         # 中波动品种回看天数
    short_lookback_low_vol: int = 25         # 低波动品种回看天数

    # 近3日跌幅过滤
    loss: float = 0.97                       # 单日跌幅超3%排除

    # 滤波器参数
    laplace_s_param: float = 0.05
    laplace_min_slope: float = 0.001
    gaussian_sigma: float = 1.2
    gaussian_min_slope: float = 0.002

    # 震荡期状态
    enable_range_bound_mode: bool = True
    current_filter: str = '正常期'            # '正常期' 或 '震荡期'


# ============ Mock 价格序列生成器 ============
class MockPriceGenerator:
    """生成各种市场形态的 mock 日线价格序列"""

    @staticmethod
    def trend_up(days: int = 60, annual_return: float = 0.20,
                 volatility: float = 0.15, start_price: float = 1.0,
                 seed: Optional[int] = None) -> np.ndarray:
        """趋势上涨行情"""
        if seed is not None:
            np.random.seed(seed)
        daily_return = annual_return / 250
        daily_vol = volatility / math.sqrt(250)
        returns = np.random.normal(daily_return, daily_vol, days)
        prices = start_price * np.cumprod(1 + returns)
        return prices

    @staticmethod
    def slow_bull(days: int = 60, annual_return: float = 0.28,
                  volatility: float = 0.10, start_price: float = 1.0,
                  seed: Optional[int] = None) -> np.ndarray:
        """
        2017年式慢牛：低波动、缓慢、持续上涨
        波动率约10%，年化收益约28%
        """
        if seed is not None:
            np.random.seed(seed)
        daily_return = annual_return / 250
        daily_vol = volatility / math.sqrt(250)
        returns = np.random.normal(daily_return, daily_vol, days)
        prices = start_price * np.cumprod(1 + returns)
        return prices

    @staticmethod
    def choppy(days: int = 60, amplitude: float = 0.05,
               start_price: float = 1.0, seed: Optional[int] = None) -> np.ndarray:
        """震荡行情：无明显趋势，上下波动"""
        if seed is not None:
            np.random.seed(seed)
        t = np.arange(days)
        prices = start_price * (1 + amplitude * np.sin(t * 2 * np.pi / 20))
        noise = np.random.normal(0, 0.005, days)
        prices = prices + noise
        # 确保价格始终为正
        prices = np.maximum(prices, start_price * 0.5)
        return prices

    @staticmethod
    def crash_and_recover(days: int = 60, start_price: float = 1.0,
                          seed: Optional[int] = None) -> np.ndarray:
        """暴跌后反弹：前1/3涨，中1/3暴跌，后1/3反弹"""
        if seed is not None:
            np.random.seed(seed)
        prices = np.ones(days) * start_price
        crash_day = days // 3
        for i in range(1, crash_day):
            prices[i] = prices[i - 1] * (1 + np.random.normal(0.001, 0.01))
        for i in range(crash_day, crash_day * 2):
            prices[i] = prices[i - 1] * (1 + np.random.normal(-0.005, 0.02))
        for i in range(crash_day * 2, days):
            prices[i] = prices[i - 1] * (1 + np.random.normal(0.002, 0.015))
        return prices

    @staticmethod
    def with_pullback(prices: np.ndarray, pullback_days: int = 10,
                      pullback_pct: float = 0.015) -> np.ndarray:
        """
        在已有价格序列末端叠加一段小幅回调
        用于模拟"长期趋势完好，但近N天小幅回调"的场景
        """
        prices = prices.copy()
        n = len(prices)
        if pullback_days >= n:
            pullback_days = n // 2
        base = prices[n - pullback_days - 1]
        # 线性回调：从 base 跌到 base * (1 - pullback_pct)
        prices[n - pullback_days:] = base * np.linspace(
            1.0, 1.0 - pullback_pct, pullback_days
        )
        return prices


# ============ 核心计算逻辑（剥离聚宽依赖） ============
def laplace_filter(price: np.ndarray, s: float = 0.05) -> np.ndarray:
    """拉普拉斯滤波器（正常期使用）"""
    alpha = 1 - np.exp(-s)
    L = np.zeros(len(price))
    L[0] = price[0]
    for t in range(1, len(price)):
        L[t] = alpha * price[t] + (1 - alpha) * L[t - 1]
    return L


def gaussian_filter_last_two(price: np.ndarray, sigma: float = 1.2) -> Tuple[float, float]:
    """仅计算高斯滤波最后两个点（震荡期使用）"""
    n = len(price)
    if n < 2:
        return 0.0, 0.0
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


def calculate_metrics(
    price_history: np.ndarray,
    current_price: float,
    config: StrategyConfig,
    etf_name: str = "ETF"
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    纯计算版 momentum metrics。

    Args:
        price_history: 历史日线收盘价，不含当天（长度应 >= lookback_days + 5）
        current_price: 当前最新价（模拟盘中或收盘价）
        config: 策略配置
        etf_name: ETF名称，用于日志

    Returns:
        (metrics_dict, None)  或  (None, reason_string)
    """
    # 构造含当天的完整价格序列（与原策略逻辑一致）
    price_series = np.append(price_history, current_price)

    # ---------- 1. 短期动量过滤（动态 lookback） ----------
    vol_window = min(30, len(price_series) - 1)
    if vol_window >= 5:
        vol_returns = np.diff(np.log(price_series[-(vol_window + 1):]))
        volatility = float(np.std(vol_returns) * np.sqrt(250))
        if volatility > config.short_volatility_high:
            dynamic_short_lb = config.short_lookback_high_vol
            vol_tag = '高波'
        elif volatility < config.short_volatility_low:
            dynamic_short_lb = config.short_lookback_low_vol
            vol_tag = '低波'
        else:
            dynamic_short_lb = config.short_lookback_mid_vol
            vol_tag = '中波'
    else:
        dynamic_short_lb = config.short_lookback_days
        volatility = 0.0
        vol_tag = '默认'

    if len(price_series) >= dynamic_short_lb + 1:
        short_return = price_series[-1] / price_series[-(dynamic_short_lb + 1)] - 1
        short_annualized = (1 + short_return) ** (250 / dynamic_short_lb) - 1
    else:
        short_annualized = 0.0

    if short_annualized < config.short_momentum_threshold:
        return None, (
            f"[{vol_tag}]短期动量{short_annualized * 100:.1f}%"
            f"({dynamic_short_lb}天) < 阈值{config.short_momentum_threshold * 100:.1f}%，过滤"
        )

    # ---------- 2. 长期动量计算（得分） ----------
    if len(price_series) < config.lookback_days + 1:
        return None, f"历史数据不足({len(price_series)} < {config.lookback_days + 1})"

    recent = price_series[-(config.lookback_days + 1):]
    y = np.log(recent)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    annualized_returns = math.exp(slope * 250) - 1

    ss_res = np.sum(weights * (y - (slope * x + intercept)) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    score = annualized_returns * r_squared

    # ---------- 3. 近3日单日跌幅过滤 ----------
    if len(price_series) >= 4:
        day1 = price_series[-1] / price_series[-2]
        day2 = price_series[-2] / price_series[-3]
        day3 = price_series[-3] / price_series[-4]
        if min(day1, day2, day3) < config.loss:
            return None, (
                f"近3日有单日跌幅超{(1 - config.loss) * 100:.1f}%，过滤"
            )

    # ---------- 4. 动态滤波器过滤（震荡期机制） ----------
    if config.enable_range_bound_mode and len(price_series) >= 10:
        laplace_values = laplace_filter(price_series, s=config.laplace_s_param)
        laplace_slope = laplace_values[-1] - laplace_values[-2] if len(laplace_values) >= 2 else 0
        passed_laplace = (
            current_price > laplace_values[-1]
            and laplace_slope > config.laplace_min_slope
        )
        g1_val, g2_val = gaussian_filter_last_two(price_series, sigma=config.gaussian_sigma)
        gaussian_slope = g1_val - g2_val
        passed_gaussian = (
            current_price > g1_val
            and gaussian_slope > config.gaussian_min_slope
        )

        if config.current_filter == '正常期':
            passed_filter = passed_laplace
            filter_name = '拉普拉斯'
        else:
            passed_filter = passed_gaussian
            filter_name = '高斯'

        if not passed_filter:
            return None, f"未通过{filter_name}滤波器({config.current_filter})，过滤"

    return {
        'etf_name': etf_name,
        'annualized_returns': annualized_returns,
        'r_squared': r_squared,
        'score': score,
        'volatility': volatility,
        'vol_tag': vol_tag,
        'short_annualized': short_annualized,
        'dynamic_short_lb': dynamic_short_lb,
    }, None


# ============ 测试用例 ============
def test_single(
    name: str,
    prices: np.ndarray,
    config: StrategyConfig,
    description: str = ""
) -> None:
    """运行单个场景测试并打印结果"""
    print(f"\n{'=' * 60}")
    print(f"场景: {name}")
    if description:
        print(f"说明: {description}")
    print(f"价格序列: {len(prices)}天")

    # 模拟原策略的数据结构：history 不含当天，current_price 是最后一根
    result, reason = calculate_metrics(
        price_history=prices[:-1],
        current_price=float(prices[-1]),
        config=config,
        etf_name=name
    )

    if result is None:
        print(f"结果: ❌ 被过滤 → {reason}")
    else:
        print(f"结果: ✅ 通过")
        print(f"  波动率: {result['vol_tag']} (σ={result['volatility']:.2%})")
        print(f"  短期回看: {result['dynamic_short_lb']}天")
        print(f"  短期动量: {result['short_annualized'] * 100:.1f}%")
        print(f"  长期年化: {result['annualized_returns'] * 100:.1f}%")
        print(f"  R²: {result['r_squared']:.4f}")
        print(f"  得分: {result['score']:.4f}")


def test_parameter_sweep(
    name: str,
    prices: np.ndarray,
    param_name: str,
    param_values: List[Any],
    base_config: Optional[StrategyConfig] = None
) -> None:
    """
    参数扫描：对同一个价格序列，测试不同参数值的效果
    """
    print(f"\n{'=' * 60}")
    print(f"参数扫描: {name}")
    print(f"扫描参数: {param_name}")
    print(f"{'值':>10} | {'结果':>6} | {'短期动量':>10} | {'长期年化':>10} | {'得分':>8} | {'过滤原因'}")
    print("-" * 90)

    for val in param_values:
        config = base_config or StrategyConfig()
        setattr(config, param_name, val)

        result, reason = calculate_metrics(
            price_history=prices[:-1],
            current_price=float(prices[-1]),
            config=config,
            etf_name=name
        )

        if result is None:
            print(f"{str(val):>10} | {'❌':>6} | {'--':>10} | {'--':>10} | {'--':>8} | {reason}")
        else:
            print(
                f"{str(val):>10} | {'✅':>6} | "
                f"{result['short_annualized'] * 100:>9.1f}% | "
                f"{result['annualized_returns'] * 100:>9.1f}% | "
                f"{result['score']:>8.4f} | {'通过'}"
            )


def main():
    print("=" * 60)
    print("ETF-Clone 策略 Mock 测试框架")
    print("=" * 60)

    config = StrategyConfig()

    # ---------- 场景测试 ----------
    print("\n【一、场景测试】")

    # 1. 2017年式慢牛（低波动缓慢上涨）
    prices_slow = MockPriceGenerator.slow_bull(
        days=60, annual_return=0.28, volatility=0.10, seed=42
    )
    test_single(
        "2017慢牛(纳指模拟)", prices_slow, config,
        "低波动(约10%)缓慢上涨，2017年纳指特征"
    )

    # 2. 高波动上涨（创业板风格）
    prices_high = MockPriceGenerator.trend_up(
        days=60, annual_return=0.20, volatility=0.30, seed=42
    )
    test_single(
        "高波动上涨(创业板模拟)", prices_high, config,
        "高波动(约30%)上涨，波动率自适应应切换为10天回看"
    )

    # 3. 震荡行情
    prices_choppy = MockPriceGenerator.choppy(days=60, seed=42)
    test_single(
        "震荡行情", prices_choppy, config,
        "无明显趋势，上下震荡，预期被滤波器过滤"
    )

    # 4. 暴跌后反弹
    prices_crash = MockPriceGenerator.crash_and_recover(days=60, seed=42)
    test_single(
        "暴跌后反弹", prices_crash, config,
        "前20天涨，中20天暴跌，后20天反弹"
    )

    # 5. 慢牛 + 末端小幅回调（复现2017年1月问题）
    prices_pullback = MockPriceGenerator.with_pullback(
        prices_slow.copy(), pullback_days=10, pullback_pct=0.015
    )
    test_single(
        "慢牛+10日小幅回调(1.5%)", prices_pullback, config,
        "模拟2017年1月：长期涨28%，但近10天回调1.5%，旧10天逻辑会被过滤"
    )

    # 6. 正常期 vs 震荡期滤波器对比
    print(f"\n{'=' * 60}")
    print("【二、正常期 vs 震荡期滤波器对比】")
    prices_compare = MockPriceGenerator.slow_bull(
        days=60, annual_return=0.15, volatility=0.12, seed=123
    )

    config_normal = StrategyConfig(current_filter='正常期')
    config_range = StrategyConfig(current_filter='震荡期')

    _, reason_n = calculate_metrics(
        prices_compare[:-1], float(prices_compare[-1]), config_normal, "正常期"
    )
    _, reason_r = calculate_metrics(
        prices_compare[:-1], float(prices_compare[-1]), config_range, "震荡期"
    )

    print(f"正常期(拉普拉斯): {'✅ 通过' if reason_n is None else '❌ ' + reason_n}")
    print(f"震荡期(高斯):     {'✅ 通过' if reason_r is None else '❌ ' + reason_r}")

    # ---------- 参数扫描 ----------
    print("\n【三、参数扫描】")

    # 扫描 short_lookback_low_vol 对"慢牛+回调"的影响
    test_parameter_sweep(
        name="慢牛+回调场景",
        prices=prices_pullback,
        param_name="short_lookback_low_vol",
        param_values=[10, 15, 20, 25, 30, 40],
        base_config=config
    )

    # 扫描 laplace_min_slope 对慢牛的影响
    test_parameter_sweep(
        name="慢牛场景(拉普拉斯斜率)",
        prices=prices_slow,
        param_name="laplace_min_slope",
        param_values=[0, 0.0005, 0.001, 0.002, 0.005],
        base_config=config
    )

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
