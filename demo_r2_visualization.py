"""
R² 可视化演示脚本
----------------
用几组模拟价格序列，直观展示：
1. 为什么 "涨得多" 不等于 "值得买"
2. R² 如何衡量价格走势的 "线性程度"
3. 策略最终得分 = 年化收益 × R² 的惩罚效果

运行：python demo_r2_visualization.py
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

LOOKBACK_DAYS = 25


def calc_metrics(price_series):
    """
    复刻策略中的核心计算逻辑（加权对数回归）
    """
    y = np.log(price_series)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))

    # 加权线性回归: y = slope * x + intercept
    slope, intercept = np.polyfit(x, y, 1, w=weights)

    # 年化收益率
    annualized = math.exp(slope * 250) - 1

    # R²（加权版本）
    y_pred = slope * x + intercept
    ss_res = np.sum(weights * (y - y_pred) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    # 最终得分
    score = annualized * r_squared

    return {
        'slope': slope,
        'intercept': intercept,
        'annualized': annualized,
        'r_squared': r_squared,
        'score': score,
        'y_pred': y_pred,
        'log_prices': y,
    }


def generate_scenarios():
    """生成 4 种典型的价格走势"""
    n = LOOKBACK_DAYS + 1  # 26 个点（含起点）
    t = np.arange(n)

    scenarios = []

    # ---------- 场景 1: 完美直线（R² ≈ 1.0）----------
    # 每天固定涨 0.3%，像国债一样稳
    p1 = 100 * np.exp(0.003 * t)
    scenarios.append((
        p1,
        "场景 A：完美直线型",
        "每天固定涨幅，走势极其规律\n类比：高信用债、货币基金"
    ))

    # ---------- 场景 2: 稳步上涨（高 R²，≈0.85+）----------
    # 趋势清晰的小幅波动，不要 cumsum 随机游走过猛
    noise2 = np.random.normal(0, 0.012, n)
    p2 = 100 * np.exp(0.004 * t + 0.3 * noise2)  # 不加 cumsum，避免漂移过强
    scenarios.append((
        p2,
        "场景 B：稳步上涨型（高 R²）",
        "趋势清晰，小碎步上涨，偶尔微调\n类比：强势蓝筹、黄金慢牛"
    ))

    # ---------- 场景 3: 波动上涨（中 R²，≈0.50）----------
    # 分段构造：涨 → 浅跌 → 涨，总体向上但中间有波折
    log_p3 = np.concatenate([
        np.linspace(4.605, 4.680, 10),  # 前 10 天：温和涨
        np.linspace(4.680, 4.640, 5),   # 中 5 天：浅跌
        np.linspace(4.640, 4.750, 11)   # 后 11 天：涨
    ]) + np.random.normal(0, 0.012, n)
    p3 = np.exp(log_p3)
    scenarios.append((
        p3,
        "场景 C：波动上涨型（中 R²）",
        "总体向上，但中间有大起大落\n类比：周期股、科技股震荡上行"
    ))

    # ---------- 场景 4: 大阳线拉起（低 R²，≈0.25）----------
    # 策略最警惕的情况：前面横盘/阴跌，最后 2 天暴力拉升
    p4 = np.ones(n) * 100
    p4 += np.random.normal(0, 1.5, n)  # 前面小幅震荡
    p4[-2:] = [105, 118]  # 最后两天突然涨停式拉升
    p4 = np.maximum.accumulate(p4)  # 确保非负
    scenarios.append((
        p4,
        "场景 D：大阳线拉起型（低 R²）[WARNING]",
        "前 23 天横盘/阴跌，最后 2 天暴力拉升\n类比：消息刺激、游资拉板、末日轮\n→ 动量高，但 R² 极低，策略会惩罚"
    ))

    return scenarios


def main():
    scenarios = generate_scenarios()
    n_scenes = len(scenarios)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(n_scenes, 3, figure=fig, width_ratios=[1.2, 1.2, 1])

    for row, (prices, title, desc) in enumerate(scenarios):
        metrics = calc_metrics(prices)

        # ---- 左图：价格走势 + 拟合线 ----
        ax_price = fig.add_subplot(gs[row, 0])
        days = np.arange(len(prices))

        # 画真实价格
        ax_price.plot(days, prices, 'o-', color='#2E86AB', linewidth=2, markersize=4, label='真实价格')

        # 画对数回归的指数拟合线 (转回价格空间)
        fitted_prices = np.exp(metrics['y_pred'])
        ax_price.plot(days, fitted_prices, '--', color='#F24236', linewidth=2, label='趋势拟合线')

        # 标注残差区域（阴影表示偏离程度）
        ax_price.fill_between(days, prices, fitted_prices, alpha=0.15, color='gray', label='偏离区域')

        ax_price.set_title(title, fontsize=13, fontweight='bold', loc='left')
        ax_price.set_xlabel('交易日')
        ax_price.set_ylabel('价格')
        ax_price.legend(loc='upper left', fontsize=9)
        ax_price.grid(True, alpha=0.3)

        # 在图上写指标
        textstr = (
            f"年化收益: {metrics['annualized']*100:+.1f}%\n"
            f"R² (线性度): {metrics['r_squared']:.3f}\n"
            f"最终得分: {metrics['score']:.3f}"
        )
        ax_price.text(
            0.98, 0.05, textstr,
            transform=ax_price.transAxes,
            fontsize=11, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        # ---- 中图：对数坐标下的回归（更直观） ----
        ax_log = fig.add_subplot(gs[row, 1])
        ax_log.plot(days, metrics['log_prices'], 'o-', color='#2E86AB', linewidth=2, markersize=4, label='log(价格)')
        ax_log.plot(days, metrics['y_pred'], '--', color='#F24236', linewidth=2, label='加权回归线')
        ax_log.fill_between(days, metrics['log_prices'], metrics['y_pred'], alpha=0.15, color='gray')

        ax_log.set_title('对数空间视角（线性回归在此进行）', fontsize=13, loc='left')
        ax_log.set_xlabel('交易日')
        ax_log.set_ylabel('log(价格)')
        ax_log.legend(loc='upper left', fontsize=9)
        ax_log.grid(True, alpha=0.3)

        # ---- 右图：残差分布 ----
        ax_res = fig.add_subplot(gs[row, 2])
        residuals = metrics['log_prices'] - metrics['y_pred']
        colors = ['#28A745' if r >= 0 else '#DC3545' for r in residuals]
        ax_res.bar(days, residuals, color=colors, alpha=0.7)
        ax_res.axhline(0, color='black', linewidth=0.8)
        ax_res.set_title('残差（偏离趋势的程度）', fontsize=13, loc='left')
        ax_res.set_xlabel('交易日')
        ax_res.set_ylabel('残差')
        ax_res.grid(True, alpha=0.3)

        # 描述文字
        ax_res.text(
            0.5, -0.15, desc,
            transform=ax_res.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            wrap=True
        )

    fig.suptitle(
        'R² 可视化：为什么 "涨得多" 不等于 "值得买"\n'
        '策略打分 = 年化收益 × R²，低 R² 会被大幅惩罚',
        fontsize=16, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('r2_visualization.png', dpi=150, bbox_inches='tight')
    print("图片已保存: r2_visualization.png")
    # plt.show()  # CLI 环境下注释掉，避免阻塞

    # ---------- 额外：打印对比表格 ----------
    print("\n" + "="*70)
    print("四组场景的核心指标对比")
    print("="*70)
    print(f"{'场景':<20} {'年化收益':>12} {'R²':>10} {'最终得分':>12} {'策略态度':>12}")
    print("-"*70)
    for (prices, title, desc) in scenarios:
        m = calc_metrics(prices)
        short_title = title.split("：")[1]
        attitude = "[PASS] 优选" if m['r_squared'] > 0.8 else ("[WARN] 谨慎" if m['r_squared'] > 0.5 else "[DROP] 排除")
        print(f"{short_title:<20} {m['annualized']*100:>+11.1f}% {m['r_squared']:>10.3f} {m['score']:>12.3f} {attitude:>12}")
    print("="*70)


if __name__ == '__main__':
    main()
