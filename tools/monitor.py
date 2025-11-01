#!/usr/bin/env python3
"""
训练日志可视化 Gradio 应用
用法: python training_monitor_gradio.py
然后在浏览器打开显示的URL
"""

import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import warnings


# 设置matplotlib样式
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['axes.unicode_minus'] = False

# 禁用警告
warnings.filterwarnings('ignore', category=UserWarning)

def scan_training_tasks() -> List[str]:
    """扫描所有训练任务目录"""
    if not ROOT_DIR.exists():
        return []
    
    tasks = []
    for item in ROOT_DIR.iterdir():
        if item.is_dir():
            rollout_dir = item / "rollout_log"
            if rollout_dir.exists():
                tasks.append(item.name)
    
    return sorted(tasks)

def load_rollout_data(task_name: str) -> pd.DataFrame:
    """加载rollout数据"""
    rollout_dir = ROOT_DIR / task_name / "rollout_log"
    
    data = []
    for jsonl_file in sorted(rollout_dir.glob("*.jsonl")):
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    data.append({
                        'step': item.get('step'),
                        'score': item.get('score'),
                        'acc': item.get('acc'),
                        'file': jsonl_file.stem
                    })
                except:
                    continue
    
    df = pd.DataFrame(data)
    df = df.dropna(subset=['score'])
    return df

def load_validation_data(task_name: str) -> pd.DataFrame:
    """加载validation数据 - 从文件名提取step"""
    val_dir = ROOT_DIR / task_name / "validation_log"
    
    if not val_dir.exists():
        return pd.DataFrame()
    
    data = []
    for jsonl_file in sorted(val_dir.glob("*.jsonl")):
        # 从文件名提取step
        try:
            file_step = int(jsonl_file.stem)
        except ValueError:
            print(f"⚠️  跳过无法解析的文件: {jsonl_file.name}")
            continue
            
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    data.append({
                        'step': file_step,
                        'score': item.get('score'),
                        'reward': item.get('reward'),
                        'acc': item.get('acc')
                    })
                except:
                    continue
    
    if data:
        df = pd.DataFrame(data)
        df_agg = df.groupby('step').agg({
            'score': 'mean',
            'reward': 'mean',
            'acc': 'mean'
        }).reset_index()
        return df_agg
    
    return pd.DataFrame()

def recommend_best_checkpoint(task_name: str, df: pd.DataFrame, val_df: pd.DataFrame = None, save_interval: int = 20) -> str:
    """推荐最佳checkpoint（只考虑实际保存的checkpoint）"""
    recommendation = []
    recommendation.append("=" * 60)
    recommendation.append("🏆 模型检查点推荐")
    recommendation.append("=" * 60)
    recommendation.append("")
    
    df_agg = df.groupby('step').agg({
        'score': ['mean', 'std', 'count']
    }).reset_index()
    df_agg.columns = ['step', 'score_mean', 'score_std', 'count']
    
    # ✅ 只保留能被save_interval整除的步数（实际保存的checkpoint）
    df_agg = df_agg[df_agg['step'] % save_interval == 0].copy()
    
    if df_agg.empty:
        recommendation.append("⚠️  未找到符合保存间隔的checkpoint数据")
        recommendation.append(f"   当前保存间隔: 每 {save_interval} 步")
        recommendation.append("")
        recommendation.append("=" * 60)
        return "\n".join(recommendation)
    
    recommendation.append(f"ℹ️  Checkpoint保存间隔: 每 {save_interval} 步")
    recommendation.append(f"   可用的checkpoint步数: {sorted(df_agg['step'].tolist())}")
    recommendation.append("")
    
    # 计算综合得分（考虑均值和稳定性）
    df_agg['composite_score'] = df_agg['score_mean'] - 0.2 * df_agg['score_std']
    
    # 策略1：最高平均分数
    best_avg_step = df_agg.loc[df_agg['score_mean'].idxmax()]
    
    # 策略2：最高综合得分（平衡均值和稳定性）
    best_composite_step = df_agg.loc[df_agg['composite_score'].idxmax()]
    
    # 策略3：最后一个checkpoint
    last_step = df_agg.iloc[-1]
    
    recommendation.append("📊 候选检查点分析：")
    recommendation.append("")
    
    recommendation.append("1️⃣  最高平均分数模型：")
    recommendation.append(f"   Step: {int(best_avg_step['step'])}")
    recommendation.append(f"   平均分数: {best_avg_step['score_mean']:.4f}")
    recommendation.append(f"   标准差:   {best_avg_step['score_std']:.4f}")
    recommendation.append(f"   样本数:   {int(best_avg_step['count'])}")
    recommendation.append("")
    
    recommendation.append("2️⃣  最稳定高分模型：")
    recommendation.append(f"   Step: {int(best_composite_step['step'])}")
    recommendation.append(f"   平均分数: {best_composite_step['score_mean']:.4f}")
    recommendation.append(f"   标准差:   {best_composite_step['score_std']:.4f}")
    recommendation.append(f"   综合得分: {best_composite_step['composite_score']:.4f}")
    recommendation.append("")
    
    recommendation.append("3️⃣  最新模型：")
    recommendation.append(f"   Step: {int(last_step['step'])}")
    recommendation.append(f"   平均分数: {last_step['score_mean']:.4f}")
    recommendation.append(f"   标准差:   {last_step['score_std']:.4f}")
    recommendation.append("")
    
    # 如果有验证集数据，加入验证集表现
    recommended_from_val = None
    if val_df is not None and not val_df.empty:
        # ✅ 同样只考虑保存的checkpoint
        val_df_filtered = val_df[val_df['step'] % save_interval == 0].copy()
        
        if not val_df_filtered.empty:
            recommendation.append("4️⃣  验证集最佳模型：")
            best_val_step = val_df_filtered.loc[val_df_filtered['score'].idxmax()]
            recommended_from_val = int(best_val_step['step'])
            recommendation.append(f"   Step: {recommended_from_val}")
            recommendation.append(f"   验证分数: {best_val_step['score']:.4f}")
            if 'acc' in val_df_filtered.columns:
                recommendation.append(f"   验证准确率: {best_val_step['acc']:.4f}")
            recommendation.append("")
    
    recommendation.append("=" * 60)
    recommendation.append("🎯 最终推荐：")
    recommendation.append("=" * 60)
    recommendation.append("")
    
    # 决策逻辑
    if recommended_from_val is not None:
        # 如果有验证集，优先考虑验证集表现
        recommended_step = recommended_from_val
        reason = "验证集表现最佳"
        
        recommendation.append(f"✅ 推荐使用 Step {recommended_step} 的模型")
        recommendation.append(f"   推荐理由: {reason}")
        
        val_perf = val_df[val_df['step'] == recommended_step]
        if not val_perf.empty:
            recommendation.append(f"   验证分数: {val_perf['score'].values[0]:.4f}")
        
        # 检查训练集对应的表现
        train_perf = df_agg[df_agg['step'] == recommended_step]
        if not train_perf.empty:
            recommendation.append(f"   训练分数: {train_perf['score_mean'].values[0]:.4f}")
        
    else:
        # 没有验证集，根据训练集决策
        if best_avg_step['step'] == best_composite_step['step']:
            recommended_step = int(best_avg_step['step'])
            reason = "训练集平均分数最高且稳定"
        else:
            # 如果相差不大，推荐综合得分最高的（更稳定）
            if best_avg_step['score_mean'] - best_composite_step['score_mean'] < 0.05:
                recommended_step = int(best_composite_step['step'])
                reason = "综合考虑分数和稳定性"
            else:
                recommended_step = int(best_avg_step['step'])
                reason = "训练集平均分数显著最高"
        
        recommendation.append(f"✅ 推荐使用 Step {recommended_step} 的模型")
        recommendation.append(f"   推荐理由: {reason}")
        
        rec_perf = df_agg[df_agg['step'] == recommended_step]
        recommendation.append(f"   平均分数: {rec_perf['score_mean'].values[0]:.4f}")
        recommendation.append(f"   标准差:   {rec_perf['score_std'].values[0]:.4f}")
    
    recommendation.append("")
    
    # 补充建议
    recommendation.append("💡 补充建议：")
    
    # 检查是否有过拟合迹象
    if val_df is not None and not val_df.empty:
        val_df_filtered = val_df[val_df['step'] % save_interval == 0]
        if not val_df_filtered.empty:
            train_val_gap = df_agg['score_mean'].mean() - val_df_filtered['score'].mean()
            if train_val_gap > 0.2:
                recommendation.append("   ⚠️  检测到训练集和验证集差距较大，建议:")
                recommendation.append("      - 考虑使用较早期的checkpoint")
                recommendation.append("      - 关注验证集表现而非训练集")
            else:
                recommendation.append("   ✅ 训练验证一致性良好，模型泛化能力好")
    
    # 检查训练趋势
    if len(df_agg) > 1:
        score_trend = df_agg['score_mean'].iloc[-1] - df_agg['score_mean'].iloc[0]
        if score_trend > 0.1:
            recommendation.append("   📈 训练仍在持续改进中，可以考虑:")
            recommendation.append("      - 继续训练更多步数")
            recommendation.append("      - 或使用当前最佳checkpoint")
        elif score_trend < -0.1:
            recommendation.append("   📉 后期训练有下降趋势，建议:")
            recommendation.append("      - 使用中期表现最好的checkpoint")
            recommendation.append("      - 检查训练配置和数据质量")
    
    # 稳定性建议
    if df_agg['score_std'].mean() > 1.0:
        recommendation.append("   🔄 模型输出波动较大，建议:")
        recommendation.append("      - 优先选择标准差较小的checkpoint")
        recommendation.append("      - 考虑调整采样温度或其他解码参数")
    
    # 显示所有可用checkpoint的排名
    recommendation.append("")
    recommendation.append("📋 所有可用Checkpoint排名（按平均分数）：")
    df_sorted = df_agg.sort_values('score_mean', ascending=False)
    for idx, row in df_sorted.head(5).iterrows():
        rank_symbol = "👑" if row['step'] == recommended_step else "  "
        recommendation.append(f"   {rank_symbol} Step {int(row['step']):>4}: {row['score_mean']:>8.4f} (std: {row['score_std']:.4f})")
    
    recommendation.append("")
    recommendation.append("=" * 60)
    recommendation.append("📁 推荐模型路径：")
    recommendation.append(f"   {ROOT_DIR / task_name / f'checkpoint-{recommended_step}'}")
    recommendation.append("=" * 60)
    
    return "\n".join(recommendation)

def analyze_score_trend(df: pd.DataFrame, val_df: pd.DataFrame = None) -> str:
    """分析Score趋势"""
    analysis = []
    analysis.append("=" * 60)
    analysis.append("📈 分数趋势分析")
    analysis.append("=" * 60)
    analysis.append("")
    
    df_agg = df.groupby('step')['score'].agg(['mean', 'std']).reset_index()
    
    first_score = df_agg['mean'].iloc[0]
    last_score = df_agg['mean'].iloc[-1]
    max_score = df_agg['mean'].max()
    min_score = df_agg['mean'].min()
    
    analysis.append(f"🎯 关键指标:")
    analysis.append(f"   初始分数: {first_score:>8.4f}")
    analysis.append(f"   最终分数: {last_score:>8.4f}")
    analysis.append(f"   最高分数: {max_score:>8.4f}")
    analysis.append(f"   最低分数: {min_score:>8.4f}")
    analysis.append("")
    
    improvement = last_score - first_score
    improvement_pct = (improvement / abs(first_score) * 100) if first_score != 0 else 0
    
    analysis.append(f"📊 整体趋势:")
    analysis.append(f"   分数变化: {improvement:>+8.4f}")
    analysis.append(f"   变化率:   {improvement_pct:>+8.1f}%")
    
    if improvement > 0.1:
        conclusion = "✅ 训练效果良好，分数持续提升"
    elif improvement > 0:
        conclusion = "🟢 训练稳步进行，分数略有提升"
    elif improvement > -0.1:
        conclusion = "🟡 分数基本稳定，建议观察后续趋势"
    else:
        conclusion = "⚠️  分数下降，建议检查训练配置"
    
    analysis.append(f"   结论: {conclusion}")
    analysis.append("")
    
    volatility = df_agg['std'].mean()
    analysis.append(f"📉 波动性分析:")
    analysis.append(f"   平均标准差: {volatility:.4f}")
    
    if volatility < 0.5:
        stability = "稳定性好"
    elif volatility < 1.0:
        stability = "稳定性中等"
    else:
        stability = "波动较大，建议检查数据质量"
    analysis.append(f"   评估: {stability}")
    analysis.append("")
    
    if val_df is not None and not val_df.empty:
        val_mean = val_df['score'].mean()
        train_mean = df_agg['mean'].mean()
        gap = train_mean - val_mean
        
        analysis.append(f"🔍 验证集对比:")
        analysis.append(f"   训练集平均: {train_mean:>8.4f}")
        analysis.append(f"   验证集平均: {val_mean:>8.4f}")
        analysis.append(f"   差距:       {gap:>+8.4f}")
        
        if abs(gap) < 0.1:
            analysis.append(f"   结论: ✅ 训练验证一致性好")
        elif gap > 0.2:
            analysis.append(f"   结论: ⚠️  可能存在过拟合")
        else:
            analysis.append(f"   结论: 🟢 表现正常")
    
    analysis.append("")
    analysis.append("=" * 60)
    
    return "\n".join(analysis)

def analyze_score_distribution(df: pd.DataFrame) -> str:
    """分析Score分布"""
    analysis = []
    analysis.append("=" * 60)
    analysis.append("📊 分数分布分析")
    analysis.append("=" * 60)
    analysis.append("")
    
    scores = df['score'].values
    
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    
    analysis.append(f"📈 分布特征:")
    analysis.append(f"   均值:   {mean_score:>8.4f}")
    analysis.append(f"   中位数: {median_score:>8.4f}")
    analysis.append(f"   标准差: {std_score:>8.4f}")
    analysis.append("")
    
    skewness = np.mean((scores - mean_score) ** 3) / (std_score ** 3) if std_score > 0 else 0
    analysis.append(f"🔄 偏度分析:")
    analysis.append(f"   偏度系数: {skewness:.4f}")
    
    if skewness > 0.5:
        skew_desc = "右偏（正偏），存在较多高分样本"
    elif skewness < -0.5:
        skew_desc = "左偏（负偏），存在较多低分样本"
    else:
        skew_desc = "接近对称分布"
    analysis.append(f"   特征: {skew_desc}")
    analysis.append("")
    
    q25 = np.percentile(scores, 25)
    q75 = np.percentile(scores, 75)
    iqr = q75 - q25
    
    analysis.append(f"📦 四分位数:")
    analysis.append(f"   Q25 (下四分位): {q25:>8.4f}")
    analysis.append(f"   Q75 (上四分位): {q75:>8.4f}")
    analysis.append(f"   IQR (四分位距): {iqr:>8.4f}")
    analysis.append("")
    
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = np.sum((scores < lower_bound) | (scores > upper_bound))
    outlier_pct = outliers / len(scores) * 100
    
    analysis.append(f"🎯 异常值检测:")
    analysis.append(f"   异常值数量: {outliers} ({outlier_pct:.2f}%)")
    
    if outlier_pct < 5:
        analysis.append(f"   结论: ✅ 数据质量良好")
    elif outlier_pct < 10:
        analysis.append(f"   结论: 🟡 存在少量异常值")
    else:
        analysis.append(f"   结论: ⚠️  异常值较多，建议检查数据")
    analysis.append("")
    
    pos_count = np.sum(scores > 0)
    neg_count = np.sum(scores <= 0)
    pos_ratio = pos_count / len(scores) * 100
    
    analysis.append(f"⚖️  正负样本分布:")
    analysis.append(f"   正样本: {pos_count} ({pos_ratio:.1f}%)")
    analysis.append(f"   负样本: {neg_count} ({100-pos_ratio:.1f}%)")
    
    if pos_ratio > 60:
        balance = "正样本占优，模型学习方向良好"
    elif pos_ratio > 40:
        balance = "正负样本较为均衡"
    else:
        balance = "负样本偏多，建议关注策略质量"
    analysis.append(f"   评估: {balance}")
    
    analysis.append("")
    analysis.append("=" * 60)
    
    return "\n".join(analysis)

def analyze_accuracy_trend(df: pd.DataFrame, val_df: pd.DataFrame = None) -> str:
    """分析准确率趋势"""
    analysis = []
    analysis.append("=" * 60)
    analysis.append("✅ 准确率趋势分析")
    analysis.append("=" * 60)
    analysis.append("")
    
    if 'acc' not in df.columns or df['acc'].isna().all():
        analysis.append("⚠️  未找到准确率数据")
        analysis.append("")
        analysis.append("=" * 60)
        return "\n".join(analysis)
    
    df_agg = df.groupby('step')['acc'].agg(['mean', 'std']).reset_index()
    
    first_acc = df_agg['mean'].iloc[0]
    last_acc = df_agg['mean'].iloc[-1]
    max_acc = df_agg['mean'].max()
    avg_acc = df_agg['mean'].mean()
    
    analysis.append(f"🎯 准确率指标:")
    analysis.append(f"   初始准确率: {first_acc:>8.4f}")
    analysis.append(f"   最终准确率: {last_acc:>8.4f}")
    analysis.append(f"   最高准确率: {max_acc:>8.4f}")
    analysis.append(f"   平均准确率: {avg_acc:>8.4f}")
    analysis.append("")
    
    improvement = last_acc - first_acc
    analysis.append(f"📈 提升情况:")
    analysis.append(f"   准确率提升: {improvement:>+8.4f}")
    
    if improvement > 0.1:
        conclusion = "✅ 准确率显著提升，模型学习效果好"
    elif improvement > 0.02:
        conclusion = "🟢 准确率稳步提升"
    elif improvement > -0.02:
        conclusion = "🟡 准确率基本稳定"
    else:
        conclusion = "⚠️  准确率下降，需要关注"
    
    analysis.append(f"   结论: {conclusion}")
    analysis.append("")
    
    if val_df is not None and not val_df.empty and 'acc' in val_df.columns:
        val_acc_mean = val_df['acc'].mean()
        train_acc_mean = df_agg['mean'].mean()
        gap = train_acc_mean - val_acc_mean
        
        analysis.append(f"🔍 训练验证对比:")
        analysis.append(f"   训练集准确率: {train_acc_mean:>8.4f}")
        analysis.append(f"   验证集准确率: {val_acc_mean:>8.4f}")
        analysis.append(f"   差距:         {gap:>+8.4f}")
        
        if abs(gap) < 0.05:
            analysis.append(f"   结论: ✅ 泛化能力良好")
        elif gap > 0.1:
            analysis.append(f"   结论: ⚠️  可能存在过拟合")
        else:
            analysis.append(f"   结论: 🟢 表现正常")
    
    analysis.append("")
    analysis.append("=" * 60)
    
    return "\n".join(analysis)

def analyze_positive_ratio(df: pd.DataFrame) -> str:
    """分析正样本占比"""
    analysis = []
    analysis.append("=" * 60)
    analysis.append("⚖️  正样本占比分析")
    analysis.append("=" * 60)
    analysis.append("")
    
    df['is_positive'] = df['score'] > 0
    ratio_df = df.groupby('step')['is_positive'].agg(['sum', 'count']).reset_index()
    ratio_df['ratio'] = ratio_df['sum'] / ratio_df['count'] * 100
    
    first_ratio = ratio_df['ratio'].iloc[0]
    last_ratio = ratio_df['ratio'].iloc[-1]
    avg_ratio = ratio_df['ratio'].mean()
    max_ratio = ratio_df['ratio'].max()
    min_ratio = ratio_df['ratio'].min()
    
    analysis.append(f"📊 占比统计:")
    analysis.append(f"   初始占比: {first_ratio:>6.2f}%")
    analysis.append(f"   最终占比: {last_ratio:>6.2f}%")
    analysis.append(f"   平均占比: {avg_ratio:>6.2f}%")
    analysis.append(f"   最高占比: {max_ratio:>6.2f}%")
    analysis.append(f"   最低占比: {min_ratio:>6.2f}%")
    analysis.append("")
    
    ratio_change = last_ratio - first_ratio
    analysis.append(f"📈 变化趋势:")
    analysis.append(f"   占比变化: {ratio_change:>+6.2f}%")
    
    if ratio_change > 10:
        trend = "正样本占比显著提升，策略优化效果好"
    elif ratio_change > 5:
        trend = "正样本占比稳步提升"
    elif ratio_change > -5:
        trend = "占比基本稳定"
    else:
        trend = "正样本占比下降，需要关注"
    analysis.append(f"   趋势: {trend}")
    analysis.append("")
    
    analysis.append(f"🏥 健康度评估:")
    
    if avg_ratio > 70:
        health = "🟢 优秀 - 正样本占比很高，策略质量好"
    elif avg_ratio > 55:
        health = "🟢 良好 - 正样本占主导"
    elif avg_ratio > 45:
        health = "🟡 中等 - 正负样本较为均衡"
    elif avg_ratio > 30:
        health = "🟠 偏低 - 负样本偏多"
    else:
        health = "🔴 较差 - 负样本占主导，建议检查策略"
    
    analysis.append(f"   状态: {health}")
    analysis.append(f"   当前占比: {last_ratio:.2f}%")
    analysis.append("")
    
    ratio_std = ratio_df['ratio'].std()
    analysis.append(f"📉 稳定性:")
    analysis.append(f"   标准差: {ratio_std:.2f}%")
    
    if ratio_std < 5:
        stability = "稳定性很好"
    elif ratio_std < 10:
        stability = "稳定性良好"
    else:
        stability = "波动较大"
    analysis.append(f"   评估: {stability}")
    
    analysis.append("")
    analysis.append("=" * 60)
    
    return "\n".join(analysis)

def analyze_validation_metrics(val_df: pd.DataFrame) -> str:
    """分析验证集指标"""
    analysis = []
    analysis.append("=" * 60)
    analysis.append("🔍 验证集指标分析")
    analysis.append("=" * 60)
    analysis.append("")
    
    if val_df.empty:
        analysis.append("⚠️  未找到验证集数据")
        analysis.append("")
        analysis.append("提示: 验证集数据通常在训练过程中定期生成")
        analysis.append("=" * 60)
        return "\n".join(analysis)
    
    analysis.append(f"📋 基本信息:")
    analysis.append(f"   验证次数: {len(val_df)}")
    analysis.append(f"   验证步数: {val_df['step'].tolist()}")
    analysis.append("")
    
    score_mean = val_df['score'].mean()
    score_std = val_df['score'].std()
    score_trend = val_df['score'].iloc[-1] - val_df['score'].iloc[0] if len(val_df) > 1 else 0
    
    analysis.append(f"📊 Score指标:")
    analysis.append(f"   平均分数: {score_mean:>8.4f}")
    analysis.append(f"   标准差:   {score_std:>8.4f}")
    analysis.append(f"   趋势变化: {score_trend:>+8.4f}")
    
    if score_mean > 0.5:
        score_eval = "✅ 验证集表现优秀"
    elif score_mean > 0:
        score_eval = "🟢 验证集表现良好"
    elif score_mean > -0.3:
        score_eval = "🟡 验证集表现一般"
    else:
        score_eval = "⚠️  验证集表现较差"
    analysis.append(f"   评估: {score_eval}")
    analysis.append("")
    
    if 'reward' in val_df.columns:
        reward_mean = val_df['reward'].mean()
        reward_trend = val_df['reward'].iloc[-1] - val_df['reward'].iloc[0] if len(val_df) > 1 else 0
        
        analysis.append(f"🎁 Reward指标:")
        analysis.append(f"   平均奖励: {reward_mean:>8.4f}")
        analysis.append(f"   趋势变化: {reward_trend:>+8.4f}")
        analysis.append("")
    
    if 'acc' in val_df.columns:
        acc_mean = val_df['acc'].mean()
        acc_trend = val_df['acc'].iloc[-1] - val_df['acc'].iloc[0] if len(val_df) > 1 else 0
        
        analysis.append(f"✅ 准确率指标:")
        analysis.append(f"   平均准确率: {acc_mean:>8.4f}")
        analysis.append(f"   趋势变化:   {acc_trend:>+8.4f}")
        
        if acc_mean > 0.7:
            acc_eval = "✅ 准确率很高"
        elif acc_mean > 0.5:
            acc_eval = "🟢 准确率良好"
        else:
            acc_eval = "🟡 准确率有提升空间"
        analysis.append(f"   评估: {acc_eval}")
        analysis.append("")
    
    analysis.append(f"🎯 整体评估:")
    
    if score_trend > 0.1:
        overall = "✅ 模型在验证集上持续改进，训练效果显著"
    elif score_trend > 0:
        overall = "🟢 模型稳步提升"
    elif score_trend > -0.1:
        overall = "🟡 模型表现稳定"
    else:
        overall = "⚠️  模型在验证集上表现下降，建议检查"
    
    analysis.append(f"   {overall}")
    
    analysis.append("")
    analysis.append("=" * 60)
    
    return "\n".join(analysis)

def plot_score_trend(df: pd.DataFrame, val_df: pd.DataFrame = None) -> plt.Figure:
    """绘制Score趋势图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df_agg = df.groupby('step')['score'].agg(['mean', 'std', 'count']).reset_index()
    
    # 主曲线
    ax.plot(df_agg['step'], df_agg['mean'], 'b-', linewidth=2, label='Train Score (Mean)', alpha=0.8)
    
    # 标准差阴影
    if len(df_agg) > 0:
        ax.fill_between(df_agg['step'], 
                        df_agg['mean'] - df_agg['std'], 
                        df_agg['mean'] + df_agg['std'],
                        alpha=0.2, color='blue', label='±1 Std Dev')
    
    # 滑动平均
    window = max(3, len(df_agg) // 20)
    if len(df_agg) >= window:
        df_agg['ma'] = df_agg['mean'].rolling(window=window, center=True).mean()
        ax.plot(df_agg['step'], df_agg['ma'], 'darkblue', linewidth=3, 
                linestyle='--', label=f'Moving Avg (window={window})')
    
    # 验证点
    if val_df is not None and not val_df.empty:
        ax.scatter(val_df['step'], val_df['score'], color='red', s=150, 
                  marker='*', label='Validation', zorder=5, edgecolors='darkred', linewidths=2)
    
    # 趋势线
    if len(df_agg) > 1:
        z = np.polyfit(df_agg['step'], df_agg['mean'], 1)
        p = np.poly1d(z)
        ax.plot(df_agg['step'], p(df_agg['step']), "g--", alpha=0.5, linewidth=2, 
                label=f'Trend Line (slope={z[0]:.4f})')
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Training Score Trend', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_score_distribution(df: pd.DataFrame) -> plt.Figure:
    """绘制Score分布图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    scores = df['score'].values
    
    # 直方图
    ax1.hist(scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=np.mean(scores), color='red', linestyle='--', linewidth=2, 
                label=f'Mean={np.mean(scores):.3f}')
    ax1.axvline(x=np.median(scores), color='green', linestyle='--', linewidth=2,
                label=f'Median={np.median(scores):.3f}')
    ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Score Distribution Histogram', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 箱线图
    df_sorted = df.sort_values('step')
    steps = df_sorted['step'].unique()
    step_sample = steps[::max(1, len(steps)//15)]
    
    box_data = [df[df['step'] == s]['score'].values for s in step_sample]
    bp = ax2.boxplot(box_data, labels=[str(int(s)) for s in step_sample], patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Score Distribution by Step (Boxplot)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def plot_accuracy_trend(df: pd.DataFrame, val_df: pd.DataFrame = None) -> plt.Figure:
    """绘制Accuracy趋势图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if 'acc' not in df.columns or df['acc'].isna().all():
        ax.text(0.5, 0.5, 'No Accuracy Data Found', 
                ha='center', va='center', fontsize=20, color='gray')
        ax.axis('off')
        return fig
    
    df_agg = df.groupby('step')['acc'].agg(['mean', 'std']).reset_index()
    
    ax.plot(df_agg['step'], df_agg['mean'], 'g-', linewidth=2, label='Train Accuracy', alpha=0.8)
    ax.fill_between(df_agg['step'], 
                    df_agg['mean'] - df_agg['std'], 
                    df_agg['mean'] + df_agg['std'],
                    alpha=0.2, color='green', label='±1 Std Dev')
    
    if val_df is not None and not val_df.empty and 'acc' in val_df.columns:
        ax.scatter(val_df['step'], val_df['acc'], color='orange', s=150,
                  marker='s', label='Validation Accuracy', zorder=5, edgecolors='darkorange', linewidths=2)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Training Accuracy Trend', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_positive_ratio(df: pd.DataFrame) -> plt.Figure:
    """绘制正负样本占比"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df['is_positive'] = df['score'] > 0
    ratio_df = df.groupby('step')['is_positive'].agg(['sum', 'count']).reset_index()
    ratio_df['ratio'] = ratio_df['sum'] / ratio_df['count'] * 100
    
    ax.plot(ratio_df['step'], ratio_df['ratio'], 'purple', linewidth=2.5, marker='o', markersize=6)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='50% Baseline')
    ax.fill_between(ratio_df['step'], ratio_df['ratio'], 50, 
                    where=(ratio_df['ratio'] >= 50), alpha=0.3, color='green', label='Above 50%')
    ax.fill_between(ratio_df['step'], ratio_df['ratio'], 50,
                    where=(ratio_df['ratio'] < 50), alpha=0.3, color='red', label='Below 50%')
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Positive Sample Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Positive Sample Ratio Trend', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_validation_metrics(val_df: pd.DataFrame) -> plt.Figure:
    """绘制Validation指标"""
    if val_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Validation Data Found', 
                ha='center', va='center', fontsize=20, color='gray')
        ax.axis('off')
        return fig
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Score
    axes[0].plot(val_df['step'], val_df['score'], 'b-o', linewidth=2, markersize=8)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Step', fontsize=11)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title('Validation Score', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Reward
    if 'reward' in val_df.columns:
        axes[1].plot(val_df['step'], val_df['reward'], 'purple', marker='s', 
                    linewidth=2, markersize=8)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Step', fontsize=11)
        axes[1].set_ylabel('Reward', fontsize=11)
        axes[1].set_title('Validation Reward', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No Reward Data', ha='center', va='center', fontsize=14, color='gray')
        axes[1].axis('off')
    
    # Accuracy
    if 'acc' in val_df.columns:
        axes[2].plot(val_df['step'], val_df['acc'], 'g-^', linewidth=2, markersize=8)
        axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Step', fontsize=11)
        axes[2].set_ylabel('Accuracy', fontsize=11)
        axes[2].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No Accuracy Data', ha='center', va='center', fontsize=14, color='gray')
        axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def generate_statistics(df: pd.DataFrame, val_df: pd.DataFrame = None) -> str:
    """生成统计信息文本"""
    stats = []
    stats.append("=" * 60)
    stats.append("📊 训练统计信息总览")
    stats.append("=" * 60)
    stats.append("")
    
    stats.append(f"🔢 基本信息:")
    stats.append(f"   总训练步数: {df['step'].max()}")
    stats.append(f"   总样本数:   {len(df)}")
    stats.append(f"   每步样本数: {len(df) / df['step'].nunique():.1f} (平均)")
    stats.append("")
    
    scores = df['score'].values
    stats.append("📈 分数统计:")
    stats.append(f"   均值:     {np.mean(scores):>8.4f}")
    stats.append(f"   标准差:   {np.std(scores):>8.4f}")
    stats.append(f"   中位数:   {np.median(scores):>8.4f}")
    stats.append(f"   最小值:   {np.min(scores):>8.4f}")
    stats.append(f"   最大值:   {np.max(scores):>8.4f}")
    stats.append(f"   Q25:      {np.percentile(scores, 25):>8.4f}")
    stats.append(f"   Q75:      {np.percentile(scores, 75):>8.4f}")
    stats.append("")
    
    pos_count = np.sum(scores > 0)
    neg_count = np.sum(scores < 0)
    zero_count = np.sum(scores == 0)
    stats.append("🎯 样本分布:")
    stats.append(f"   正样本: {pos_count:>6} ({pos_count/len(scores)*100:>5.1f}%)")
    stats.append(f"   负样本: {neg_count:>6} ({neg_count/len(scores)*100:>5.1f}%)")
    stats.append(f"   零值:   {zero_count:>6} ({zero_count/len(scores)*100:>5.1f}%)")
    stats.append("")
    
    if 'acc' in df.columns:
        accs = df['acc'].dropna().values
        if len(accs) > 0:
            stats.append("✅ 准确率统计:")
            stats.append(f"   平均值: {np.mean(accs):>8.4f}")
            stats.append(f"   最终值: {df.groupby('step')['acc'].mean().iloc[-1]:>8.4f}")
            stats.append("")
    
    early_scores = scores[:len(scores)//5]
    recent_scores = scores[-len(scores)//5:]
    change = np.mean(recent_scores) - np.mean(early_scores)
    
    stats.append("📊 趋势分析:")
    stats.append(f"   初期平均 (前20%): {np.mean(early_scores):>8.4f}")
    stats.append(f"   近期平均 (后20%): {np.mean(recent_scores):>8.4f}")
    
    if change > 0.01:
        trend_indicator = "📈 (改进中)"
    elif change < -0.01:
        trend_indicator = "📉 (下降中)"
    else:
        trend_indicator = "➡️  (稳定)"
    
    stats.append(f"   变化量:           {change:>8.4f} {trend_indicator}")
    stats.append("")
    
    if val_df is not None and not val_df.empty:
        stats.append("🔍 验证集统计:")
        stats.append(f"   验证次数: {len(val_df)}")
        stats.append(f"   平均分数: {val_df['score'].mean():>8.4f}")
        if 'reward' in val_df.columns:
            stats.append(f"   平均奖励: {val_df['reward'].mean():>8.4f}")
        if 'acc' in val_df.columns:
            stats.append(f"   平均准确率: {val_df['acc'].mean():>8.4f}")
        stats.append("")
    
    stats.append("🏥 训练健康度评估:")
    recent_avg = np.mean(recent_scores)
    
    if recent_avg > 0.5:
        health = "🟢 优秀 - 模型学习效果很好"
    elif recent_avg > 0:
        health = "🟡 良好 - 训练进展顺利"
    elif recent_avg > -0.3:
        health = "🟠 一般 - 需要关注"
    else:
        health = "🔴 较差 - 建议调整超参数"
    
    stats.append(f"   状态: {health}")
    stats.append(f"   近期分数: {recent_avg:.4f}")
    stats.append("")
    
    stats.append("=" * 60)
    
    return "\n".join(stats)

def analyze_training(task_name: str, plots: List[str]) -> Tuple:
    """主分析函数"""
    if not task_name:
        empty_msg = "⚠️ 请选择一个训练任务"
        return None, empty_msg, None, empty_msg, None, empty_msg, None, empty_msg, None, empty_msg, empty_msg, empty_msg
    
    try:
        df = load_rollout_data(task_name)
        val_df = load_validation_data(task_name)
        
        if df.empty:
            error_msg = f"❌ 未找到任务数据: {task_name}"
            return None, error_msg, None, error_msg, None, error_msg, None, error_msg, None, error_msg, error_msg, error_msg
        
        stats_text = generate_statistics(df, val_df)
        recommendation_text = recommend_best_checkpoint(task_name, df, val_df)  # ✅ 传入task_name
        
        plot1, analysis1 = (None, "")
        plot2, analysis2 = (None, "")
        plot3, analysis3 = (None, "")
        plot4, analysis4 = (None, "")
        plot5, analysis5 = (None, "")
        
        if "分数趋势" in plots:
            plot1 = plot_score_trend(df, val_df)
            analysis1 = analyze_score_trend(df, val_df)
        
        if "分数分布" in plots:
            plot2 = plot_score_distribution(df)
            analysis2 = analyze_score_distribution(df)
        
        if "准确率趋势" in plots:
            plot3 = plot_accuracy_trend(df, val_df)
            analysis3 = analyze_accuracy_trend(df, val_df)
        
        if "正样本占比" in plots:
            plot4 = plot_positive_ratio(df)
            analysis4 = analyze_positive_ratio(df)
        
        if "验证指标" in plots:
            plot5 = plot_validation_metrics(val_df)
            analysis5 = analyze_validation_metrics(val_df)
        
        return plot1, analysis1, plot2, analysis2, plot3, analysis3, plot4, analysis4, plot5, analysis5, stats_text, recommendation_text
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 错误: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, None, error_msg, None, error_msg, None, error_msg, None, error_msg, error_msg, error_msg

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="训练监控台", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎯 训练日志监控台
        ### 实时训练日志可视化分析工具
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔧 控制面板")
                
                task_dropdown = gr.Dropdown(
                    choices=scan_training_tasks(),
                    label="📁 选择训练任务",
                    info="选择一个训练任务进行分析",
                    interactive=True
                )
                
                refresh_btn = gr.Button("🔄 刷新任务列表", size="sm")
                
                plot_checkboxes = gr.CheckboxGroup(
                    choices=[
                        "分数趋势",
                        "分数分布", 
                        "准确率趋势",
                        "正样本占比",
                        "验证指标"
                    ],
                    value=["分数趋势", "分数分布"],
                    label="📊 选择要显示的图表",
                    info="勾选你想查看的可视化图表"
                )
                
                analyze_btn = gr.Button("🚀 开始分析", variant="primary", size="lg")
                
                gr.Markdown("""
                ---
                **使用提示：**
                - 从下拉菜单选择训练任务
                - 勾选想要查看的图表类型
                - 点击"开始分析"生成可视化
                - 每个图表下方都有详细分析
                - **新增：模型推荐** 🏆
                """)
            
            with gr.Column(scale=3):
                gr.Markdown("### 📈 可视化结果")
                
                with gr.Tabs():
                    with gr.Tab("🏆 模型推荐"):
                        recommendation_output = gr.Textbox(
                            label="检查点推荐",
                            lines=35,
                            max_lines=50,
                            show_label=True,
                            elem_classes="monospace",
                            interactive=False,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("📊 统计总览"):
                        stats_output = gr.Textbox(
                            label="训练统计信息",
                            lines=35,
                            max_lines=50,
                            show_label=False,
                            elem_classes="monospace",
                            interactive=False,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("📈 分数趋势"):
                        plot1 = gr.Plot(label="分数趋势图")
                        analysis1 = gr.Textbox(
                            label="趋势分析",
                            lines=25,
                            max_lines=40,
                            show_label=True,
                            interactive=False,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("📊 分数分布"):
                        plot2 = gr.Plot(label="分数分布图")
                        analysis2 = gr.Textbox(
                            label="分布分析",
                            lines=25,
                            max_lines=40,
                            show_label=True,
                            interactive=False,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("✅ 准确率"):
                        plot3 = gr.Plot(label="准确率趋势图")
                        analysis3 = gr.Textbox(
                            label="准确率分析",
                            lines=25,
                            max_lines=40,
                            show_label=True,
                            interactive=False,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("⚖️ 正样本占比"):
                        plot4 = gr.Plot(label="正样本比例图")
                        analysis4 = gr.Textbox(
                            label="占比分析",
                            lines=25,
                            max_lines=40,
                            show_label=True,
                            interactive=False,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("🔍 验证集"):
                        plot5 = gr.Plot(label="验证集指标图")
                        analysis5 = gr.Textbox(
                            label="验证集分析",
                            lines=25,
                            max_lines=40,
                            show_label=True,
                            interactive=False,
                            show_copy_button=True
                        )
        
        def refresh_tasks():
            return gr.Dropdown(choices=scan_training_tasks())
        
        refresh_btn.click(
            fn=refresh_tasks,
            outputs=task_dropdown
        )
        
        analyze_btn.click(
            fn=analyze_training,
            inputs=[task_dropdown, plot_checkboxes],
            outputs=[plot1, analysis1, plot2, analysis2, plot3, analysis3, 
                    plot4, analysis4, plot5, analysis5, stats_output, recommendation_output]
        )
        
        gr.Markdown("""
        ---
        💡 **关于**: 此工具用于分析RLHF训练日志（rollout和validation数据）  
        📍 **根目录**: `/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/output_models`  
        🎨 **特性**: 实时可视化、智能分析、多维度评估、自动推荐最佳checkpoint
        """)
    
    return demo

if __name__ == "__main__":
    print("🚀 启动训练监控台...")
    ROOT_DIR = Path("/lpai/volumes/base-mindgpt-ali-sh-mix/zhaojiale/why_ask/output_models")
    print(f"📂 扫描目录: {ROOT_DIR}")
    
    tasks = scan_training_tasks()
    print(f"✅ 发现 {len(tasks)} 个训练任务")
    
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )