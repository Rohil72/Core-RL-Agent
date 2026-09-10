#!/usr/bin/env python3
"""
Publication Figure Generator for Digital Finance (Springer Nature)
Generates high-resolution vector (PDF) and 300-DPI raster (PNG) figures:
  1. Figure 2: Algorithmic Worked Decision Evidence Cards & Price Trajectories
  2. Figure 3: Evidentiary Faithfulness Intervention Chain
  3. Figure 4: Event-Level Kaplan-Meier Position Survival with Greenwood 95% Bands
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# Configure publication typography and styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Palatino', 'serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': ':',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = REPO_ROOT / "paper" / "internal" / "evidence" / "research_defense_extract" / "v4_deep_robustness"
FIG_DIR = REPO_ROOT / "paper" / "internal" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_figure_2_decision_cards():
    """
    Figure 2: Complete Algorithmic Decision Evidence Cards and Normalized Price Trajectories.
    Shows 3 algorithmically selected cases:
      - Case 1: Successful Concordant Win (US: Visa, V, +14.22% return)
      - Case 2: Active Downside Stop Truncation (China: Hengrui, 600276.SS, -11.46%, chandelier stop)
      - Case 3: Conflicted Disagreement (Brazil: Magazine Luiza, MGLU3.SA, conflict 1.0049)
    """
    print("[*] Generating Figure 2: Algorithmic Worked Decision Cards...")
    json_path = EVIDENCE_DIR / "algorithmic_worked_decisions.json"
    parquet_path = EVIDENCE_DIR / "worked_decision_path_data.parquet"

    with open(json_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    path_df = pd.read_parquet(parquet_path)

    fig = plt.figure(figsize=(14.5, 12.0), constrained_layout=False)
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.35, 1.0], hspace=0.35, wspace=0.25,
                           left=0.06, right=0.98, top=0.94, bottom=0.05)

    case_colors = ['#1f77b4', '#d62728', '#9467bd']
    prec_cmap = plt.cm.Blues

    for row_idx, case in enumerate(cases):
        cid = case['case_id']
        cdf = path_df[path_df['case_id'] == cid].copy()
        
        # Determine anchor price at t=0 for query and precedents
        # Query anchor
        q_sub = cdf[cdf['path_role'] == 'query_observation_252d']
        q_p0 = q_sub[q_sub['relative_session'] == 0]['close'].values
        q_anchor = q_p0[0] if len(q_p0) > 0 else q_sub['close'].iloc[-1]

        # -------------------------------------------------------------
        # Left Panel: Trajectories
        # -------------------------------------------------------------
        ax_plot = fig.add_subplot(gs[row_idx, 0])
        
        # Plot top 5 precedent histories and forward paths
        top_precs = case['top_5_precedents']
        for p in top_precs:
            rank = p['rank']
            h_sub = cdf[cdf['path_role'] == f"precedent_rank_{rank}_history"]
            f_sub = cdf[cdf['path_role'] == f"precedent_rank_{rank}_forward_63d"]
            
            if len(h_sub) > 0:
                p_p0 = h_sub[h_sub['relative_session'] == 0]['close'].values
                p_anchor = p_p0[0] if len(p_p0) > 0 else h_sub['close'].iloc[-1]
                
                h_norm = h_sub['close'] / p_anchor
                f_norm = f_sub['close'] / p_anchor if len(f_sub) > 0 else pd.Series([], dtype=float)
                
                c_val = 0.4 + 0.5 * (6 - rank) / 5.0
                p_color = prec_cmap(c_val)
                
                ax_plot.plot(h_sub['relative_session'], h_norm, color=p_color,
                             linewidth=1.1, alpha=0.7, linestyle='-')
                if len(f_sub) > 0:
                    ax_plot.plot(f_sub['relative_session'], f_norm, color=p_color,
                                 linewidth=1.1, alpha=0.7, linestyle=':')

        # Plot Query observation history (-252 to 0)
        q_norm = q_sub['close'] / q_anchor
        ax_plot.plot(q_sub['relative_session'], q_norm, color='#111111',
                     linewidth=2.3, label=f"Query: {case['ticker']} ({case['market']})")

        # Plot Trade Execution path (0 to holding_days)
        exec_sub = cdf[cdf['path_role'] == 'trade_execution']
        if len(exec_sub) > 0:
            exec_p0 = exec_sub['close'].iloc[0]
            exec_norm = exec_sub['close'] / exec_p0
            trade_ret = case['trade_execution']['return_pct']
            exec_color = '#2ca02c' if trade_ret >= 0 else '#d62728'
            exit_reason = case['trade_execution']['exit_reason']
            label_trade = f"Execution ({trade_ret:+.1f}%, {exit_reason})"
            ax_plot.plot(exec_sub['relative_session'], exec_norm, color=exec_color,
                         linewidth=2.5, linestyle='-', label=label_trade)
            # Mark exit point
            exit_x = exec_sub['relative_session'].iloc[-1]
            exit_y = exec_norm.iloc[-1]
            marker_shape = 'o' if exit_reason == 'calendar_end' else 'X'
            ax_plot.scatter([exit_x], [exit_y], color=exec_color, s=70, zorder=5,
                            edgecolors='black', linewidth=1.2, marker=marker_shape)

        # Baseline reference line at y=1.0 and decision boundary at x=0
        ax_plot.axvline(0, color='#555555', linestyle='--', linewidth=1.2, alpha=0.8)
        ax_plot.axhline(1.0, color='#888888', linestyle='-', linewidth=0.8, alpha=0.5)

        ax_plot.set_xlim(-260, 70)
        ax_plot.set_xlabel("Trading Sessions Relative to Decision ($t=0$)", fontsize=9.5)
        ax_plot.set_ylabel("Normalized Price ($P_t / P_0$)", fontsize=9.5)
        ax_plot.set_title(f"{case['case_id']}: {case['case_type']} | {case['ticker']} ({case['market']})",
                          fontweight='bold', loc='left', color='#111111', fontsize=10.5)
        ax_plot.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=8)

        # -------------------------------------------------------------
        # Right Panel: Structured Decision Card Box
        # -------------------------------------------------------------
        ax_card = fig.add_subplot(gs[row_idx, 1])
        ax_card.axis('off')

        sd = case['score_decomposition']
        te = case['trade_execution']
        ts = case['timestamps']

        card_title = f"{case['case_id']}: {case['case_type']}"
        subtitle = f"Ticker: {case['ticker']} | Market: {case['market']} | Signal: {ts['signal_date']} | Entry: {ts['entry_date']}"

        # Build table of precedents
        prec_rows = []
        for p in top_precs:
            prec_rows.append([
                f"#{p['rank']}",
                f"{p['ticker']} ({p['market']})",
                p['date'],
                f"{p['cosine_similarity']:.3f}",
                f"{p['normalized_weight']:.3f}",
                f"{p['realized_return_63d_pct']:+.1f}%",
                f"{p['realized_drawdown_63d_pct']:+.1f}%"
            ])

        # Render structured text card
        text_box_y = 0.98
        ax_card.text(0.02, text_box_y, card_title, fontsize=11, fontweight='bold', color='#1a237e', va='top')
        text_box_y -= 0.08
        ax_card.text(0.02, text_box_y, subtitle, fontsize=8.5, color='#424242', va='top')
        text_box_y -= 0.07

        score_line_1 = f"Direct Utility (y_hat): {sd['pred_utility_y_hat']:+.4f}   |   Precedent Mean (mu_mem): {sd['precedent_mean_mu_mem']:+.4f}"
        score_line_2 = f"Precedent CVaR_0.05: {sd['expected_shortfall_cvar_05']:.4f}     |   Dispersion (sigma_mem): {sd['precedent_dispersion_sigma_mem']:.4f}"
        score_line_3 = f"Decision Score: {sd['decision_score']:.3f}         |   Std Conflict: {sd['standardized_conflict']:.3f}"
        
        ax_card.text(0.02, text_box_y, score_line_1, fontsize=8.0, family='monospace', va='top')
        text_box_y -= 0.055
        ax_card.text(0.02, text_box_y, score_line_2, fontsize=8.0, family='monospace', va='top')
        text_box_y -= 0.055
        ax_card.text(0.02, text_box_y, score_line_3, fontsize=8.0, family='monospace', fontweight='bold', color='#0d47a1', va='top')
        text_box_y -= 0.07

        # Add precedent table
        col_labels = ["Rank", "Precedent", "Date", "Sim (w)", "Weight", "Ret_63d", "MaxDD"]
        table = ax_card.table(cellText=prec_rows, colLabels=col_labels,
                              loc='upper left', bbox=[0.01, text_box_y - 0.40, 0.98, 0.38])
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        for k, cell in table.get_celld().items():
            cell.set_edgecolor('#cfd8dc')
            if k[0] == 0:
                cell.set_facecolor('#eceff1')
                cell.set_text_props(weight='bold', color='#263238')
            else:
                cell.set_facecolor('#ffffff' if k[0] % 2 == 1 else '#f8f9fa')

        text_box_y -= 0.44
        pnl_color = '#2e7d32' if te['return_pct'] >= 0 else '#c62828'
        exec_summary = (f"Realized Execution: Return = {te['return_pct']:+.2f}%  |  Net PnL = ${te['net_realized_pnl']:+,.2f}\n"
                        f"Exit Reason: {te['exit_reason']}  |  Holding: {ts['holding_days']} sessions  |  Exit Date: {ts['exit_date']}")
        ax_card.text(0.02, text_box_y, exec_summary, fontsize=8.5, fontweight='bold', color=pnl_color, va='top')

        # Background bounding box for card
        rect = plt.Rectangle((0.0, 0.0), 1.0, 1.0, fill=True, facecolor='#fafafa',
                             edgecolor='#b0bec5', linewidth=1.0, transform=ax_card.transAxes, zorder=-1)
        ax_card.add_patch(rect)

    fig.suptitle("Figure 2: Complete Algorithmic Decision Evidence Cards and Normalized Trajectory Paths\n(Primary System P0, Seed 7, Aligned Entry Decisions)",
                 fontsize=12, fontweight='bold', y=0.985)

    pdf_out = FIG_DIR / "figure_2_decision_evidence_cards.pdf"
    png_out = FIG_DIR / "figure_2_decision_evidence_cards.png"
    plt.savefig(pdf_out, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_out, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [+] Saved Figure 2 -> {pdf_out.name}, {png_out.name}")


def generate_figure_3_faithfulness_chain():
    """
    Figure 3: Evidentiary Faithfulness Intervention Chain.
    2x2 panel grid connecting static ranking sensitivity to dynamic portfolio performance.
      (a) Static Ranking Sensitivity: Spearman rho & Kendall tau
      (b) Static Selection Agreement: Top-1 Hit Agreement & Top-3 Overlap
      (c) Dynamic Downstream Return Impact: Ann Return (%) & Delta Return (bps)
      (d) Portfolio Execution Realization: Sharpe Ratio & Changed Trades (%)
    """
    print("[*] Generating Figure 3: Faithfulness Intervention Chain...")
    df_stat = pd.read_csv(EVIDENCE_DIR / "faithfulness_static_decision_matrix.csv")
    df_dyn = pd.read_csv(EVIDENCE_DIR / "faithfulness_dynamic_portfolio_matrix.csv")

    cond_order = [
        "Baseline Replay",
        "Provenance-Only Negative Control",
        "Top-3 Precedent Occlusion",
        "Matched Random Precedent Replacement",
        "Same-Date Cross-Candidate Bundle Swap"
    ]
    cond_labels = [
        "Baseline\nReplay",
        "Provenance\nControl",
        "Top-3\nOcclusion",
        "Random\nReplacement",
        "Bundle\nSwap"
    ]

    p0_stat = df_stat[df_stat['system'] == 'P0'].set_index('condition').reindex(cond_order)
    p0s_stat = df_stat[df_stat['system'] == 'P0*'].set_index('condition').reindex(cond_order)
    p0_dyn = df_dyn[df_dyn['system'] == 'P0'].set_index('condition').reindex(cond_order)
    p0s_dyn = df_dyn[df_dyn['system'] == 'P0*'].set_index('condition').reindex(cond_order)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), constrained_layout=True)
    x = np.arange(len(cond_order))
    width = 0.35

    # -------------------------------------------------------------
    # (a) Static Ranking Sensitivity: Spearman rho
    # -------------------------------------------------------------
    ax = axes[0, 0]
    ax.bar(x - width/2, p0_stat['rank_spearman_rho'], width, label="Primary P0",
           color='#1f77b4', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.bar(x + width/2, p0s_stat['rank_spearman_rho'], width, label="Exploratory P0*",
           color='#ff7f0e', edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8.5)
    ax.set_ylabel("Spearman Rank Correlation ($\\rho$)", fontsize=9.5)
    ax.set_title("(a) Static Candidate Ranking Sensitivity", fontweight='bold', loc='left')
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(loc='upper right', frameon=True)
    for i in x:
        ax.text(i - width/2, p0_stat['rank_spearman_rho'].iloc[i] + 0.02,
                f"{p0_stat['rank_spearman_rho'].iloc[i]:.2f}", ha='center', fontsize=7.5)
        ax.text(i + width/2, p0s_stat['rank_spearman_rho'].iloc[i] + 0.02,
                f"{p0s_stat['rank_spearman_rho'].iloc[i]:.2f}", ha='center', fontsize=7.5)

    # -------------------------------------------------------------
    # (b) Static Selection Agreement: Top-1 Hit & Top-3 Overlap
    # -------------------------------------------------------------
    ax = axes[0, 1]
    ax.bar(x - width/2, p0_stat['top1_hit_agreement_pct'], width, label="P0 Top-1 Hit (%)",
           color='#2ca02c', edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.bar(x + width/2, p0_stat['top3_set_overlap_pct'], width, label="P0 Top-3 Overlap (%)",
           color='#17becf', edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8.5)
    ax.set_ylabel("Agreement Percentage (%)", fontsize=9.5)
    ax.set_title("(b) Cross-Sectional Selection Retention (P0)", fontweight='bold', loc='left')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', frameon=True)
    for i in x:
        ax.text(i - width/2, p0_stat['top1_hit_agreement_pct'].iloc[i] + 2,
                f"{p0_stat['top1_hit_agreement_pct'].iloc[i]:.1f}%", ha='center', fontsize=7.5)
        ax.text(i + width/2, p0_stat['top3_set_overlap_pct'].iloc[i] + 2,
                f"{p0_stat['top3_set_overlap_pct'].iloc[i]:.1f}%", ha='center', fontsize=7.5)

    # -------------------------------------------------------------
    # (c) Dynamic Return Impact: Annualized Return (%) & Delta Return
    # -------------------------------------------------------------
    ax = axes[1, 0]
    ax.bar(x - width/2, p0_dyn['annualized_return_pct'], width, label="Primary P0 Return (%)",
           color='#1f77b4', edgecolor='black', linewidth=0.8, alpha=0.9)
    ax.bar(x + width/2, p0s_dyn['annualized_return_pct'], width, label="Exploratory P0* Return (%)",
           color='#ff7f0e', edgecolor='black', linewidth=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8.5)
    ax.set_ylabel("Annualized Net Return (%)", fontsize=9.5)
    ax.set_title("(c) Dynamic Portfolio Return under Frictions", fontweight='bold', loc='left')
    ax.axhline(0.0, color='black', linewidth=1.0)
    ax.legend(loc='upper right', frameon=True)
    for i in x:
        r0 = p0_dyn['annualized_return_pct'].iloc[i]
        d0 = p0_dyn['delta_return_bps'].iloc[i]
        txt0 = f"{r0:+.2f}%\n({d0:+.0f}bp)"
        ax.text(i - width/2, r0 + (0.15 if r0 >= 0 else -0.55), txt0, ha='center', fontsize=7.0)
        
        r0s = p0s_dyn['annualized_return_pct'].iloc[i]
        d0s = p0s_dyn['delta_return_bps'].iloc[i]
        txt0s = f"{r0s:+.2f}%\n({d0s:+.0f}bp)"
        ax.text(i + width/2, r0s + (0.15 if r0s >= 0 else -0.55), txt0s, ha='center', fontsize=7.0)
    ax.set_ylim(-1.8, 4.6)

    # -------------------------------------------------------------
    # (d) Dynamic Execution: Sharpe Ratio & Changed Trade %
    # -------------------------------------------------------------
    ax = axes[1, 1]
    ax.plot(x, p0_dyn['sharpe_ratio'], marker='o', color='#1f77b4', linewidth=2.0,
            markersize=7, label="P0 Sharpe Ratio (left)")
    ax.plot(x, p0s_dyn['sharpe_ratio'], marker='s', color='#ff7f0e', linewidth=2.0,
            markersize=7, linestyle='--', label="P0* Sharpe Ratio (left)")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, fontsize=8.5)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=9.5)
    ax.set_ylim(-0.02, 0.35)
    ax.set_title("(d) Execution Degradation and Changed Trade Entries", fontweight='bold', loc='left')

    # Secondary axis for changed trade %
    ax2 = ax.twinx()
    ax2.spines['right'].set_visible(True)
    ax2.bar(x + width/3, p0_dyn['changed_trade_pct'], width=0.25, color='#d62728',
            alpha=0.35, edgecolor='#d62728', label="P0 Changed Trades % (right)")
    ax2.set_ylabel("Trades Altered (%)", color='#c62828', fontsize=9.5)
    ax2.tick_params(axis='y', labelcolor='#c62828')
    ax2.set_ylim(0, 100)
    ax2.grid(False)

    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, fontsize=8)

    fig.suptitle("Figure 3: Evidentiary Faithfulness Intervention Chain\n(Bridging Static Candidate Ranking Sensitivities to Realized Portfolio Execution under 10 bps Frictions)",
                 fontsize=12, fontweight='bold')

    pdf_out = FIG_DIR / "figure_3_faithfulness_intervention_chain.pdf"
    png_out = FIG_DIR / "figure_3_faithfulness_intervention_chain.png"
    plt.savefig(pdf_out, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_out, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [+] Saved Figure 3 -> {pdf_out.name}, {png_out.name}")


def generate_figure_4_kaplan_meier_survival():
    """
    Figure 4: Kaplan-Meier Position Survival with Greenwood 95% Confidence Bands.
    Panel (a): Kaplan-Meier survival curves S(t) through day 63 with Greenwood bands for P0, P0*, P1, P2, P4.
    Panel (b): Cause-specific exit incidence (ATR Chandelier Stop vs. 63-session ceiling vs. calendar-end censor)
               along with number of positions at risk over time.
    """
    print("[*] Generating Figure 4: Kaplan-Meier Survival Analysis...")
    km_df = pd.read_csv(EVIDENCE_DIR / "kaplan_meier_survival_with_greenwood_bands.csv")
    with open(EVIDENCE_DIR / "holding_survival_summary.json", 'r', encoding='utf-8') as f:
        summary = json.load(f)

    fig, (ax_km, ax_cause) = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)

    system_styles = {
        'P0':  {'label': 'Primary P0 (Retrieval-Grounded)', 'color': '#1f77b4', 'ls': '-', 'lw': 2.2, 'zorder': 5},
        'P0*': {'label': 'Exploratory P0* (Domestic Guardrail)', 'color': '#ff7f0e', 'ls': '--', 'lw': 1.8, 'zorder': 4},
        'P1':  {'label': 'P1 (No-Memory Transformer)', 'color': '#2ca02c', 'ls': '-.', 'lw': 1.6, 'zorder': 3},
        'P2':  {'label': 'P2 (Mean-Outcome Only)', 'color': '#9467bd', 'ls': ':', 'lw': 1.6, 'zorder': 2},
        'P4':  {'label': 'P4 (21-Day Momentum)', 'color': '#7f7f7f', 'ls': (0, (3, 1, 1, 1)), 'lw': 1.5, 'zorder': 1}
    }

    # -------------------------------------------------------------
    # (a) Kaplan-Meier Survival Curves S(t) with Greenwood Bands
    # -------------------------------------------------------------
    for sys_id, style in system_styles.items():
        sub = km_df[km_df['system'] == sys_id].sort_values('holding_day')
        days = sub['holding_day'].values
        surv = sub['survival_prob'].values
        lower = sub['ci_95_lower'].values
        upper = sub['ci_95_upper'].values

        rmst_val = summary[sys_id]['rmst_day_63']
        rmst_se = summary[sys_id]['rmst_cluster_se']
        lbl = f"{style['label']} (RMST: {rmst_val:.1f}d)"

        ax_km.step(days, surv, where='post', label=lbl, color=style['color'],
                   linestyle=style['ls'], linewidth=style['lw'], zorder=style['zorder'])

        # Plot Greenwood confidence bands for primary P0 and P0*
        if sys_id in ['P0', 'P0*']:
            ax_km.fill_between(days, lower, upper, step='post', color=style['color'],
                               alpha=0.15, zorder=style['zorder'] - 1)

    ax_km.set_xlim(0, 64)
    ax_km.set_ylim(0.0, 1.05)
    ax_km.set_xlabel("Holding Duration (Trading Sessions)", fontsize=10)
    ax_km.set_ylabel("Position Survival Probability $S(t)$", fontsize=10)
    ax_km.set_title("(a) Kaplan-Meier Survival with 95% Greenwood Confidence Bands", fontweight='bold', loc='left')
    ax_km.axvline(63, color='#888888', linestyle='--', linewidth=1.0, alpha=0.7)
    ax_km.text(62, 0.05, "63-Session Ceiling", rotation=90, va='bottom', ha='right', color='#555555', fontsize=8)
    ax_km.legend(loc='lower left', frameon=True, framealpha=0.92, fontsize=8)

    # -------------------------------------------------------------
    # (b) Cause-Specific Exit Dynamics & Risk Set Decomposition (P0)
    # -------------------------------------------------------------
    p0_km = km_df[km_df['system'] == 'P0'].sort_values('holding_day')
    days = p0_km['holding_day'].values
    
    # Cumulative incidence of ATR stop exits vs max horizon vs calendar censors
    cum_stop = np.cumsum(p0_km['events_stop'].values) / p0_km['n_at_risk'].iloc[0] * 100.0
    cum_max_h = np.cumsum(p0_km['events_max_horizon'].values) / p0_km['n_at_risk'].iloc[0] * 100.0
    cum_censor = np.cumsum(p0_km['censored_calendar'].values) / p0_km['n_at_risk'].iloc[0] * 100.0

    ax_cause.plot(days, cum_stop, color='#d62728', linewidth=2.2, label="Active ATR Chandelier Stop (Downside Exit)")
    ax_cause.plot(days, cum_max_h, color='#2ca02c', linewidth=2.0, linestyle='--', label="63-Session Maturity Exit")
    ax_cause.plot(days, cum_censor, color='#7f7f7f', linewidth=1.8, linestyle=':', label="Administrative Calendar-End Censor")

    # Stacked fill
    ax_cause.fill_between(days, 0, cum_stop, color='#d62728', alpha=0.15)
    ax_cause.fill_between(days, cum_stop, cum_stop + cum_max_h, color='#2ca02c', alpha=0.12)
    ax_cause.fill_between(days, cum_stop + cum_max_h, cum_stop + cum_max_h + cum_censor, color='#7f7f7f', alpha=0.10)

    ax_cause.set_xlim(0, 64)
    ax_cause.set_ylim(0, 105)
    ax_cause.set_xlabel("Holding Duration (Trading Sessions)", fontsize=10)
    ax_cause.set_ylabel("Cumulative Exit Incidence (% of Initial Risk Set)", fontsize=10)
    ax_cause.set_title("(b) Cause-Specific Exit Incidence & Risk Dynamics (Primary P0)", fontweight='bold', loc='left')
    ax_cause.legend(loc='upper left', frameon=True, framealpha=0.92, fontsize=8)

    # Inset annotation box for RMST & complete follow-up cohort
    p0_sum = summary['P0']
    box_text = (
        "Primary P0 Survival Invariants:\n"
        f"• All-Trades RMST: {p0_sum['rmst_day_63']:.2f}d (SE {p0_sum['rmst_cluster_se']:.2f})\n"
        f"• Median Hold: {p0_sum['median_holding_days']:.1f} sessions (Active Truncation)\n"
        f"• Complete Follow-up (T >= 63d):\n"
        f"   N = {p0_sum['eligible_cohort_trades']} trades | Censors = 0.0%\n"
        f"   Cohort RMST: {p0_sum['eligible_rmst_day_63']:.2f}d"
    )
    ax_cause.text(0.52, 0.18, box_text, transform=ax_cause.transAxes,
                  fontsize=8.0, family='monospace',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#b0bec5', alpha=0.95))

    fig.suptitle("Figure 4: Event-Level Position Survival & Greenwood Risk Accounting\n(2,637 Evaluated Trades Across 6 Sovereign Markets, 2024 Authoritative Audit)",
                 fontsize=12, fontweight='bold')

    pdf_out = FIG_DIR / "figure_4_kaplan_meier_survival.pdf"
    png_out = FIG_DIR / "figure_4_kaplan_meier_survival.png"
    plt.savefig(pdf_out, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_out, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   [+] Saved Figure 4 -> {pdf_out.name}, {png_out.name}")


if __name__ == "__main__":
    generate_figure_2_decision_cards()
    generate_figure_3_faithfulness_chain()
    generate_figure_4_kaplan_meier_survival()
    print("\n[SUCCESS] All publication figures generated successfully.")
