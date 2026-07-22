import sys

with open('scripts/run_final_direction.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'Entropy vs random (z-score)' in line and 'is not None else' in line:
        new_lines.append('        f"- **Entropy vs random (z-score)**: {ds[\\'entropy_vs_random_zscore\\']:.2f}" if ds[\\'entropy_vs_random_zscore\\'] is not None else "- **Entropy vs random (z-score)**: —",\n')
    elif 'Entropy as % of maximum' in line and 'is not None else' in line:
        new_lines.append('        f"- **Entropy as % of maximum**: {ds[\\'entropy_pct_of_max\\']:.1%}" if ds[\\'entropy_pct_of_max\\'] is not None else "",\n')
    elif 'MAE |' in line and 'is not None else' in line:
        new_lines.append('                f"| MAE | {f\\"{rm[\\'mae\\']:.4f}\\" if rm[\\'mae\\'] is not None else \\"—\\"} | {f\\"{mm[\\'mae\\']:.4f}\\" if mm[\\'mae\\'] is not None else \\"—\\"} | — |",\n')
    elif 'RMSE |' in line and 'is not None else' in line:
        new_lines.append('                f"| RMSE | {f\\"{rm[\\'rmse\\']:.4f}\\" if rm[\\'rmse\\'] is not None else \\"—\\"} | {f\\"{mm[\\'rmse\\']:.4f}\\" if mm[\\'rmse\\'] is not None else \\"—\\"} | — |",\n')
    elif 'Pearson |' in line and 'is not None else' in line:
        new_lines.append('                f"| Pearson | {f\\"{rm[\\'pearson\\']:.4f}\\" if rm[\\'pearson\\'] is not None else \\"—\\"} | {f\\"{mm[\\'pearson\\']:.4f}\\" if mm[\\'pearson\\'] is not None else \\"—\\"} | — |",\n')
    elif 'Spearman |' in line and 'is not None else' in line:
        new_lines.append('                f"| Spearman | {f\\"{rm[\\'spearman\\']:.4f}\\" if rm[\\'spearman\\'] is not None else \\"—\\"} | {f\\"{mm[\\'spearman\\']:.4f}\\" if mm[\\'spearman\\'] is not None else \\"—\\"} | **{winner.upper()}** |",\n')
    elif 'R² |' in line and 'is not None else' in line:
        new_lines.append('                f"| R² | {f\\"{rm[\\'r2\\']:.4f}\\" if rm[\\'r2\\'] is not None else \\"—\\"} | {f\\"{mm[\\'r2\\']:.4f}\\" if mm[\\'r2\\'] is not None else \\"—\\"} | — |",\n')
    elif 'r_str =' in line and 'is not None else' in line:
        new_lines.append('                r_str = f"{r_val:.4f}" if r_val is not None else "—"\n')
    elif 'm_str =' in line and 'is not None else' in line:
        new_lines.append('                m_str = f"{m_val:.4f}" if m_val is not None else "—"\n')
    elif 'u_str =' in line and 'is not None else' in line:
        new_lines.append('                u_str = f"{univ:.4f}" if univ is not None else "—"\n')
    elif 'Universe mean target:' in line and 'is not None else' in line:
        new_lines.append('            f"- Universe mean target: **{s.get(\\'universe_mean_mean\\'):.4f}**" if s.get(\\'universe_mean_mean\\') is not None else "- Universe mean target: —",\n')
    elif 'capitalize()' in line and 'is not None else' in line:
        new_lines.append('            lines.append(f"| {m.capitalize()} | {f\\"{sp:.4f}\\" if sp is not None else \\"—\\"} |")\n')
    elif 'v5:.4f if v5' in line:
        new_lines.append('                f"| {v5:.4f}" if v5 is not None else "| — "\n')
    elif 'v10:.4f if v10' in line:
        new_lines.append('                f"| {v10:.4f}" if v10 is not None else "| — "\n')
    elif 'v20:.4f if v20' in line:
        new_lines.append('                f"| {v20:.4f}" if v20 is not None else "| — "\n')
    elif 'univ:.4f if univ' in line:
        pass
    elif 'rate:.1%' in line and 'is not None else' in line:
        new_lines.append('        lines.append(f"| `{t}` | {f\\"{rate:.1%}\\" if rate is not None else \\"—\\"} |")\n')
    elif 'query=' in line and 'neighbour_mean=' in line and 'is not None else' in line:
        new_lines.append('            lines.append(f"  - `{t}`: query={v if v is not None else \\"—\\"}  neighbour_mean={m if m is not None else \\"—\\"}")\n')
    elif 'mr:.4f if mr' in line:
        new_lines.append('                f"| {mr:.4f}" if mr is not None else "| — "\n')
    elif 'mnr:.4f if mnr' in line:
        new_lines.append('                f"| {mnr:.4f}" if mnr is not None else "| — "\n')
    elif 'up:.4f if up' in line:
        new_lines.append('                f"| {up:.4f} |" if up is not None else "| — |"\n')
    elif 'Directional agreement for `{t}`:' in line and 'is not None else' in line:
        new_lines.append('        lines.append(f"- Directional agreement for `{t}`: {rate:.1%}" if rate is not None else f"- Directional agreement for `{t}`: —")\n')
    else:
        if 'univ:.4f if univ' in line:
             new_lines.append('                f"| {univ:.4f} |" if univ is not None else "| — |"\n            )\n')
        else:
             new_lines.append(line)

with open('scripts/run_final_direction.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
