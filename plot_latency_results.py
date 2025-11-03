import json
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'grid.color': '#999999',
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'legend.frameon': False,
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'Liberation Serif', 'Computer Modern Roman'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

PALETTE_NAME = 'Set2'


def _format_ms(val: float) -> str:
    """Format milliseconds nicely for bar labels."""
    try:
        if val >= 1000:
            return f"{val/1000:.1f}s"
        if val >= 100:
            return f"{val:.0f}"
        return f"{val:.1f}"
    except Exception:
        return str(val)


def _annotate_bars(ax, bars, values):
    """Add numeric labels on top of bars."""
    for bar, v in zip(bars, values):
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                _format_ms(v),
                ha='center', va='bottom', fontsize=8, color='#222222'
            )


def load_json_data(filepath):
    """Load JSON data from file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def load_config():
    """Load configuration from plot_config.json"""
    config_path = Path('./plot_config.json')
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                excluded = config.get('excluded_models', [])
                model_order = config.get('model_order', [])
                display_names = config.get('model_display_names', {})
                model_colors = config.get('model_colors', {})
                mode_labels = config.get('mode_labels', {'cc': 'CC', 'no_cc': 'No CC'})
                if excluded:
                    print(f"Excluding models from plots: {', '.join(excluded)}")
                if model_order:
                    print(f"Using custom model order from config")
                if display_names:
                    print(f"Using custom display names for {len(display_names)} models")
                if model_colors:
                    print(f"Using custom colors for {len(model_colors)} models")
                return set(excluded), model_order, display_names, model_colors, mode_labels
        except Exception as e:
            print(f"Warning: Could not load plot_config.json: {e}")
            return set(), [], {}, {}, {'cc': 'CC', 'no_cc': 'No CC'}
    return set(), [], {}, {}, {'cc': 'CC', 'no_cc': 'No CC'}


def get_display_name(model, display_names):
    """Get display name for a model from display_names dict, or return model name if not found"""
    return display_names.get(model, model)


def sort_models_by_config(models, model_order):
    """Sort models according to config order, with unlisted models at the end alphabetically"""
    if not model_order:
        return sorted(models)

    ordered = []
    remaining = []

    for model in model_order:
        if model in models:
            ordered.append(model)

    for model in sorted(models):
        if model not in ordered:
            remaining.append(model)

    return ordered + remaining


def collect_latency_data():
    """Collect latency metrics for each scenario separately"""
    cc_results_dir = Path('./cc-results')
    no_cc_results_dir = Path('./no-cc-results')
    excluded_models, _, _, _, _ = load_config()

    scenarios = [
        'high_load_64concurrent',
        'max_throughput_rampup',
        'medium_load_32concurrent',
        'single_request_sharegpt'
    ]

    all_data = {}

    for scenario in scenarios:
        all_data[scenario] = {'cc': {}, 'no_cc': {}}

        for results_dir, cc_key in [(cc_results_dir, 'cc'), (no_cc_results_dir, 'no_cc')]:
            if not results_dir.exists():
                continue

            for result_group in results_dir.iterdir():
                if result_group.is_dir() and result_group.name.endswith('_results'):
                    serve_dir = result_group / 'serve'
                    if serve_dir.exists():
                        for model_dir in serve_dir.iterdir():
                            if model_dir.is_dir():
                                if model_dir.name in excluded_models:
                                    continue
                                filepath = model_dir / f"{scenario}.json"
                                if filepath.exists():
                                    json_data = load_json_data(filepath)
                                    if json_data:
                                        all_data[scenario][cc_key][model_dir.name] = {
                                            'mean_ttft_ms': float(json_data.get('mean_ttft_ms', 0) or 0),
                                            'std_ttft_ms': float(json_data.get('std_ttft_ms', 0) or 0),
                                            'mean_e2el_ms': float(json_data.get('mean_e2el_ms', 0) or 0),
                                            'std_e2el_ms': float(json_data.get('std_e2el_ms', 0) or 0),
                                        }

    return all_data


def create_latency_plots(all_data):
    """Create latency plots (TTFT and E2E) for each scenario and save to PDF"""
    try:
        import seaborn as sns
        sns.set_theme(style='darkgrid', context='talk', font='serif')
        mpl.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times', 'Liberation Serif', 'Computer Modern Roman']
        })
        have_sns = True
    except Exception:
        have_sns = False
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except Exception:
            plt.style.use('ggplot')

    pdf = matplotlib.backends.backend_pdf.PdfPages("latency_results.pdf")
    _, model_order, display_names, model_colors, mode_labels = load_config()

    scenario_titles = {
        'high_load_64concurrent': 'High Load (64 Concurrent Requests)',
        'max_throughput_rampup': 'Max Throughput (Rampup)',
        'medium_load_32concurrent': 'Medium Load (32 Concurrent Requests)',
        'single_request_sharegpt': 'Single Request (ShareGPT Dataset)'
    }

    for scenario, data in all_data.items():
        models_present = (
            set(all_data[scenario].get('cc', {}).keys()) |
            set(all_data[scenario].get('no_cc', {}).keys())
        )
        models_present = sort_models_by_config(models_present, model_order)

        page_model_colors = {}
        models_without_color = []
        for m in models_present:
            if m in model_colors:
                page_model_colors[m] = model_colors[m]
            else:
                models_without_color.append(m)

        if models_without_color:
            try:
                page_palette = sns.color_palette(PALETTE_NAME, n_colors=max(1, len(models_without_color))) if have_sns else None
            except Exception:
                page_palette = None
            if page_palette is None:
                try:
                    cmap = plt.colormaps.get_cmap(PALETTE_NAME)
                except Exception:
                    cmap = plt.colormaps.get_cmap('tab10')
                n = max(1, len(models_without_color))
                colors_attr = getattr(cmap, 'colors', None)
                if colors_attr and len(colors_attr) >= n:
                    page_palette = list(colors_attr[:n])
                else:
                    xs = np.linspace(0.15, 0.95, num=n)
                    page_palette = [cmap(float(x)) for x in xs]
            for i, m in enumerate(models_without_color):
                page_model_colors[m] = page_palette[i]

        fig, (ax_ttft, ax_e2e) = plt.subplots(1, 2, figsize=(16, 8))

        cc_models = set(all_data[scenario].get('cc', {}).keys())
        no_cc_models = set(all_data[scenario].get('no_cc', {}).keys())
        union_models = sort_models_by_config(cc_models | no_cc_models, model_order)

        # TTFT (CC vs No CC per model)
        ax = ax_ttft
        ax.set_facecolor('#fafafa')
        if union_models:
            x_pos = np.arange(len(union_models))
            width = 0.35
            vals_cc = [all_data[scenario]['cc'][m]['mean_ttft_ms'] if m in all_data[scenario].get('cc', {}) else 0 for m in union_models]
            err_cc = [all_data[scenario]['cc'][m]['std_ttft_ms'] if m in all_data[scenario].get('cc', {}) else 0 for m in union_models]
            colors = [page_model_colors.get(m) for m in union_models]
            bars_cc = ax.bar(x_pos - width/2, vals_cc, width,
                             color=colors, alpha=0.95, edgecolor=colors, linewidth=0, zorder=3)
            hatch_color = '#333333'
            ax.bar(x_pos - width/2, vals_cc, width,
                   facecolor='none', edgecolor=hatch_color, hatch='///', linewidth=0, zorder=4)
            _annotate_bars(ax, bars_cc, vals_cc)
            vals_no_cc = [all_data[scenario]['no_cc'][m]['mean_ttft_ms'] if m in all_data[scenario].get('no_cc', {}) else 0 for m in union_models]
            err_no_cc = [all_data[scenario]['no_cc'][m]['std_ttft_ms'] if m in all_data[scenario].get('no_cc', {}) else 0 for m in union_models]
            bars_no_cc = ax.bar(x_pos + width/2, vals_no_cc, width,
                                color=colors, alpha=0.95, edgecolor=colors, linewidth=0, zorder=3)
            _annotate_bars(ax, bars_no_cc, vals_no_cc)
            ax.set_xticks([])
        ax.set_ylabel('TTFT (ms)', fontsize=12, fontfamily='serif')
        ax.set_title('Time to First Token', fontsize=14, fontweight='bold', fontfamily='serif')
        for tick in ax.get_yticklabels():
            tick.set_fontfamily('serif')
        ax.grid(axis='y')
        ax.margins(y=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)

        # E2E (CC vs No CC per model)
        ax = ax_e2e
        ax.set_facecolor('#fafafa')
        if union_models:
            x_pos = np.arange(len(union_models))
            width = 0.35
            vals_cc = [all_data[scenario]['cc'][m]['mean_e2el_ms'] if m in all_data[scenario].get('cc', {}) else 0 for m in union_models]
            err_cc = [all_data[scenario]['cc'][m]['std_e2el_ms'] if m in all_data[scenario].get('cc', {}) else 0 for m in union_models]
            colors = [page_model_colors.get(m) for m in union_models]
            bars_cc = ax.bar(x_pos - width/2, vals_cc, width,
                             color=colors, alpha=0.95, edgecolor=colors, linewidth=0, zorder=3)
            hatch_color = '#333333'
            ax.bar(x_pos - width/2, vals_cc, width,
                   facecolor='none', edgecolor=hatch_color, hatch='///', linewidth=0, zorder=4)
            _annotate_bars(ax, bars_cc, vals_cc)
            vals_no_cc = [all_data[scenario]['no_cc'][m]['mean_e2el_ms'] if m in all_data[scenario].get('no_cc', {}) else 0 for m in union_models]
            err_no_cc = [all_data[scenario]['no_cc'][m]['std_e2el_ms'] if m in all_data[scenario].get('no_cc', {}) else 0 for m in union_models]
            bars_no_cc = ax.bar(x_pos + width/2, vals_no_cc, width,
                                color=colors, alpha=0.95, edgecolor=colors, linewidth=0, zorder=3)
            _annotate_bars(ax, bars_no_cc, vals_no_cc)
            ax.set_xticks([])
        ax.set_ylabel('E2E Latency (ms)', fontsize=12, fontfamily='serif')
        ax.set_title('End-to-End Latency', fontsize=14, fontweight='bold', fontfamily='serif')
        for tick in ax.get_yticklabels():
            tick.set_fontfamily('serif')
        ax.grid(axis='y')
        ax.margins(y=0.15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_alpha(0.3)
        ax.spines['bottom'].set_alpha(0.3)

        from matplotlib.patches import Patch
        model_legend_elements = [Patch(facecolor=page_model_colors.get(m), label=get_display_name(m, display_names)) for m in union_models]
        ncols = max(3, min(len(model_legend_elements), 6)) if model_legend_elements else 3
        cc_style_elements = [
            Patch(facecolor='#777777', hatch='///', edgecolor='#333333', label=mode_labels['cc']),
            Patch(facecolor='#777777', label=mode_labels['no_cc'])
        ]
        fig.legend(handles=cc_style_elements, loc='lower center', ncol=2,
                   fontsize=9, bbox_to_anchor=(0.5, 0.16), borderaxespad=1.0,
                   prop={'family': 'serif'})
        fig.legend(handles=model_legend_elements, loc='lower center', ncol=ncols,
                   fontsize=8, bbox_to_anchor=(0.5, 0.08), borderaxespad=1.0,
                   prop={'family': 'serif'})
        plt.suptitle(f"{scenario_titles.get(scenario, scenario)}", fontsize=16,
                     fontweight='bold', y=0.96, fontfamily='serif')
        plt.subplots_adjust(top=0.88, bottom=0.24, left=0.06, right=0.98, wspace=0.25)
        try:
            if have_sns:
                sns.despine(fig=fig)
        except Exception:
            pass
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)
        print(f"✓ Created latency plot for: {scenario_titles.get(scenario, scenario)}")

    pdf.close()


def print_latency_summary(all_data):
    print("\n" + "="*80)
    print("DETAILED LATENCY SUMMARY (ms)")
    print("="*80)
    _, model_order, display_names, _, mode_labels = load_config()

    for scenario, data in all_data.items():
        print(f"\n{scenario.upper().replace('_', ' ')}")
        print("-"*80)
        for mode in ['cc', 'no_cc']:
            if mode in data and data[mode]:
                mode_label = mode_labels[mode]
                print(f"  {mode_label}:")
                models = sort_models_by_config(data[mode].keys(), model_order)
                for model in models:
                    m = data[mode][model]
                    display_name = get_display_name(model, display_names)
                    print(
                        f"    {display_name:30} TTFT: {m['mean_ttft_ms']:8.2f} ± {m['std_ttft_ms']:7.2f}   "
                        f"E2E: {m['mean_e2el_ms']:8.2f} ± {m['std_e2el_ms']:7.2f}"
                    )


def print_performance_overhead(all_data):
    """Print performance overhead between CC and No-CC"""
    print("\n" + "="*80)
    print("CC VS NO-CC PERFORMANCE OVERHEAD")
    print("="*80)
    print("\nOverhead = ((CC - No-CC) / No-CC) * 100%")
    print("Positive values mean CC is slower (worse)\n")
    _, model_order, display_names, _, _ = load_config()

    all_models = set()
    for scenario in all_data.values():
        all_models.update(set(scenario.get('cc', {}).keys()) & set(scenario.get('no_cc', {}).keys()))
    all_models = sort_models_by_config(all_models, model_order)

    metrics = [
        ('mean_ttft_ms', 'TTFT (ms)'),
        ('mean_e2el_ms', 'E2E Latency (ms)')
    ]

    for model in all_models:
        display_name = get_display_name(model, display_names)
        print(f"\n{display_name}:")
        print("-"*80)

        for metric_key, metric_label in metrics:
            overheads = []

            for scenario in all_data.keys():
                if model in all_data[scenario].get('cc', {}) and model in all_data[scenario].get('no_cc', {}):
                    cc_val = all_data[scenario]['cc'][model][metric_key]
                    no_cc_val = all_data[scenario]['no_cc'][model][metric_key]

                    if no_cc_val > 0:
                        overhead_pct = ((cc_val - no_cc_val) / no_cc_val) * 100
                        overheads.append(overhead_pct)

            if overheads:
                mean_overhead = np.mean(overheads)
                median_overhead = np.median(overheads)
                min_overhead = np.min(overheads)
                max_overhead = np.max(overheads)

                print(f"  {metric_label:25} Mean: {mean_overhead:6.2f}%  Median: {median_overhead:6.2f}%  "
                      f"Range: [{min_overhead:6.2f}%, {max_overhead:6.2f}%]")
            else:
                print(f"  {metric_label:25} No data available")


if __name__ == "__main__":
    print("Collecting latency data for all scenarios...")
    all_data = collect_latency_data()

    print("\nCreating latency charts (TTFT and E2E) for each scenario...")
    create_latency_plots(all_data)

    print_latency_summary(all_data)

    print_performance_overhead(all_data)

    print("\n✓ All latency charts saved to: latency_results.pdf")
