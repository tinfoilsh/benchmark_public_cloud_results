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
    """Get display name for a model, falling back to the model name if not configured"""
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


def collect_single_request_data():
    """Collect single request latency data for 10000 input, 100 output tokens"""
    cc_results_dir = Path('./cc-results')
    no_cc_results_dir = Path('./no-cc-results')
    excluded_models, _, _, _, _ = load_config()

    input_len = 10000
    output_len = 100

    all_data = {'cc': {}, 'no_cc': {}}

    for results_dir, cc_key in [(cc_results_dir, 'cc'), (no_cc_results_dir, 'no_cc')]:
        if not results_dir.exists():
            continue

        for result_group in results_dir.iterdir():
            if result_group.is_dir() and result_group.name.endswith('_results'):
                serve_dir = result_group / 'serve'
                if serve_dir.exists():
                    for model_dir in serve_dir.iterdir():
                        if model_dir.is_dir():
                            model_name = model_dir.name
                            if model_name in excluded_models:
                                continue
                            filename = f"single_request_random_{input_len}_{output_len}.json"
                            filepath = model_dir / filename
                            if filepath.exists():
                                json_data = load_json_data(filepath)
                                if json_data:
                                    duration = json_data.get('duration', 1)
                                    total_output = json_data.get('total_output_tokens', 0)
                                    output_throughput = total_output / duration if duration > 0 else 0
                                    all_data[cc_key][model_name] = {
                                        'mean_itl_ms': float(json_data.get('mean_itl_ms', 0) or 0),
                                        'mean_ttft_ms': float(json_data.get('mean_ttft_ms', 0) or 0),
                                        'mean_e2el_ms': float(json_data.get('mean_e2el_ms', 0) or 0),
                                        'output_throughput': output_throughput,
                                    }

    return all_data


def create_single_request_bar_plots(all_data):
    """Create bar plots for single request latency (10k input, 100 output)"""
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

    pdf = matplotlib.backends.backend_pdf.PdfPages("single_request_latency.pdf")
    _, model_order, display_names, model_colors, mode_labels = load_config()

    all_models = set(all_data['cc'].keys()) | set(all_data['no_cc'].keys())
    all_models = sort_models_by_config(all_models, model_order)

    page_model_colors = {}
    models_without_color = []
    for m in all_models:
        if m in model_colors:
            page_model_colors[m] = model_colors[m]
        else:
            models_without_color.append(m)

    if models_without_color:
        try:
            model_palette = sns.color_palette(PALETTE_NAME, n_colors=max(1, len(models_without_color))) if have_sns else None
        except Exception:
            model_palette = None
        if model_palette is None:
            try:
                cmap = plt.colormaps.get_cmap(PALETTE_NAME)
            except Exception:
                cmap = plt.colormaps.get_cmap('tab10')
            n = max(1, len(models_without_color))
            colors_attr = getattr(cmap, 'colors', None)
            if colors_attr and len(colors_attr) >= n:
                model_palette = list(colors_attr[:n])
            else:
                xs = np.linspace(0.15, 0.95, num=n)
                model_palette = [cmap(float(x)) for x in xs]
        for i, m in enumerate(models_without_color):
            page_model_colors[m] = model_palette[i]

    def format_value(val: float, is_throughput: bool) -> str:
        if is_throughput:
            if val >= 1000:
                return f"{val/1000:.1f}k"
            return f"{val:.0f}"
        else:
            if val >= 1000:
                return f"{val/1000:.1f}s"
            return f"{val:.0f}"

    metrics = [
        ('mean_ttft_ms', 'Time to First Token (TTFT)', 'ms', False),
        ('mean_itl_ms', 'Inter-Token Latency (ITL)', 'ms', False),
        ('mean_e2el_ms', 'End-to-End Latency', 'ms', False)
    ]

    for metric_key, metric_title, metric_unit, is_throughput in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_facecolor('#fafafa')

        cc_models = set(all_data.get('cc', {}).keys())
        no_cc_models = set(all_data.get('no_cc', {}).keys())
        union_models = sort_models_by_config(cc_models | no_cc_models, model_order)

        n_models = max(1, len(union_models))
        pair_gap_ratio = 0.0
        group_gap_ratio = 0.60
        denom = (2 * n_models) + (n_models - 1) * group_gap_ratio
        bar_width = min(0.12, 0.8 / denom)
        pair_width = 2 * bar_width + pair_gap_ratio * bar_width
        group_gap = group_gap_ratio * bar_width
        total_span = n_models * pair_width + (n_models - 1) * group_gap
        base_start = -total_span / 2.0

        x = np.arange(1)

        for j, model in enumerate(union_models):
            group_left = base_start + j * (pair_width + group_gap)
            cc_center = group_left + bar_width / 2.0
            no_cc_center = group_left + (3 * bar_width) / 2.0 + pair_gap_ratio * bar_width / 2.0

            mcolor = page_model_colors.get(model)
            display_name = get_display_name(model, display_names)

            if model in all_data.get('cc', {}):
                cc_val = all_data['cc'][model].get(metric_key, 0)
                cc_bars = ax.bar(
                    x + cc_center, [cc_val], bar_width,
                    label=f'{display_name} ({mode_labels["cc"]})',
                    color=mcolor,
                    alpha=0.95,
                    edgecolor=mcolor,
                    linewidth=0,
                    zorder=3
                )
                hatch_color = '#333333'
                ax.bar(
                    x + cc_center, [cc_val], bar_width,
                    facecolor='none', edgecolor=hatch_color, hatch='///', linewidth=0, zorder=4
                )
                for bar in cc_bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                format_value(cc_val, is_throughput), ha='center', va='bottom', fontsize=8,
                                color='#222222')

            if model in no_cc_models:
                no_cc_val = all_data['no_cc'][model].get(metric_key, 0)
                no_cc_bars = ax.bar(
                    x + no_cc_center, [no_cc_val], bar_width,
                    label=f'{display_name} ({mode_labels["no_cc"]})',
                    color=mcolor,
                    alpha=0.95,
                    edgecolor=mcolor,
                    linewidth=0,
                    zorder=3
                )
                for bar in no_cc_bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                format_value(no_cc_val, is_throughput), ha='center', va='bottom', fontsize=8,
                                color='#222222')

        ax.set_ylabel(f'{metric_title} ({metric_unit})', fontsize=12, fontfamily='serif')
        ax.set_xticks(x)
        ax.set_xticklabels(['Single Request'], fontfamily='serif')
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
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

        plt.suptitle(f'{metric_title} (10k input tokens, 100 output tokens)',
                    fontsize=14, fontweight='bold', y=0.96, fontfamily='serif')
        plt.subplots_adjust(top=0.90, bottom=0.26, left=0.08, right=0.95)

        try:
            if have_sns:
                sns.despine(fig=fig)
        except Exception:
            pass

        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)
        print(f"✓ Created {metric_title} plot")

    pdf.close()
    return all_data


def print_single_request_summary(all_data):
    """Print summary of single request data"""
    print("\n" + "="*80)
    print("SINGLE REQUEST LATENCY SUMMARY (10k input, 100 output tokens)")
    print("="*80)
    _, model_order, display_names, _, mode_labels = load_config()

    for mode in ['cc', 'no_cc']:
        mode_label = mode_labels[mode]
        print(f"\n{mode_label}:")
        print("-"*80)

        models = sort_models_by_config(all_data[mode].keys(), model_order)
        for model in models:
            display_name = get_display_name(model, display_names)
            m = all_data[mode][model]
            print(
                f"  {display_name:30} TTFT: {m.get('mean_ttft_ms', 0):8.2f}  "
                f"ITL: {m.get('mean_itl_ms', 0):8.2f}  E2E: {m.get('mean_e2el_ms', 0):8.2f}"
            )


def print_performance_differences(all_data):
    """Print performance differences between CC and No-CC"""
    print("\n" + "="*80)
    print("CC VS NO-CC PERFORMANCE OVERHEAD (10k input, 100 output tokens)")
    print("="*80)
    print("\nOverhead = ((CC - No-CC) / No-CC) * 100%")
    print("Positive values mean CC is slower (worse)\n")
    _, model_order, display_names, _, _ = load_config()

    all_models = sort_models_by_config(set(all_data['cc'].keys()) & set(all_data['no_cc'].keys()), model_order)

    metrics = [
        ('mean_ttft_ms', 'TTFT (ms)'),
        ('mean_itl_ms', 'ITL (ms)'),
        ('mean_e2el_ms', 'E2E Latency (ms)')
    ]

    for model in all_models:
        display_name = get_display_name(model, display_names)
        print(f"\n{display_name}:")
        print("-"*80)

        for metric_key, metric_label in metrics:
            cc_val = all_data['cc'][model].get(metric_key, 0)
            no_cc_val = all_data['no_cc'][model].get(metric_key, 0)

            if no_cc_val > 0:
                overhead_pct = ((cc_val - no_cc_val) / no_cc_val) * 100
                print(f"  {metric_label:25} {overhead_pct:6.2f}%")
            else:
                print(f"  {metric_label:25} No data available")


if __name__ == "__main__":
    print("Collecting single request latency data...")
    all_data = collect_single_request_data()

    print("\nCreating single request latency bar plots...")
    create_single_request_bar_plots(all_data)

    print_single_request_summary(all_data)

    print_performance_differences(all_data)

    print("\n✓ All single request charts saved to: single_request_latency.pdf")
