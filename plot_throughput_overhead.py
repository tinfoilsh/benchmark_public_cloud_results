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
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def load_config():
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

def sort_models_by_config(models, model_order):
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

def get_display_name(model, display_names):
    return display_names.get(model, model)

def collect_concurrency_data():
    results_dir = Path('./results_gcp')
    excluded_models, _, _, _, _ = load_config()

    concurrency_levels = [1, 50, 100]

    data = {}
    for concurrency in concurrency_levels:
        data[concurrency] = {'cc': {}, 'no_cc': {}}

    if not results_dir.exists():
        print("Results directory not found")
        return data

    for concurrency in concurrency_levels:
        for file_path in results_dir.glob(f"*_summarization_rate{concurrency}.json"):
            filename = file_path.name

            is_non_cc = "_non-cc_" in filename

            if is_non_cc:
                parts = filename.split("_non-cc_", 1)
                model = parts[0].replace("results_", "")
            else:
                model = filename.replace("results_", "").replace(f"_summarization_rate{concurrency}.json", "")

            if model in excluded_models:
                continue

            mode = 'no_cc' if is_non_cc else 'cc'

            json_data = load_json_data(file_path)
            if json_data:
                duration = json_data.get('duration', 1)
                total_input = json_data.get('total_input_tokens', 0)
                total_output = json_data.get('total_output_tokens', 0)
                input_throughput = total_input / duration if duration > 0 else 0
                output_throughput = total_output / duration if duration > 0 else 0

                data[concurrency][mode][model] = {
                    'input_throughput': input_throughput,
                    'output_throughput': output_throughput
                }

    return data

def create_concurrency_plot(data):
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

    pdf = matplotlib.backends.backend_pdf.PdfPages("results_throughput_overhead.pdf")

    def format_rate(val: float) -> str:
        if val >= 1_000_000:
            return f"{val/1_000_000:.1f}M"
        if val >= 1_000:
            return f"{val/1_000:.1f}k"
        return f"{val:.0f}"

    _, model_order, display_names, model_colors, mode_labels = load_config()

    all_models = set()
    for concurrency in [1, 50, 100]:
        all_models.update(data[concurrency]['cc'].keys())
        all_models.update(data[concurrency]['no_cc'].keys())

    models = sort_models_by_config(all_models, model_order)

    page_model_colors = {}
    models_without_color = []
    for m in models:
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

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_facecolor('#fafafa')

    concurrency_levels = [1, 50, 100]
    x = np.arange(len(concurrency_levels))

    n_models = len(models)
    group_gap_ratio = 0.60
    bar_width = 0.8 / (n_models + (n_models - 1) * group_gap_ratio)
    group_gap = group_gap_ratio * bar_width
    total_span = n_models * bar_width + (n_models - 1) * group_gap
    base_start = -total_span / 2.0

    for j, model in enumerate(models):
        offset = base_start + j * (bar_width + group_gap) + bar_width / 2.0

        overhead_values = []

        for concurrency in concurrency_levels:
            cc_throughput = 0
            no_cc_throughput = 0

            if model in data[concurrency]['cc']:
                cc_throughput = data[concurrency]['cc'][model]['input_throughput']
            if model in data[concurrency]['no_cc']:
                no_cc_throughput = data[concurrency]['no_cc'][model]['input_throughput']

            if no_cc_throughput > 0:
                overhead_pct = ((no_cc_throughput - cc_throughput) / no_cc_throughput) * 100
                overhead_values.append(overhead_pct)
            else:
                overhead_values.append(0)

        mcolor = page_model_colors.get(model)
        display_name = get_display_name(model, display_names)

        bars = ax.bar(
            x + offset, overhead_values, bar_width,
            label=display_name,
            color=mcolor,
            alpha=0.95,
            edgecolor=mcolor,
            linewidth=0,
            zorder=3
        )
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=8,
                        color='#222222')

    ax.set_ylabel('CC Overhead (%)', fontsize=12, fontfamily='serif')
    ax.set_xlabel('Request Rate', fontsize=12, fontfamily='serif')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in concurrency_levels], fontfamily='serif')
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('serif')
    ax.grid(axis='y')
    ax.margins(y=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)

    from matplotlib.patches import Patch
    model_legend_elements = [Patch(facecolor=page_model_colors.get(m), label=get_display_name(m, display_names)) for m in models]
    ncols = max(3, min(len(model_legend_elements), 6)) if model_legend_elements else 3
    fig.legend(handles=model_legend_elements, loc='lower center', ncol=ncols,
               fontsize=9, bbox_to_anchor=(0.5, 0.08), borderaxespad=1.0,
               prop={'family': 'serif'})
    plt.suptitle(f"CC Throughput Overhead (Random 4000 ⇒ 1000)", fontsize=16,
                 fontweight='bold', y=0.96, fontfamily='serif')
    plt.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.95)
    try:
        if have_sns:
            sns.despine(fig=fig)
    except Exception:
        pass
    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.35)
    plt.close(fig)
    print(f"✓ Created throughput concurrency plot")

    pdf.close()

    return data

def print_summary(data):
    print("\n" + "="*80)
    print("THROUGHPUT OVERHEAD SUMMARY (Random 4000 ⇒ 1000)")
    print("="*80)

    _, model_order, display_names, _, mode_labels = load_config()

    all_models = set()
    for concurrency in [1, 50, 100]:
        all_models.update(data[concurrency]['cc'].keys())
        all_models.update(data[concurrency]['no_cc'].keys())

    models = sort_models_by_config(all_models, model_order)

    for concurrency in [1, 50, 100]:
        print(f"\n{concurrency} CONCURRENT REQUESTS")
        print("-"*80)

        if 'cc' in data[concurrency] and data[concurrency]['cc']:
            print(f"  {mode_labels['cc']}:")
            for model in models:
                if model in data[concurrency]['cc']:
                    display_name = get_display_name(model, display_names)
                    print(f"    {display_name:30} Input: {data[concurrency]['cc'][model]['input_throughput']:7.1f}")

        if 'no_cc' in data[concurrency] and data[concurrency]['no_cc']:
            print(f"  {mode_labels['no_cc']}:")
            for model in models:
                if model in data[concurrency]['no_cc']:
                    display_name = get_display_name(model, display_names)
                    print(f"    {display_name:30} Input: {data[concurrency]['no_cc'][model]['input_throughput']:7.1f}")

        print("  CC Overhead:")
        for model in models:
            cc_throughput = 0
            no_cc_throughput = 0

            if model in data[concurrency]['cc']:
                cc_throughput = data[concurrency]['cc'][model]['input_throughput']
            if model in data[concurrency]['no_cc']:
                no_cc_throughput = data[concurrency]['no_cc'][model]['input_throughput']

            if no_cc_throughput > 0:
                overhead_pct = ((no_cc_throughput - cc_throughput) / no_cc_throughput) * 100
                display_name = get_display_name(model, display_names)
                print(f"    {display_name:30} Input: {overhead_pct:+6.1f}%")

if __name__ == "__main__":
    print("Collecting throughput data by concurrency...")
    data = collect_concurrency_data()

    print("\nCreating throughput concurrency plot...")
    create_concurrency_plot(data)

    print_summary(data)

    print("\n✓ Chart saved to: results_throughput_overhead.pdf")
