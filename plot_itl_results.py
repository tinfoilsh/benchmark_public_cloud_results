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
    """Get display name for a model, falling back to model name if not in config"""
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

def collect_itl_data():
    """Collect ITL metrics for each scenario separately from the new file structure"""
    results_dir = Path('./results_gcp')
    excluded_models, _, _, _, _ = load_config()

    # Define the base scenarios
    base_scenarios = [
        "random",
        "summarization", 
        "translation",
        "sharegpt",
        "edit_10k_char",
        "numina_math",
        "reasoning"
    ]
    
    # Request rates are also part of the scenario identification
    request_rates = [100, 50, 1]
    
    # Create full scenario list (scenario_rate combinations)
    scenarios = []
    for base_scenario in base_scenarios:
        for rate in request_rates:
            scenarios.append(f"{base_scenario}_rate{rate}")

    all_data = {}

    # Initialize all_data structure
    for scenario in scenarios:
        all_data[scenario] = {'cc': {}, 'no_cc': {}}

    if not results_dir.exists():
        print("Results directory not found")
        return all_data

    # Process all JSON files in results_gcp
    for file_path in results_dir.glob("*.json"):
        filename = file_path.name
        
        # Skip non-result files
        if not filename.startswith("results_") or not filename.endswith(".json"):
            continue
            
        # Parse filename: results_{model}_{scenario}_rate{rate}.json
        # Or: results_{model}_non-cc_{scenario}_rate{rate}.json
        
        # Remove 'results_' prefix and '.json' suffix
        name_part = filename[8:-5]
        
        # Check if it's non-cc
        is_non_cc = "_non-cc_" in name_part
        
        # Extract model and scenario
        if is_non_cc:
            # Split by '_non-cc_'
            parts = name_part.split("_non-cc_", 1)
            model = parts[0]
            rest = parts[1]
            
            # Find rate part in rest
            rate_pos = rest.rfind("_rate")
            if rate_pos == -1:
                print(f"Could not find rate in file: {filename}")
                continue
                
            # Extract scenario name (without rate)
            scenario_name = rest[:rate_pos]
            rate_part = rest[rate_pos+1:]  # Include 'rate' part
            
            # Reconstruct full scenario name
            scenario = f"{scenario_name}_{rate_part}"
        else:
            # For CC files, find the rate part first
            rate_pos = name_part.rfind("_rate")
            if rate_pos == -1:
                print(f"Could not find rate in file: {filename}")
                continue
                
            # Extract the rate part
            rate_part = name_part[rate_pos+1:]  # Include the 'rate' part
            
            # Everything before the rate part is the model + scenario
            model_scenario_part = name_part[:rate_pos]
            
            # Now we need to separate model from scenario
            # We look for the longest matching scenario from the end of the string
            scenario_found = None
            model_part = ""
            
            # Sort scenarios by length descending to match longer scenarios first
            # (e.g., match "edit_10k_char" before "char")
            sorted_scenarios = sorted(base_scenarios, key=len, reverse=True)
            
            for base_scenario in sorted_scenarios:
                if base_scenario in model_scenario_part:
                    scenario_found = f"{base_scenario}_{rate_part}"
                    # Extract model part (everything before the scenario)
                    scenario_start = model_scenario_part.rfind(base_scenario)
                    if scenario_start > 0:
                        model_part = model_scenario_part[:scenario_start-1]  # -1 to remove the underscore
                    else:
                        model_part = model_scenario_part
                    break
            
            if scenario_found is None:
                print(f"Could not determine scenario for file: {filename}")
                continue
                
            model = model_part
            scenario = scenario_found
        
        # Skip excluded models
        if model in excluded_models:
            continue
            
        # Check if scenario is valid
        if scenario not in scenarios:
            # Skip unknown scenarios
            continue
        
        # Determine CC mode
        mode = 'no_cc' if is_non_cc else 'cc'
        
        # Load JSON data
        json_data = load_json_data(file_path)
        if json_data:
            # Initialize model entry if not exists
            if model not in all_data[scenario][mode]:
                all_data[scenario][mode][model] = {
                    'mean_itl': 0,
                    'median_itl': 0,
                    'p50_itl': 0,
                    'p90_itl': 0,
                    'p95_itl': 0,
                    'p99_itl': 0,
                    'std_itl': 0
                }
            
            # Update with data from JSON
            all_data[scenario][mode][model] = {
                'mean_itl': float(json_data.get('mean_itl_ms', 0) or 0),
                'median_itl': float(json_data.get('median_itl_ms', 0) or 0),
                'p50_itl': float(json_data.get('p50_itl_ms', 0) or 0),
                'p90_itl': float(json_data.get('p90_itl_ms', 0) or 0),
                'p95_itl': float(json_data.get('p95_itl_ms', 0) or 0),
                'p99_itl': float(json_data.get('p99_itl_ms', 0) or 0),
                'std_itl': float(json_data.get('std_itl_ms', 0) or 0)
            }

    return all_data

def create_itl_plots(all_data):
    """Create ITL plots for each scenario and save to PDF."""
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

    pdf = matplotlib.backends.backend_pdf.PdfPages("itl_results.pdf")
    _, model_order, display_names, model_colors, mode_labels = load_config()

    # Create scenario titles for the new naming convention
    base_scenario_titles = {
        'random': 'Random Generation',
        'summarization': 'Summarization',
        'translation': 'Translation',
        'sharegpt': 'ShareGPT',
        'edit_10k_char': 'Edit 10K Characters',
        'numina_math': 'Numina Math',
        'reasoning': 'Reasoning'
    }
    
    # Generate titles for all scenario_rate combinations
    scenario_titles = {}
    for base_scenario, title in base_scenario_titles.items():
        for rate in [100, 50, 1]:
            scenario_key = f"{base_scenario}_rate{rate}"
            scenario_titles[scenario_key] = f"{title} (Rate {rate})"

    for scenario, data in all_data.items():
        x = np.arange(2)

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

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_facecolor('#fafafa')
        cc_models = set(all_data[scenario].get('cc', {}).keys())
        no_cc_models = set(all_data[scenario].get('no_cc', {}).keys())
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

        for j, model in enumerate(union_models):
            group_left = base_start + j * (pair_width + group_gap)
            cc_center = group_left + bar_width / 2.0
            plain_center = group_left + (3 * bar_width) / 2.0 + pair_gap_ratio * bar_width / 2.0

            cc_values = [
                all_data[scenario]['cc'][model]['mean_itl'] if model in all_data[scenario].get('cc', {}) else 0,
                all_data[scenario]['cc'][model]['p50_itl'] if model in all_data[scenario].get('cc', {}) else 0
            ]
            mcolor = page_model_colors.get(model)
            display_name = get_display_name(model, display_names)
            cc_bars = ax.bar(
                x + cc_center, cc_values, bar_width,
                label=f'{display_name} ({mode_labels["cc"]})',
                color=mcolor,
                alpha=0.95,
                edgecolor=mcolor,
                linewidth=0,
                zorder=3
            )
            hatch_color = '#333333'
            ax.bar(
                x + cc_center, cc_values, bar_width,
                facecolor='none', edgecolor=hatch_color, hatch='///', linewidth=0, zorder=4
            )
            _annotate_bars(ax, cc_bars, cc_values)

            if model in no_cc_models:
                no_cc_values = [
                    all_data[scenario]['no_cc'][model]['mean_itl'] if model in all_data[scenario].get('no_cc', {}) else 0,
                    all_data[scenario]['no_cc'][model]['p50_itl'] if model in all_data[scenario].get('no_cc', {}) else 0
                ]
                no_cc_bars = ax.bar(
                    x + plain_center, no_cc_values, bar_width,
                    label=f'{display_name} ({mode_labels["no_cc"]})',
                    color=mcolor,
                    alpha=0.95,
                    edgecolor=mcolor,
                    linewidth=0,
                    zorder=3
                )
                _annotate_bars(ax, no_cc_bars, no_cc_values)
        ax.set_ylabel('Inter-Token Latency (ms)', fontsize=12, fontfamily='serif')
        ax.set_xticks(x)
        ax.set_xticklabels(['Mean', 'Median'], fontfamily='serif')
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
        plt.suptitle(f"{scenario_titles.get(scenario, scenario)}", fontsize=16,
                     fontweight='bold', y=0.96, fontfamily='serif')
        plt.subplots_adjust(top=0.88, bottom=0.26, left=0.08, right=0.95)
        try:
            if have_sns:
                sns.despine(fig=fig)
        except Exception:
            pass
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.35)
        plt.close(fig)
        print(f"✓ Created ITL plot for: {scenario_titles.get(scenario, scenario)}")

    pdf.close()

    return all_data

def print_itl_summary(all_data):
    """Print detailed summary statistics for all scenarios"""
    print("\n" + "="*80)
    print("DETAILED ITL SUMMARY (milliseconds)")
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
                    if model in data[mode]:
                        m = data[mode][model]
                        display_name = get_display_name(model, display_names)
                        print(
                            f"    {display_name:30} Mean: {m['mean_itl']:7.1f}  Median: {m['p50_itl']:7.1f}"
                        )


def _ratio(cc_val: float, plain_val: float):
    """Compute multiplicative overhead ratio (CC/No-CC)."""
    try:
        if plain_val is None or plain_val == 0:
            return None
        return cc_val / plain_val
    except Exception:
        return None


def print_cc_vs_plain_ratio_ranges(all_data):
    """Print min/max CC-to-No-CC ratios for key metrics."""
    metrics = [
        ('mean_itl', 'Mean'),
        ('p50_itl', 'Median')
    ]

    print("\n" + "="*80)
    print("CC VS NO-CC RATIO RANGES (CC / No-CC)")
    print("="*80)
    _, model_order, display_names, _, mode_labels = load_config()

    for scenario, data in all_data.items():
        cc_models = data.get('cc', {}) or {}
        no_cc_models = data.get('no_cc', {}) or {}
        overlapping_models = sort_models_by_config(set(cc_models.keys()) & set(no_cc_models.keys()), model_order)

        print(f"\n{scenario.upper().replace('_', ' ')}")
        print("-"*80)

        if not overlapping_models:
            print("  No overlapping models with both CC and No-CC data.")
            continue

        for key, label in metrics:
            ratios = []
            for model in overlapping_models:
                cc_val = cc_models[model].get(key)
                no_cc_val = no_cc_models[model].get(key)
                ratio = _ratio(cc_val, no_cc_val)
                if ratio is not None:
                    ratios.append((ratio, model))

            if ratios:
                min_ratio = min(ratios, key=lambda item: item[0])
                max_ratio = max(ratios, key=lambda item: item[0])
                min_display = get_display_name(min_ratio[1], display_names)
                max_display = get_display_name(max_ratio[1], display_names)
                print(
                    f"  {label:>4}: min {min_ratio[0]:7.3f}x ({min_display})  "
                    f"max {max_ratio[0]:7.3f}x ({max_display})"
                )
            else:
                print(f"  {label:>4}: insufficient data to compute ratio")


def print_performance_overhead(all_data):
    """Print performance overhead between CC and No-CC"""
    print("\n" + "="*80)
    print("CC VS NO-CC PERFORMANCE OVERHEAD")
    print("="*80)
    print("\nOverhead = ((CC - No-CC) / No-CC) * 100%")
    print("Positive values mean CC is slower (worse)\n")
    _, model_order, display_names, _, mode_labels = load_config()

    all_models = set()
    for scenario in all_data.values():
        all_models.update(set(scenario.get('cc', {}).keys()) & set(scenario.get('no_cc', {}).keys()))
    all_models = sort_models_by_config(all_models, model_order)

    metrics = [
        ('mean_itl', 'Mean ITL (ms)'),
        ('p50_itl', 'Median ITL (ms)')
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
    print("Collecting ITL data for all scenarios...")
    all_data = collect_itl_data()

    print("\nCreating ITL comparison charts for each scenario...")
    create_itl_plots(all_data)

    print_itl_summary(all_data)

    print_cc_vs_plain_ratio_ranges(all_data)

    print_performance_overhead(all_data)

    print("\n✓ All ITL charts saved to: itl_results.pdf")