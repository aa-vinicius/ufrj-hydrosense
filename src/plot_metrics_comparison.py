import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics_comparison(model1_metrics_path, model2_metrics_path, output_dir='outputs'):
    """Gera um arquivo de plot por métrica comparando treino vs teste para cada algoritmo e estação.

    Para cada métrica (RMSE, MAE, R2, KGE, PBIAS, Bias, Nash_Sutcliffe):
      - Cria uma figura com subplots (1 por algoritmo)
      - Cada subplot contém barras lado a lado (Train/Test) por estação
      - Para Nash_Sutcliffe plota apenas valores positivos (> 0)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Carrega dados
    model1_df = pd.read_csv(model1_metrics_path)
    model2_df = pd.read_csv(model2_metrics_path)

    # Uniformiza nome da coluna de estação
    model1_df = model1_df.rename(columns={'station_id': 'Flow_Station'})
    if 'station_id' in model2_df.columns:
        model2_df = model2_df.rename(columns={'station_id': 'Flow_Station'})

    metrics = ['RMSE', 'MAE', 'R2', 'KGE', 'PBIAS', 'Bias', 'Nash_Sutcliffe']

    # Pré-calcula limites globais por métrica (após regras de filtragem para NSE)
    global_metric_limits = {}
    for metric in metrics:
        if metric == 'Nash_Sutcliffe':
            # Apenas valores positivos contam
            vals1 = model1_df[model1_df['Nash_Sutcliffe'] > 0]['Nash_Sutcliffe']
            vals2 = model2_df[model2_df['Nash_Sutcliffe'] > 0]['Nash_Sutcliffe']
        else:
            vals1 = model1_df[metric] if metric in model1_df.columns else pd.Series(dtype=float)
            vals2 = model2_df[metric] if metric in model2_df.columns else pd.Series(dtype=float)
        combined = pd.concat([vals1, vals2], ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
        if combined.empty:
            global_metric_limits[metric] = (0, 1)  # fallback
        else:
            ymin = combined.min()
            ymax = combined.max()
            if np.isclose(ymin, ymax):
                # Expande ligeiramente se todos valores iguais
                delta = abs(ymin) * 0.05 if ymin != 0 else 0.1
                ymin -= delta
                ymax += delta
            # Pequena margem visual
            span = ymax - ymin
            global_metric_limits[metric] = (ymin - 0.05*span, ymax + 0.05*span)

    def generate_plots(df, label_prefix):
        algos = sorted(df['Model'].unique())
        stations_all = sorted(df['Flow_Station'].unique())
        for metric in metrics:
            # Filtragem especial para NSE
            metric_df = df.copy()
            if metric == 'Nash_Sutcliffe':
                metric_df = metric_df[metric_df['Nash_Sutcliffe'] > 0]
            if metric_df.empty:
                print(f"[Info] Nenhum valor positivo para {metric} em {label_prefix}, plot será omitido.")
                continue
            stations = sorted(metric_df['Flow_Station'].unique())
            fig, axes = plt.subplots(1, len(algos), figsize=(5*len(algos), 5), sharey=False)
            if len(algos) == 1:
                axes = [axes]
            ylims = global_metric_limits.get(metric, None)
            for i, algo in enumerate(algos):
                ax = axes[i]
                algo_df = metric_df[metric_df['Model'] == algo]
                train_rows = algo_df[algo_df['Dataset'] == 'Training']
                test_rows = algo_df[algo_df['Dataset'] == 'Test']
                train_vals = []
                test_vals = []
                for st in stations:
                    tr = train_rows[train_rows['Flow_Station'] == st]
                    te = test_rows[test_rows['Flow_Station'] == st]
                    train_vals.append(tr[metric].values[0] if not tr.empty else np.nan)
                    test_vals.append(te[metric].values[0] if not te.empty else np.nan)
                x = np.arange(len(stations))
                width = 0.4
                ax.bar(x - width/2, train_vals, width, label='Train', alpha=0.7)
                ax.bar(x + width/2, test_vals, width, label='Test', alpha=0.5)
                ax.set_xticks(x)
                ax.set_xticklabels([str(s) for s in stations], rotation=45)
                ax.set_title(f'{algo}')
                ax.grid(True, alpha=0.3)
                if metric == 'PBIAS':
                    ax.axhline(0, color='black', linewidth=0.8)
                if ylims:
                    ax.set_ylim(ylims)
            fig.suptitle(f'{label_prefix} - {metric} (Train vs Test)', fontsize=14, fontweight='bold')
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right')
            fig.tight_layout(rect=[0, 0, 0.95, 0.92])
            out_path = os.path.join(output_dir, f'{label_prefix.lower().replace(" ", "_")}_metric_{metric}.png')
            plt.savefig(out_path, bbox_inches='tight')
            plt.close()
            print(f'Plot salvo: {out_path}')

    generate_plots(model1_df, 'Model 1')
    generate_plots(model2_df, 'Model 2')

if __name__ == "__main__":
    plot_metrics_comparison(
        model1_metrics_path="outputs/model1_performance_metrics.csv",
        model2_metrics_path="outputs/model2_error_prediction_metrics.csv"
    )
