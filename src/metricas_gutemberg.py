# =============================================================================
# Rotina de cálculo de métricas hidrológicas
#
# MÉTRICAS IMPLEMENTADAS:
#
# 1) NSE – Nash–Sutcliffe Efficiency
#    Fórmula:
#        NSE = 1 - [ Σ (Qobs_t - Qsim_t)² ] / [ Σ (Qobs_t - mean(Qobs))² ]
#    Objetivo:
#        Mede o quão bem a previsão reproduz as observações em comparação
#        com o uso da média observada como referência.
#        Valores próximos de 1 indicam previsões muito boas.
#        Valores negativos indicam que prever a média seria melhor.
#
# 2) KGE – Kling–Gupta Efficiency
#    Fórmula:
#        KGE = 1 - sqrt( (r - 1)² + (β - 1)² + (γ - 1)² )
#        onde:
#            r   = correlação de Pearson(obs, sim)
#            β   = média(sim) / média(obs)  (viés relativo)
#            γ   = (CV(sim)) / (CV(obs))    (razão da variabilidade)
#    Objetivo:
#        Fornece uma avaliação mais equilibrada que o NSE,
#        integrando correlação (força linear), viés e variabilidade.
#        Valores próximos de 1 indicam alto desempenho em todos os aspectos.
#
# 3) RMSE – Root Mean Squared Error
#    Fórmula:
#        RMSE = sqrt( Σ (Qobs_t - Qsim_t)² / N )
#    Objetivo:
#        Mede o erro médio quadrático em unidades da variável (ex.: m³/s).
#        Penaliza fortemente grandes desvios. Útil para avaliar extremos.
#
# 4) MAE – Mean Absolute Error
#    Fórmula:
#        MAE = (1/N) Σ |Qobs_t - Qsim_t|
#    Objetivo:
#        Mede o erro médio absoluto, sem dar peso extra para grandes erros.
#        Mais robusto a outliers que o RMSE.
#
# 5) R² – Coeficiente de Determinação
#    Fórmula:
#        R² = [ cov(Qobs, Qsim)² ] / [ var(Qobs) * var(Qsim) ]
#    Objetivo:
#        Mede a proporção da variância observada que é explicada pelo modelo.
#        Valores próximos de 1 indicam forte ajuste linear.
#        Não distingue viés sistemático (pode dar alto valor mesmo com desvio).
#
# 6) PBIAS – Percent Bias
#    Fórmula:
#        PBIAS = 100 * Σ(Qobs_t - Qsim_t) / Σ(Qobs_t)
#    Objetivo:
#        Mede a tendência sistemática do modelo.
#        Valores positivos → subestimação (modelo prevê menos vazão).
#        Valores negativos → superestimação (modelo prevê mais vazão).
#
# Entradas: arrays/listas de valores observados (Qobs) e simulados (Qsim).
# =============================================================================

import numpy as np
from scipy.stats import pearsonr

def nse(Qobs, Qsim):
    """Nash–Sutcliffe Efficiency"""
    Qobs, Qsim = np.array(Qobs), np.array(Qsim)
    return 1 - np.sum((Qobs - Qsim)**2) / np.sum((Qobs - np.mean(Qobs))**2)

def kge(Qobs, Qsim):
    """Kling–Gupta Efficiency"""
    Qobs, Qsim = np.array(Qobs), np.array(Qsim)
    r, _ = pearsonr(Qsim, Qobs)
    beta = np.mean(Qsim) / np.mean(Qobs)
    gamma = (np.std(Qsim) / np.mean(Qsim)) / (np.std(Qobs) / np.mean(Qobs))
    return 1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)

def rmse(Qobs, Qsim):
    """Root Mean Squared Error"""
    Qobs, Qsim = np.array(Qobs), np.array(Qsim)
    return np.sqrt(np.mean((Qobs - Qsim)**2))

def mae(Qobs, Qsim):
    """Mean Absolute Error"""
    Qobs, Qsim = np.array(Qobs), np.array(Qsim)
    return np.mean(np.abs(Qobs - Qsim))

def r2(Qobs, Qsim):
    """Coeficiente de Determinação R²"""
    Qobs, Qsim = np.array(Qobs), np.array(Qsim)
    corr_matrix = np.corrcoef(Qobs, Qsim)
    return corr_matrix[0,1]**2

def pbias(Qobs, Qsim):
    """Percent Bias"""
    Qobs, Qsim = np.array(Qobs), np.array(Qsim)
    return 100.0 * np.sum(Qobs - Qsim) / np.sum(Qobs)
