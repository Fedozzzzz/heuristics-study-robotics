"""Строит графики сравнения трёх эвристик на обновлённой модели (4->100, 30 seed)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'core'))
import compare_direct_inverse_100cargos as m

m.N_SEEDS = 30
m.N_CARGOS_MAX = 100
os.makedirs(os.path.join(os.path.dirname(__file__), 'outputs'), exist_ok=True)

t0 = time.time()
cargo_range, results = m.sweep_n_cargos(n_seeds=30, use_penalty=False)
print(f'Sweep done in {time.time()-t0:.1f}s')

OUT = 'outputs'
m.plot_single_heuristic(cargo_range, results, 'direct',
    f'{OUT}/new_model_direct_4to100.png')
m.plot_single_heuristic(cargo_range, results, 'inverse',
    f'{OUT}/new_model_inverse_4to100.png')
m.plot_single_heuristic(cargo_range, results, 'ratio',
    f'{OUT}/new_model_ratio_4to100.png')
m.plot_gap_both(cargo_range, results,
    f'{OUT}/new_model_gap_all3_4to100.png')
print('Done.')
