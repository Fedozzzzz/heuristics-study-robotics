"""
ОЦЕНКА ВЫЧИСЛИТЕЛЬНОЙ СЛОЖНОСТИ ДВУХ МОДЕЛЕЙ ПО ЗАМЕРАМ ВРЕМЕНИ РАСЧЁТА
(static_v3 -- случайная статическая, dynamic_v2 -- динамическая) численными
методами: МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ и ХИ-КВАДРАТ.

Вход -- CSV, который пишет experiments/compare_static_v3_vs_dynamic_v2.py:
колонки *_time_ms_median (точка), *_time_ms_min / *_time_ms_max (размах по
--repeats прогонам точки). Анализируется ТОЛЬКО время расчёта t(n) в
зависимости от числа грузов n; стоимость Phi здесь не рассматривается.

=============================================================================
ДВА НЕЗАВИСИМЫХ ЧИСЛЕННЫХ МЕТОДА -- ЗАЧЕМ ОБА

1. МАКСИМАЛЬНОЕ ПРАВДОПОДОБИЕ (мультипликативная лог-нормальная модель шума)

       t_i = f(n_i; theta) * exp(eps_i),   eps_i ~ N(0, s^2)

   Замер времени шумит МУЛЬТИПЛИКАТИВНО (разброс растёт вместе с самим
   временем: у dynamic_v2 при n=200 размах по прогонам в миллисекундах на
   порядок больше, чем при n=10, а в процентах -- того же порядка), поэтому
   аддитивная гауссова модель тут неуместна. Логарифмируя,

       ln t_i = ln f(n_i; theta) + eps_i,

   и максимизация правдоподобия сводится к минимизации
   S(theta) = sum (ln t_i - ln f_i)^2, при этом ML-оценка масштаба шума
   s^2 = S/N, а логарифм правдоподобия lnL = -N/2 * (ln(2*pi*s^2) + 1).
   Модели сравниваются по AIC = 2k' - 2lnL (k' = k + 1: параметры модели плюс
   сам s) и BIC; это КОРРЕКТНОЕ сравнение НЕвложенных моделей (n log n против
   n^2 и т.д.), которое штрафует лишние параметры.

   Отсюда же берётся главная величина всего анализа -- ПОКАЗАТЕЛЬ СТЕПЕНИ b в
   t ~ n^b с его стандартной ошибкой и доверительным интервалом, а также
   тесты Вальда гипотез b = 1 (линейная сложность) и b = 2 (квадратичная).

2. ХИ-КВАДРАТ (взвешенный по ИЗМЕРЕННЫМ погрешностям)

   ML выше оценивает масштаб шума ИЗ ОСТАТКОВ, поэтому по построению не может
   ответить на вопрос "а согласуется ли модель с данными в пределах реальной
   погрешности замера". Для этого нужна НЕЗАВИСИМАЯ оценка sigma_i, и она в
   CSV есть -- размах min..max по прогонам точки:

       chi^2(theta) = sum ((t_i - f_i) / sigma_i)^2,   dof = N - k

   Согласие модели проверяется p-значением chi^2 (реализовано через
   регуляризованную неполную гамма-функцию, без scipy). Две независимые оценки
   sigma_i (обе печатаются, обе используются):

     range -- классическая: sigma прогона = (max-min)/d2(R) (поправка Хартли на
         размах выборки объёма R), затем стандартная ошибка МЕДИАНЫ
         = 1.2533 * sigma / sqrt(R). Оценка КОНСЕРВАТИВНАЯ (завышенная):
         распределение времени右-скошено выбросами планировщика ОС, и max
         утаскивает размах вверх.
     local -- по разбросу самого ряда медиан: вторые разности
         t_{i-1} - 2 t_i + t_{i+1} гасят гладкую составляющую (любая плавная
         f даёт по ним почти ноль) и оставляют чистый шум с дисперсией
         6 * sigma^2; масштаб берётся робастно, через MAD. Это погрешность
         ИМЕННО той величины, которая нанесена на график.

ВАЖНО: ранжирование моделей внутри chi^2 не зависит от общего масштаба sigma
(умножение всех sigma на константу умножает все chi^2 на ту же константу), от
масштаба зависит только АБСОЛЮТНОЕ p-значение согласия. Поэтому расхождение
двух оценок sigma не влияет на вывод о том, КАКАЯ сложность у алгоритма.

=============================================================================
СЕМЕЙСТВО ПРОВЕРЯЕМЫХ ЗАВИСИМОСТЕЙ

    a*n                    чистая линейная, без накладных расходов
    a + b*n                линейная со сдвигом (постоянные накладные)
    a + b*n*log2(n)        n log n
    a + b*n^2              квадратичная со сдвигом
    a + b*n + c*n^2        полная квадратичная (линейный член + квадратичный)
    a*n^b                  степенная со свободным показателем
    a + b*n^c              степенная со свободным показателем и сдвигом
    a + b*W(n)             СТРУКТУРНАЯ: W -- точное число вызовов Шага 1
    a + b*n + c*W(n)       структурная с линейным членом

СТРУКТУРНЫЙ ПРЕДИКТОР W(n) -- не подгонка, а прямой счёт по коду моделей.
dynamic_v2 в НАЧАЛЕ КАЖДОГО РАУНДА пересчитывает приоритет ВСЕХ ещё не
доставленных грузов (scheduler.select_round -> cargo_priority.rank_cargos по
всему pending), а за раунд доставляется P = min(R_d, R_b) грузов. Значит число
вызовов route_cost_for_cargo за прогон равно

    W(n) = sum_{r=0}^{ceil(n/P)-1} max(n - P*r, 0)  ~  n^2/(2P) + n/2,

то есть КВАДРАТИЧНО по n. У static_v3 приоритетов нет вообще: Шаг 3 раздаёт
грузы один раз, и число вызовов ESTIMATE-TASK-COST равно ровно n -- ЛИНЕЙНО.
Это и есть теоретическая сложность, которую проверяют оба численных метода.

W отличается от гладкого n^2 «пилой»: число раундов ceil(n/P) -- ступенчатая
функция n, и время обязано наследовать эту ступеньку. Если W выигрывает у
полного квадратичного полинома по AIC, значит в остатках сидела именно она.

Нелинейные по параметрам подгоняются алгоритмом Левенберга--Марквардта
(реализован здесь же: внешних зависимостей, кроме numpy/matplotlib, нет).

Дополнительно печатаются:
  * LR-тест вложенных моделей (a+b*n) c (a+b*n+c*n^2) -- значим ли вообще
    квадратичный член;
  * показатель b на ХВОСТАХ развёртки (n >= 50, 100, 150) -- сложность
    асимптотична, и оценка по всему диапазону смещена малыми n;
  * скользящий локальный показатель b(n) = d ln t / d ln n -- видно, к чему
    он сходится с ростом n.

ЗАПУСК:
    python experiments/analyze_time_complexity.py \\
        experiments/outputs/static_v3_vs_dynamic_v2_inverse_v3-balanced_1to200_step1.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Callable, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Распределения (без scipy): регуляризованная неполная гамма -> p-значение
# chi^2; нормальное -- через erfc.
# --------------------------------------------------------------------------

_ITMAX, _EPS = 500, 3.0e-14


def _gser(a: float, x: float) -> float:
    """Ряд для регуляризованной НИЖНЕЙ неполной гамма-функции P(a, x)."""
    ap, summ, delt = a, 1.0 / a, 1.0 / a
    for _ in range(_ITMAX):
        ap += 1.0
        delt *= x / ap
        summ += delt
        if abs(delt) < abs(summ) * _EPS:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    """Непрерывная дробь для ВЕРХНЕЙ Q(a, x) = 1 - P(a, x)."""
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, _ITMAX):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delt = d * c
        h *= delt
        if abs(delt - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, dof: int) -> float:
    """P(chi^2_dof > x) -- вероятность получить согласие ХУЖЕ наблюдаемого,
    если модель верна. Малое значение = модель отвергается."""
    if x <= 0:
        return 1.0
    a, xx = dof / 2.0, x / 2.0
    if xx < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gser(a, xx)))
    return max(0.0, min(1.0, _gcf(a, xx)))


def norm_two_sided_p(z: float) -> float:
    """Двусторонний p для стандартной нормали (тест Вальда)."""
    return math.erfc(abs(z) / math.sqrt(2.0))


# --------------------------------------------------------------------------
# Левенберг--Марквардт: минимизация sum r(theta)^2 с численным якобианом
# --------------------------------------------------------------------------

def _jacobian(resid: Callable[[np.ndarray], np.ndarray], theta: np.ndarray,
              r0: np.ndarray) -> np.ndarray:
    J = np.empty((r0.size, theta.size))
    for j in range(theta.size):
        h = 1e-7 * max(abs(theta[j]), 1e-8)
        tp = theta.copy()
        tp[j] += h
        J[:, j] = (resid(tp) - r0) / h
    return J


def lm_fit(resid: Callable[[np.ndarray], np.ndarray], theta0: Sequence[float],
           maxiter: int = 300) -> Tuple[np.ndarray, float, np.ndarray]:
    """Возвращает (theta, S = sum r^2, J). Универсален: подставляя разные
    resid, получаем и ML (остатки в логарифмах), и chi^2 (остатки, делённые на
    sigma_i) -- ровно два метода из шапки файла."""
    theta = np.array(theta0, dtype=float)
    r = resid(theta)
    S = float(r @ r)
    lam = 1e-3
    for _ in range(maxiter):
        J = _jacobian(resid, theta, r)
        JTJ, JTr = J.T @ J, J.T @ r
        diag = np.maximum(np.diag(JTJ), 1e-12)
        improved = False
        for _ in range(40):
            try:
                step = np.linalg.solve(JTJ + lam * np.diag(diag), -JTr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            th_new = theta + step
            r_new = resid(th_new)
            S_new = float(r_new @ r_new)
            if np.isfinite(S_new) and S_new < S:
                improved = abs(S - S_new) > 1e-14 * max(S, 1e-30)
                theta, r, S = th_new, r_new, S_new
                lam = max(lam * 0.3, 1e-12)
                break
            lam *= 10.0
            if lam > 1e14:
                break
        if not improved:
            break
    return theta, S, _jacobian(resid, theta, r)


# --------------------------------------------------------------------------
# Семейство моделей сложности
# --------------------------------------------------------------------------

class Model:
    def __init__(self, key: str, label: str, func, init):
        self.key, self.label, self.func, self.init = key, label, func, init

    def k(self, n: np.ndarray, t: np.ndarray) -> int:
        return len(self.init(n, t))


def _lin_init(cols):
    """Стартовое приближение для моделей, линейных по параметрам: обычный МНК
    по указанным базисным функциям."""
    def init(n, t):
        X = np.column_stack([c(n) for c in cols])
        return np.linalg.lstsq(X, t, rcond=None)[0]
    return init


_ONE = lambda n: np.ones_like(n)
_N = lambda n: n
_NLOGN = lambda n: n * np.log2(np.maximum(n, 1.0))
_N2 = lambda n: n ** 2


def _pow_init(n, t):
    """a*n^b: старт из МНК в логарифмах (это же -- точное ML-решение для этой
    модели, дальше LM только уточняет в нужной метрике)."""
    A = np.column_stack([np.ones_like(n), np.log(n)])
    c = np.linalg.lstsq(A, np.log(t), rcond=None)[0]
    return np.array([math.exp(c[0]), c[1]])


def _pow_off_init(n, t):
    p = _pow_init(n, t)
    return np.array([0.0, p[0], p[1]])


def sched_work(n: np.ndarray, pairs: int) -> np.ndarray:
    """W(n) -- точное число вызовов Шага 1 (ранжирование ВСЕХ ещё не
    доставленных грузов) за прогон dynamic_v2 при P = pairs доставках за раунд.
    Прямой счёт по коду scheduler.run_dynamic_rounds, а не аппроксимация."""
    out = np.empty_like(n, dtype=float)
    for i, v in enumerate(n):
        m, total = int(round(v)), 0
        while m > 0:
            total += m
            m -= pairs
        out[i] = total
    return out


def make_models(pairs: int) -> List[Model]:
    """Список моделей; структурные добавляются последними -- им нужен P."""
    W = lambda n: sched_work(n, pairs)
    extra = [
        Model("a+bW", f"a + b*W(n)  (структурная, P={pairs})",
              lambda th, n, W=W: th[0] + th[1] * W(n), _lin_init([_ONE, W])),
        Model("a+bn+cW", f"a + b*n + c*W(n)  (структурная + линейный)",
              lambda th, n, W=W: th[0] + th[1] * n + th[2] * W(n),
              _lin_init([_ONE, _N, W])),
    ]
    return _BASE_MODELS + extra


_BASE_MODELS: List[Model] = [
    Model("n", "a*n  (чистая линейная)",
          lambda th, n: th[0] * n, _lin_init([_N])),
    Model("a+bn", "a + b*n  (линейная со сдвигом)",
          lambda th, n: th[0] + th[1] * n, _lin_init([_ONE, _N])),
    Model("a+bnlogn", "a + b*n*log2(n)  (n log n)",
          lambda th, n: th[0] + th[1] * _NLOGN(n), _lin_init([_ONE, _NLOGN])),
    Model("a+bn2", "a + b*n^2  (квадратичная)",
          lambda th, n: th[0] + th[1] * n ** 2, _lin_init([_ONE, _N2])),
    Model("a+bn+cn2", "a + b*n + c*n^2  (полная квадратичная)",
          lambda th, n: th[0] + th[1] * n + th[2] * n ** 2,
          _lin_init([_ONE, _N, _N2])),
    Model("a*n^b", "a*n^b  (степенная, свободный показатель)",
          lambda th, n: th[0] * n ** th[1], _pow_init),
    Model("a+b*n^c", "a + b*n^c  (степенная со сдвигом)",
          lambda th, n: th[0] + th[1] * n ** th[2], _pow_off_init),
]

# Заполняется в main() через make_models(): структурным моделям нужно число
# доставок за раунд P, которое известно только после чтения CSV.
MODELS: List[Model] = list(_BASE_MODELS)


def _safe(y: np.ndarray) -> np.ndarray:
    """Модель обязана быть положительной: в логарифмах отрицательные значения
    недопустимы, а LM может забрести туда на промежуточном шаге."""
    return np.maximum(y, 1e-12)


# --------------------------------------------------------------------------
# Оценки погрешности sigma_i для метода хи-квадрат
# --------------------------------------------------------------------------

# Поправка Хартли d2: E[размах выборки объёма R] = d2 * sigma
_D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704,
       8: 2.847, 9: 2.970, 10: 3.078}
_MEDIAN_EFF = 1.2533  # se(медианы) = 1.2533 * sigma / sqrt(R) для нормали


def sigma_from_range(t_min: np.ndarray, t_max: np.ndarray, repeats: int) -> np.ndarray:
    d2 = _D2.get(repeats, 2.704)
    sigma_run = (t_max - t_min) / d2
    se = _MEDIAN_EFF * sigma_run / math.sqrt(repeats)
    # точки, где все прогоны совпали до последнего знака, погрешности не имеют;
    # подставляем наименьшую ненулевую, иначе chi^2 обращается в бесконечность
    pos = se[se > 0]
    return np.maximum(se, pos.min() if pos.size else 1e-6)


def sigma_local(n: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, float]:
    """Робастная оценка шума ОТНОСИТЕЛЬНОГО уровня по вторым разностям ряда
    медиан. Возвращает (sigma_i, относительный уровень шума)."""
    d = t[:-2] - 2.0 * t[1:-1] + t[2:]
    rel = d / t[1:-1]
    mad = np.median(np.abs(rel - np.median(rel)))
    s_rel = 1.4826 * mad / math.sqrt(6.0)
    return s_rel * t, s_rel


# --------------------------------------------------------------------------
# Подгонка одной модели двумя методами
# --------------------------------------------------------------------------

class Fit:
    pass


def fit_ml(model: Model, n: np.ndarray, t: np.ndarray) -> Fit:
    """Максимальное правдоподобие при мультипликативном лог-нормальном шуме."""
    logt = np.log(t)
    resid = lambda th: logt - np.log(_safe(model.func(th, n)))
    theta, S, J = lm_fit(resid, model.init(n, t))
    N, k = t.size, theta.size
    s2 = S / N                                    # ML-оценка дисперсии шума
    lnL = -0.5 * N * (math.log(2.0 * math.pi * s2) + 1.0)
    kk = k + 1                                    # + сам параметр масштаба s
    f = Fit()
    f.model, f.theta, f.S, f.k = model, theta, S, k
    f.s_rel = math.sqrt(S / max(N - k, 1))        # несмещённее для отчёта
    f.lnL = lnL
    f.aic = 2 * kk - 2 * lnL
    f.bic = kk * math.log(N) - 2 * lnL
    try:
        cov = (S / max(N - k, 1)) * np.linalg.inv(J.T @ J)
        f.se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        f.se = np.full(k, float("nan"))
    r = t / model.func(theta, n) - 1.0
    f.max_rel = float(np.max(np.abs(r)))
    f.rms_rel = float(np.sqrt(np.mean(r ** 2)))
    return f


def fit_chi2(model: Model, n: np.ndarray, t: np.ndarray, sigma: np.ndarray) -> Fit:
    """Метод хи-квадрат: взвешивание по НЕЗАВИСИМО измеренным погрешностям."""
    resid = lambda th: (t - model.func(th, n)) / sigma
    theta, chi2, J = lm_fit(resid, model.init(n, t))
    N, k = t.size, theta.size
    dof = max(N - k, 1)
    f = Fit()
    f.model, f.theta, f.k = model, theta, k
    f.chi2, f.dof = chi2, dof
    f.chi2_red = chi2 / dof
    f.p = chi2_sf(chi2, dof)
    f.aic = chi2 + 2 * k
    f.bic = chi2 + k * math.log(N)
    try:
        cov = np.linalg.inv(J.T @ J)
        f.se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        f.se = np.full(k, float("nan"))
    r = t / model.func(theta, n) - 1.0
    f.max_rel = float(np.max(np.abs(r)))
    return f


# --------------------------------------------------------------------------
# Показатель степени: хвосты и скользящее окно
# --------------------------------------------------------------------------

def power_exponent(n: np.ndarray, t: np.ndarray) -> Tuple[float, float, int]:
    """ML-оценка b в t = a*n^b (лог-нормальный шум) + её стандартная ошибка."""
    X = np.column_stack([np.ones_like(n), np.log(n)])
    y = np.log(t)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    N = y.size
    s2 = float(r @ r) / (N - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[1]), float(math.sqrt(cov[1, 1])), N


def rolling_exponent(n: np.ndarray, t: np.ndarray, win: int):
    xs, bs, ses = [], [], []
    for i in range(0, n.size - win + 1):
        sl = slice(i, i + win)
        b, se, _ = power_exponent(n[sl], t[sl])
        xs.append(float(np.exp(np.mean(np.log(n[sl])))))
        bs.append(b)
        ses.append(se)
    return np.array(xs), np.array(bs), np.array(ses)


def rolling_marginal(n: np.ndarray, t: np.ndarray, win: int):
    """Предельная стоимость одного груза dt/dn в скользящем окне (МНК прямой).
    Прямой тест сложности БЕЗ выбора модели: у Theta(n) она выходит на
    константу, у Theta(n^2) сама растёт линейно по n."""
    xs, ms, ses = [], [], []
    for i in range(0, n.size - win + 1):
        sl = slice(i, i + win)
        x, y = n[sl], t[sl]
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        s2 = float(r @ r) / (x.size - 2)
        cov = s2 * np.linalg.inv(X.T @ X)
        xs.append(float(np.mean(x)))
        ms.append(float(beta[1]))
        ses.append(float(math.sqrt(cov[1, 1])))
    return np.array(xs), np.array(ms), np.array(ses)


# --------------------------------------------------------------------------
# Отчёт по одному алгоритму
# --------------------------------------------------------------------------

def analyse(name: str, n: np.ndarray, t: np.ndarray, t_min: np.ndarray,
            t_max: np.ndarray, repeats: int, tail_cuts: Sequence[int],
            rows_out: List[dict]):
    N = t.size
    print("\n" + "=" * 78)
    print(f"{name}:  t(n), мс -- медиана по {repeats} прогонам, N = {N} точек, "
          f"n = {int(n[0])}..{int(n[-1])}")
    print("=" * 78)

    sig_range = sigma_from_range(t_min, t_max, repeats)
    sig_loc, s_rel_loc = sigma_local(n, t)
    print(f"\nПОГРЕШНОСТЬ ТОЧКИ (две независимые оценки):")
    print(f"  range: se(медианы) из размаха min..max, d2({repeats})="
          f"{_D2.get(repeats, 2.704)};  в среднем {np.mean(sig_range / t) * 100:5.2f}% от t")
    print(f"  local: по вторым разностям ряда медиан;      "
          f"постоянный уровень {s_rel_loc * 100:5.2f}% от t")
    print(f"  отношение range/local = {np.mean(sig_range / sig_loc):.2f}"
          f"  (>1 -- размах завышен выбросами планировщика ОС)")

    # ---------------- метод 1: максимальное правдоподобие ----------------
    ml = [fit_ml(m, n, t) for m in MODELS]
    best_aic = min(f.aic for f in ml)
    order = np.argsort([f.aic for f in ml])
    print("\n--- МЕТОД МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ "
          "(мультипликативный лог-нормальный шум) ---")
    print(f"{'модель':38s} {'k':>2s} {'шум s':>7s} {'lnL':>9s} "
          f"{'AIC':>9s} {'dAIC':>8s} {'BIC':>9s} {'макс.откл':>9s}")
    for i in order:
        f = ml[i]
        print(f"{f.model.label:38s} {f.k:2d} {f.s_rel * 100:6.2f}% "
              f"{f.lnL:9.1f} {f.aic:9.1f} {f.aic - best_aic:8.1f} "
              f"{f.bic:9.1f} {f.max_rel * 100:8.1f}%")
    ml_best = ml[int(order[0])]
    print(f"  ЛУЧШАЯ ПО AIC: {ml_best.model.label}")
    print(f"  параметры: " + ", ".join(
        f"{v:.6g} +- {e:.2g}" for v, e in zip(ml_best.theta, ml_best.se)))

    # ---------------- метод 2: хи-квадрат ----------------
    for tag, sig in (("range (размах min..max)", sig_range),
                     ("local (вторые разности)", sig_loc)):
        ch = [fit_chi2(m, n, t, sig) for m in MODELS]
        order2 = np.argsort([f.chi2 for f in ch])
        print(f"\n--- МЕТОД ХИ-КВАДРАТ, sigma = {tag} ---")
        print(f"{'модель':38s} {'k':>2s} {'chi^2':>11s} {'dof':>4s} "
              f"{'chi^2/dof':>9s} {'p':>10s} {'макс.откл':>9s}")
        for i in order2:
            f = ch[i]
            print(f"{f.model.label:38s} {f.k:2d} {f.chi2:11.1f} {f.dof:4d} "
                  f"{f.chi2_red:9.2f} {f.p:10.3g} {f.max_rel * 100:8.1f}%")
        best = ch[int(order2[0])]
        print(f"  ЛУЧШАЯ ПО chi^2: {best.model.label}  "
              f"(chi^2/dof = {best.chi2_red:.2f}, p = {best.p:.3g})")
        print(f"  параметры: " + ", ".join(
            f"{v:.6g} +- {e:.2g}" for v, e in zip(best.theta, best.se)))
        if tag.startswith("local"):
            chi_local_best = best

    # ---------------- показатель степени ----------------
    print("\n--- ПОКАЗАТЕЛЬ СТЕПЕНИ b в t ~ n^b (ML в логарифмах) ---")
    print(f"{'диапазон':>16s} {'точек':>6s} {'b':>8s} {'se(b)':>7s} "
          f"{'95% ДИ':>18s} {'z(b=1)':>8s} {'p(b=1)':>9s} {'z(b=2)':>8s} {'p(b=2)':>9s}")
    for cut in [int(n[0])] + list(tail_cuts):
        m = n >= cut
        if m.sum() < 8:
            continue
        b, se, cnt = power_exponent(n[m], t[m])
        z1, z2 = (b - 1.0) / se, (b - 2.0) / se
        rng = f"n>={cut}"
        print(f"{rng:>16s} {cnt:6d} {b:8.3f} {se:7.3f} "
              f"[{b - 1.96 * se:6.3f},{b + 1.96 * se:6.3f}] "
              f"{z1:8.1f} {norm_two_sided_p(z1):9.2g} "
              f"{z2:8.1f} {norm_two_sided_p(z2):9.2g}")
        rows_out.append(dict(model=name, kind="exponent", range=rng, n_points=cnt,
                             b=b, se_b=se, p_b_eq_1=norm_two_sided_p(z1),
                             p_b_eq_2=norm_two_sided_p(z2)))

    # ---------------- предельная стоимость одного груза ----------------
    xm, mg, mse = rolling_marginal(n, t, max(21, N // 8))
    print("\n--- ПРЕДЕЛЬНАЯ СТОИМОСТЬ ОДНОГО ГРУЗА dt/dn, мс "
          "(скользящий МНК, без выбора модели) ---")
    picks = [np.argmin(np.abs(xm - v)) for v in (25, 50, 100, 150, 175)
             if xm.min() <= v <= xm.max()]
    print("   " + "  ".join(f"n~{int(xm[i]):3d}: {mg[i]:.4f}+-{mse[i]:.4f}"
                            for i in picks))
    # растёт ли сама предельная стоимость (признак сверхлинейности)
    Xm = np.column_stack([np.ones_like(xm), xm])
    bm, *_ = np.linalg.lstsq(Xm, mg, rcond=None)
    rm = mg - Xm @ bm
    s2m = float(rm @ rm) / (xm.size - 2)
    se_slope = math.sqrt(s2m * np.linalg.inv(Xm.T @ Xm)[1, 1])
    print(f"   наклон самой dt/dn по n: {bm[1]:+.5f} +- {se_slope:.5f} мс/груз^2 "
          f"(z = {bm[1] / se_slope:+.1f})")
    print(f"   {'РАСТЁТ -> сверхлинейная сложность' if bm[1] / se_slope > 3 else ''}"
          f"{'ПАДАЕТ/постоянна -> линейная сложность (сверхлинейности нет)' if bm[1] / se_slope < 3 else ''}")

    # ---------------- LR-тест вложенных моделей ----------------
    lin = next(f for f in ml if f.model.key == "a+bn")
    quad = next(f for f in ml if f.model.key == "a+bn+cn2")
    lr = N * math.log(lin.S / quad.S) if quad.S > 0 else float("inf")
    p_lr = chi2_sf(lr, 1)
    print("\n--- LR-ТЕСТ: нужен ли квадратичный член? "
          "(a+b*n)  vs  (a+b*n+c*n^2) ---")
    print(f"  LR = N*ln(S_lin/S_quad) = {lr:.1f},  dof = 1,  p = {p_lr:.3g}"
          f"   -> {'квадратичный член ЗНАЧИМ' if p_lr < 0.01 else 'квадратичный член НЕ значим'}")
    c_hat = quad.theta[2]
    c_se = quad.se[2]
    print(f"  коэффициент при n^2: c = {c_hat:.4g} +- {c_se:.2g} мс "
          f"(z = {c_hat / c_se if c_se else float('nan'):.1f})")

    rows_out.append(dict(model=name, kind="best_ml", range=f"n>={int(n[0])}",
                         n_points=N, label=ml_best.model.label,
                         params=";".join(f"{v:.6g}" for v in ml_best.theta),
                         aic=ml_best.aic, bic=ml_best.bic, s_rel=ml_best.s_rel))
    rows_out.append(dict(model=name, kind="best_chi2_local", range=f"n>={int(n[0])}",
                         n_points=N, label=chi_local_best.model.label,
                         params=";".join(f"{v:.6g}" for v in chi_local_best.theta),
                         chi2=chi_local_best.chi2, dof=chi_local_best.dof,
                         chi2_red=chi_local_best.chi2_red, p=chi_local_best.p))
    return ml, ml_best, sig_range, sig_loc


# --------------------------------------------------------------------------
# График
# --------------------------------------------------------------------------

def plot(path, n, series, win):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    colors = {"static_v3": "#1f5fa8", "dynamic_v2": "#c0392b"}

    ax = axes[0, 0]
    for name, d in series.items():
        c = colors[name]
        ax.plot(n, d["t"], ".", color=c, markersize=3, label=f"{name}: замер")
        ax.plot(n, d["best"].model.func(d["best"].theta, n), "-", color=c,
                linewidth=1.6, label=f"{name}: {d['best'].model.label}")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("Число грузов n"), ax.set_ylabel("Время расчёта, мс")
    ax.set_title("Лог-лог: замеры и лучшая по AIC модель\n"
                 "(наклон прямой = показатель степени b)", fontsize=10,
                 fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for name, d in series.items():
        x, b, se = rolling_exponent(n, d["t"], win)
        ax.plot(x, b, "-", color=colors[name], linewidth=1.6, label=name)
        ax.fill_between(x, b - 1.96 * se, b + 1.96 * se, color=colors[name],
                        alpha=0.15, linewidth=0)
    for lvl, txt in ((1.0, "Θ(n)"), (2.0, "Θ(n²)")):
        ax.axhline(lvl, color="#555", linestyle=":", linewidth=1)
        ax.annotate(txt, (n[-1], lvl), fontsize=8, color="#555",
                    va="bottom", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Число грузов n (центр окна)")
    ax.set_ylabel("Локальный показатель b = d ln t / d ln n")
    ax.set_title(f"Локальный показатель степени, окно {win} точек\n"
                 "полоса -- 95% ДИ", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)

    for ax, name in ((axes[1, 0], "static_v3"), (axes[1, 1], "dynamic_v2")):
        d = series[name]
        rel = 100.0 * (d["t"] / d["best"].model.func(d["best"].theta, n) - 1.0)
        band = 100.0 * d["sig_loc"] / d["t"]
        ax.axhline(0, color="#333", linewidth=1)
        ax.fill_between(n, -1.96 * band, 1.96 * band, color="#888", alpha=0.2,
                        linewidth=0, label="±1.96σ (local)")
        ax.plot(n, rel, ".", color=colors[name], markersize=3.5)
        ax.set_xlabel("Число грузов n")
        ax.set_ylabel("Относительный остаток, %")
        ax.set_title(f"{name}: остатки лучшей модели\n{d['best'].model.label}",
                     fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    for ax, name in ((axes[2, 0], "static_v3"), (axes[2, 1], "dynamic_v2")):
        d = series[name]
        x, m, se = rolling_marginal(n, d["t"], max(21, n.size // 8))
        ax.plot(x, m, "-", color=colors[name], linewidth=1.8)
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=colors[name],
                        alpha=0.18, linewidth=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Число грузов n (центр окна)")
        ax.set_ylabel("dt/dn, мс на один груз")
        ax.set_title(f"{name}: предельная стоимость одного груза\n"
                     "константа = Θ(n), прямая с наклоном = Θ(n²)",
                     fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Оценка вычислительной сложности по замерам времени: "
                 "максимальное правдоподобие и хи-квадрат", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nСохранено: {path}")


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Оценка вычислительной сложности static_v3 и dynamic_v2 по "
                    "замерам времени: методы максимального правдоподобия и "
                    "хи-квадрат.")
    p.add_argument("csv", help="CSV из compare_static_v3_vs_dynamic_v2.py")
    p.add_argument("--repeats", type=int, default=7,
                   help="сколько прогонов на точку было в свипе (--repeats "
                        "исходного скрипта); нужно для пересчёта размаха "
                        "min..max в стандартную ошибку медианы")
    p.add_argument("--min-n", type=int, default=1,
                   help="отбросить точки с n меньше указанного")
    p.add_argument("--tails", type=int, nargs="*", default=[25, 50, 100, 150],
                   help="границы хвостов для асимптотической оценки показателя")
    p.add_argument("--window", type=int, default=25,
                   help="окно скользящей оценки локального показателя")
    p.add_argument("--n-pairs", type=int, default=None,
                   help="P -- сколько грузов доставляется за раунд (число пар "
                        "/ коалиций). По умолчанию восстанавливается из "
                        "колонки *_rounds: P = n_max / rounds_max")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args(argv)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("n_cargos")]

    n = np.array([float(r["n_cargos"]) for r in rows])
    keep = n >= args.min_n
    n = n[keep]

    def col(name):
        return np.array([float(r[name]) for r in rows])[keep]

    data = {
        "static_v3": (col("static_v3_time_ms_median"),
                      col("static_v3_time_ms_min"),
                      col("static_v3_time_ms_max")),
        "dynamic_v2": (col("dynamic_v2_time_ms_median"),
                       col("dynamic_v2_time_ms_min"),
                       col("dynamic_v2_time_ms_max")),
    }

    rounds = np.array([float(r["dynamic_v2_rounds"]) for r in rows])[keep]
    pairs = args.n_pairs or max(1, int(round(n[-1] / max(rounds[-1], 1.0))))
    global MODELS
    MODELS = make_models(pairs)

    print(f"Файл: {args.csv}")
    print(f"Точек: {n.size}, n = {int(n[0])}..{int(n[-1])}, "
          f"прогонов на точку: {args.repeats}, "
          f"доставок за раунд P = {pairs} "
          f"({'задано' if args.n_pairs else 'из колонки rounds'})")
    print("Анализируется ТОЛЬКО время расчёта t(n); "
          "сравниваются static_v3 и dynamic_v2.")

    out_rows: List[dict] = []
    series = {}
    for name, (t, tlo, thi) in data.items():
        _, best, sig_range, sig_loc = analyse(
            name, n, t, tlo, thi, args.repeats, args.tails, out_rows)
        series[name] = dict(t=t, best=best, sig_range=sig_range, sig_loc=sig_loc)

    # отношение времён: во сколько раз dynamic_v2 дороже и как это растёт
    ratio = data["dynamic_v2"][0] / data["static_v3"][0]
    b_r, se_r, _ = power_exponent(n, ratio)
    print("\n" + "=" * 78)
    print("ОТНОШЕНИЕ ВРЕМЁН dynamic_v2 / static_v3")
    print("=" * 78)
    print(f"  n={int(n[0])}: {ratio[0]:.2f}x   n={int(n[n.size // 2])}: "
          f"{ratio[n.size // 2]:.2f}x   n={int(n[-1])}: {ratio[-1]:.2f}x")
    print(f"  само отношение растёт как n^{b_r:.3f} +- {se_r:.3f}  "
          f"-- разность показателей сложности двух моделей")

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.csv))[0] + "_complexity"

    csv_path = os.path.join(out_dir, stem + ".csv")
    fields = ["model", "kind", "range", "n_points", "b", "se_b", "p_b_eq_1",
              "p_b_eq_2", "label", "params", "aic", "bic", "s_rel", "chi2",
              "dof", "chi2_red", "p"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nСохранено: {csv_path}")

    plot(os.path.join(out_dir, stem + ".png"), n, series, args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
