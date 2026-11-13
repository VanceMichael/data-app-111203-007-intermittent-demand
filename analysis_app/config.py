"""Cost and backtest configuration.

The replenishment backtest never hard-codes money.  Shortage and holding
costs are supplied by the caller (a JSON file through the CLI) and are echoed
back in the result together with a fingerprint so a report can be reproduced.
"""

import hashlib
import json
import math
from pathlib import Path

MODEL_NAMES = ('croston_sba', 'exposure_hurdle', 'exposure_poisson')

DEFAULTS = {
    'min_history_days': 3,
    'recent_window_days': 7,
    'availability_window_days': 7,
    'croston_alpha': 0.1,
    'min_selection_windows': 3,
    'default_model': 'exposure_hurdle',
    'exclude_promotion_from_training': True,
    'exclude_promotion_from_evaluation': True,
    'max_stockout_rate': 0.25,
    'stockout_rate_tolerance': 0.05,
    'min_order_quantile': 0.5,
    'max_order_quantile': 0.99,
    'candidate_quantiles': tuple(round(0.50 + 0.05 * step, 2) for step in range(10)),
    'demand_sensitivity_factors': (0.0, 0.5, 1.0),
    'report_quantiles': (0.5, 0.8, 0.9, 0.95),
}


def _finite_number(name, value, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'配置字段不是数值：{name}')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'配置字段不是有限数：{name}')
    if minimum is not None and value < minimum:
        raise ValueError(f'配置字段小于下限：{name}')
    if maximum is not None and value > maximum:
        raise ValueError(f'配置字段大于上限：{name}')
    return value


def _cost_setting(name, raw, minimum):
    if isinstance(raw, dict):
        if not raw:
            raise ValueError(f'成本映射为空：{name}')
        result = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f'成本映射键无效：{name}')
            result[key] = _finite_number(f'{name}.{key}', value, minimum)
        return result
    return _finite_number(name, raw, minimum)


def _positive_count(name, raw, minimum):
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f'配置字段不是整数：{name}')
    if raw < minimum:
        raise ValueError(f'配置字段小于下限：{name}')
    return raw


def _bool(name, raw):
    if not isinstance(raw, bool):
        raise ValueError(f'配置字段不是布尔值：{name}')
    return raw


def normalize_config(raw):
    """Validate and canonicalise a configuration mapping."""
    if not isinstance(raw, dict):
        raise ValueError('配置必须是 JSON 对象')
    costs = raw.get('costs')
    settings = raw.get('settings')
    if costs is None:
        # Flat form is accepted for convenience: cost keys at the top level.
        costs = {key: raw.get(key) for key in
                 ('shortage_cost_per_unit', 'holding_cost_per_unit_per_day', 'order_cost_per_order')}
        settings = raw if settings is None else settings
    if not isinstance(costs, dict):
        raise ValueError('costs 必须是 JSON 对象')
    for key in ('shortage_cost_per_unit', 'holding_cost_per_unit_per_day'):
        if key not in costs:
            raise ValueError(f'缺少成本配置：{key}')
    shortage = _cost_setting('shortage_cost_per_unit', costs['shortage_cost_per_unit'], 0.0)
    holding = _cost_setting('holding_cost_per_unit_per_day', costs['holding_cost_per_unit_per_day'], 0.0)
    order = _finite_number('order_cost_per_order', costs.get('order_cost_per_order', 0.0), 0.0)
    if isinstance(shortage, dict):
        if not any(value > 0 for value in shortage.values()):
            raise ValueError('shortage_cost_per_unit 必须包含正成本')
    elif shortage <= 0:
        raise ValueError('shortage_cost_per_unit 必须为正数')

    settings = {} if settings is None else settings
    if not isinstance(settings, dict):
        raise ValueError('settings 必须是 JSON 对象')
    params = {}
    params['min_history_days'] = _positive_count('min_history_days', settings.get('min_history_days', DEFAULTS['min_history_days']), 2)
    params['recent_window_days'] = _positive_count('recent_window_days', settings.get('recent_window_days', DEFAULTS['recent_window_days']), 1)
    params['availability_window_days'] = _positive_count('availability_window_days', settings.get('availability_window_days', DEFAULTS['availability_window_days']), 1)
    params['croston_alpha'] = _finite_number('croston_alpha', settings.get('croston_alpha', DEFAULTS['croston_alpha']), 0.0, 1.0)
    if params['croston_alpha'] <= 0:
        raise ValueError('croston_alpha 必须大于 0')
    params['min_selection_windows'] = _positive_count('min_selection_windows', settings.get('min_selection_windows', DEFAULTS['min_selection_windows']), 1)
    default_model = settings.get('default_model', DEFAULTS['default_model'])
    if default_model not in MODEL_NAMES:
        raise ValueError('default_model 不在候选模型内')
    params['default_model'] = default_model
    params['exclude_promotion_from_training'] = _bool('exclude_promotion_from_training', settings.get('exclude_promotion_from_training', DEFAULTS['exclude_promotion_from_training']))
    params['exclude_promotion_from_evaluation'] = _bool('exclude_promotion_from_evaluation', settings.get('exclude_promotion_from_evaluation', DEFAULTS['exclude_promotion_from_evaluation']))
    params['max_stockout_rate'] = _finite_number('max_stockout_rate', settings.get('max_stockout_rate', DEFAULTS['max_stockout_rate']), 0.0, 1.0)
    params['stockout_rate_tolerance'] = _finite_number('stockout_rate_tolerance', settings.get('stockout_rate_tolerance', DEFAULTS['stockout_rate_tolerance']), 0.0, 1.0)
    params['min_order_quantile'] = _finite_number('min_order_quantile', settings.get('min_order_quantile', DEFAULTS['min_order_quantile']), 0.0, 1.0)
    params['max_order_quantile'] = _finite_number('max_order_quantile', settings.get('max_order_quantile', DEFAULTS['max_order_quantile']), 0.0, 1.0)
    if params['min_order_quantile'] > params['max_order_quantile']:
        raise ValueError('min_order_quantile 不能大于 max_order_quantile')
    candidate_quantiles = settings.get('candidate_quantiles', list(DEFAULTS['candidate_quantiles']))
    if not isinstance(candidate_quantiles, (list, tuple)) or not candidate_quantiles:
        raise ValueError('candidate_quantiles 必须是非空数组')
    clean_candidates = sorted({_finite_number('candidate_quantiles', value, 0.0, 1.0)
                               for value in candidate_quantiles})
    if not clean_candidates:
        raise ValueError('candidate_quantiles 必须是非空数组')
    if not any(params['min_order_quantile'] <= q <= params['max_order_quantile']
               for q in clean_candidates):
        raise ValueError('candidate_quantiles 在允许分位区间内没有任何候选')
    params['candidate_quantiles'] = clean_candidates
    factors = settings.get('demand_sensitivity_factors', list(DEFAULTS['demand_sensitivity_factors']))
    if not isinstance(factors, (list, tuple)) or not factors:
        raise ValueError('demand_sensitivity_factors 必须是非空数组')
    clean_factors = sorted({_finite_number('demand_sensitivity_factors', value, 0.0, 1.0) for value in factors})
    if 1.0 not in clean_factors:
        raise ValueError('demand_sensitivity_factors 必须包含 1.0')
    params['demand_sensitivity_factors'] = clean_factors
    report_quantiles = settings.get('report_quantiles', list(DEFAULTS['report_quantiles']))
    if not isinstance(report_quantiles, (list, tuple)) or not report_quantiles:
        raise ValueError('report_quantiles 必须是非空数组')
    clean_quantiles = []
    for value in report_quantiles:
        quantile = _finite_number('report_quantiles', value, 0.0, 1.0)
        if quantile == 0.0 or quantile == 1.0:
            raise ValueError('report_quantiles 必须在 0 与 1 之间')
        clean_quantiles.append(quantile)
    params['report_quantiles'] = sorted(set(clean_quantiles))
    currency = raw.get('currency', 'CNY')
    if not isinstance(currency, str) or not currency:
        raise ValueError('currency 必须是非空字符串')
    return {
        'currency': currency,
        'costs': {
            'shortage_cost_per_unit': shortage,
            'holding_cost_per_unit_per_day': holding,
            'order_cost_per_order': order,
        },
        'settings': params,
    }


def load_config(path):
    with Path(path).open(encoding='utf-8') as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(f'配置不是合法 JSON：{error}') from None
    return normalize_config(raw)


def config_fingerprint(config):
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_cost(setting, sku, family):
    """Resolve a scalar, `*`, family-keyed or sku-keyed cost setting."""
    if isinstance(setting, dict):
        for key in (sku, family, '*'):
            if key in setting:
                return setting[key]
        raise ValueError(f'缺少 {sku}/{family} 的成本配置')
    return setting
