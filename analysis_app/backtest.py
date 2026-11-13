"""Rolling-origin replenishment backtest for intermittent demand.

Design rules enforced here
--------------------------

1. **Point-in-time isolation.**  A forecast with origin ``t`` is built only
   from rows of that SKU whose ``day <= t``.  Future rows are read only after
   every forecast for the origin has been frozen, and only to score it.
2. **Lead-time evaluation.**  Every origin is scored over that SKU's own
   ``lead_days`` window ``(t, t + lead_days]``, never a uniform next day.
   Origins whose window extends past the available calendar are not invented:
   only origins with a fully observed outcome are evaluated.
3. **Exposure classification.**  The evaluation window is classed as ``exact``
   (every day fully open), ``censored`` (some partial days), or
   ``unidentifiable`` (closed days, or rows absent).  Only exact, resolved
   windows with real exposure evidence feed cost-based selection; censored
   windows yield lower/upper sensitivity; unidentifiable windows are reported
   but never scored as demand.
4. **Family cost selection.**  Model and order quantile are chosen per product
   family from *earlier, already resolved* windows by shortage + holding cost,
   with costs resolved per SKU from configuration.  The fixed per-order cost is
   echoed but excluded from the criterion: it is constant across quantities for
   a single lead-time window, so it cannot rank candidates.  Costs are aggregated as an equal-weighted mean over
   SKU-origins, and a stockout-rate guard keeps expensive SKUs from hiding the
   stockout rate of cheap ones.  Money totals are displayed, not used as
   weights.
5. **No invented demand.**  Demand lost to stockouts is bounded and shown as
   sensitivity, never "recovered".
"""

import math
from collections import defaultdict

from .config import config_fingerprint, resolve_cost
from .demand_models import (
    MODEL_NAMES, exposure_profile, fit_all, model_params_for_output, pmf_mean,
    pmf_quantile, pmf_tail_probability, predict_parameters, predict_window_pmf,
    training_rows,
)

EXACT = 'exact'
CENSORED = 'censored'
UNIDENTIFIABLE = 'unidentifiable'
MISSING = 'missing'
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Static SKU identity and origin discovery
# ---------------------------------------------------------------------------

def _sku_identity(rows):
    first = rows[0]
    for row in rows:
        for field in ('family', 'lead_days', 'price'):
            if row[field] != first[field]:
                raise ValueError(f'{first["sku"]} 的 {field} 在同一 SKU 内不一致')
    if int(first['lead_days']) < 1:
        raise ValueError(f'{first["sku"]} 的 lead_days 必须为正整数')
    return {'sku': first['sku'], 'family': first['family'],
            'lead_days': int(first['lead_days']), 'price': first['price']}


def _candidate_origins(rows, settings):
    """Origins with enough history and a fully observed lead-time outcome.

    The origin list depends only on observed calendar days and the lead time,
    never on future *values*, so the set of origins is itself point-in-time
    safe.  Restricting to fully observed windows avoids labelling ordinary
    end-of-data truncation as "unidentifiable demand".
    """
    days = sorted({row['day'] for row in rows})
    first, last = days[0], days[-1]
    lead = int(rows[0]['lead_days'])
    return [day for day in days
            if day - first + 1 >= settings['min_history_days'] and day + lead <= last]


# ---------------------------------------------------------------------------
# Point-in-time forecast
# ---------------------------------------------------------------------------

def _known_rows(rows, origin):
    return [row for row in rows if row['day'] <= origin]


def _recent(rows, origin, window):
    return [row for row in rows if row['day'] > origin - window]


def build_forecast(identity, rows, origin, config):
    """Build a forecast that only uses rows with ``day <= origin``."""
    settings = config['settings']
    known = _known_rows(rows, origin)
    observed_days = len({row['day'] for row in known})
    if observed_days < settings['min_history_days']:
        raise ValueError(f'{identity["sku"]} 在起点 {origin} 的历史天数不足')

    availability_rows = _recent(known, origin, settings['availability_window_days'])
    exposure = exposure_profile(availability_rows)

    recent_rows = _recent(known, origin, settings['recent_window_days'])
    train = training_rows(recent_rows, settings)
    training_exposure = exposure_profile(train) if train else None
    # Without a single exposed training day the models have seen no sales
    # opportunity at all: a run of zeros is then *not* evidence of zero demand.
    exposure_evidence = bool(train) and training_exposure['exposed_days'] > 0

    fits = fit_all(train, settings)
    training_events = sum(1 for row in train if row['sold'] > 0)
    if not exposure_evidence:
        demand_status = 'unidentifiable_no_exposure'
    elif training_events == 0:
        demand_status = 'zero_events_exposed'
    else:
        demand_status = 'identified_from_events'
    quantiles = [q for q in settings['candidate_quantiles']
                 if settings['min_order_quantile'] <= q <= settings['max_order_quantile']]
    reported_quantiles = sorted(set(settings['report_quantiles']) | set(quantiles))
    models = {}
    for name in MODEL_NAMES:
        lead = identity['lead_days']
        window = predict_window_pmf(name, fits[name], exposure, lead)
        quantities = {}
        for quantile in quantiles:
            quantity, truncated = pmf_quantile(window, quantile)
            quantities[quantile] = {'quantity': quantity, 'support_truncated': truncated}
        models[name] = {
            'mean_units': pmf_mean(window),
            'parameters': predict_parameters(name, fits[name], exposure, lead),
            'fitted': model_params_for_output(fits[name]),
            'quantities': quantities,
            'support_units': len(window) - 1,
            'stockout_probability_by_quantile': {
                str(quantile): pmf_tail_probability(window, quantities[quantile]['quantity'])
                for quantile in reported_quantiles},
        }
    return {
        'sku': identity['sku'], 'family': identity['family'],
        'origin': origin, 'lead_days': identity['lead_days'],
        'known_days': observed_days, 'known_rows': len(known),
        'training_rows': len(train),
        'training_events': training_events,
        'promotion_rows_excluded_from_training': len(recent_rows) - len(train),
        'training_all_promotion': bool(recent_rows) and not train,
        'exposure_evidence': exposure_evidence,
        'demand_status': demand_status,
        'is_new_sku': observed_days < settings['recent_window_days'],
        'zero_event_upper_bound': next(
            (model['fitted'].get('occurrence_upper_bound_per_exposure_unit')
             for model in [models['exposure_hurdle']]
             if model['fitted'].get('occurrence_is_upper_bound')), None),
        'exposure': exposure,
        'training_exposure': training_exposure,
        'quantiles_considered': quantiles,
        'models': models,
    }


# ---------------------------------------------------------------------------
# Scoring (the only place future rows are read)
# ---------------------------------------------------------------------------

def _window_rows(rows, origin, lead_days):
    return sorted((row for row in rows if origin < row['day'] <= origin + lead_days),
                  key=lambda row: row['day'])


def _classify(window, expected_days):
    present = {row['day'] for row in window}
    missing = [day for day in expected_days if day not in present]
    hours = [row['available_hours'] for row in window]
    if missing:
        return UNIDENTIFIABLE, 'missing_rows_in_window', missing
    if any(value <= 0 for value in hours):
        return UNIDENTIFIABLE, 'closed_days_in_window', missing
    if any(value < 24 for value in hours):
        return CENSORED, 'partial_days_in_window', missing
    return EXACT, 'fully_exposed_window', missing


def score_forecast(identity, rows, origin, forecast, config):
    settings = config['settings']
    lead = identity['lead_days']
    window = _window_rows(rows, origin, lead)
    expected_days = list(range(origin + 1, origin + lead + 1))
    classification, reason, missing = _classify(window, expected_days)
    promo_windows = any(row['promotion'] == 1 for row in window)

    observed = math.fsum(row['sold'] for row in window)
    observed_hours = math.fsum(row['available_hours'] for row in window)

    bounds = None
    if classification != EXACT:
        lost_hours = max(0.0, lead * 24.0 - observed_hours)
        if forecast['exposure_evidence']:
            upper_by_model = {}
            for name, model in forecast['models'].items():
                mean_day = model['parameters']['mean_units_per_day']
                mean_hours = model['parameters'].get(
                    'mean_hours_per_day', forecast['exposure']['mean_hours_per_day'])
                rate = (mean_day / mean_hours) if mean_hours > 0 else 0.0
                upper_by_model[name] = observed + rate * lost_hours
        else:
            upper_by_model = {name: None for name in forecast['models']}
        bounds = {
            'demand_lower_bound': observed,
            'demand_estimate_ceiling_by_model': upper_by_model,
            'ceiling_available': forecast['exposure_evidence'],
            'upper_bound_is_model_dependent': True,
            'lost_exposure_hours': lost_hours,
            'assumption': ('demand_lower_bound 是已观测销量；'
                           'demand_estimate_ceiling_by_model 是按模型外推的敏感上界，'
                           '并非缺货损失的严格上限；两者都不是恢复的真实丢失订单，'
                           '部分可售日的售出量也没有被当作完整可售日的真实需求'),
        }

    if settings['exclude_promotion_from_evaluation'] and promo_windows:
        exclusion = 'promotion_window'
    elif classification != EXACT:
        exclusion = 'not_identifiable_as_exact'
    elif not forecast['exposure_evidence']:
        exclusion = 'no_exposure_evidence_in_history'
    else:
        exclusion = None

    return {
        'classification': classification,
        'classification_reason': reason,
        'window_days': [row['day'] for row in window],
        'missing_days': missing,
        'observed_units': observed,
        'observed_available_hours': observed_hours,
        'observed_units_are_lower_bound': classification != EXACT,
        'promotion_in_window': promo_windows,
        'used_for_selection': exclusion is None,
        'selection_exclusion_reason': exclusion,
        'demand_identifiable': classification == EXACT,
        'bounds': bounds,
    }


# ---------------------------------------------------------------------------
# Costs
# ---------------------------------------------------------------------------

def _cost_context(config, identity):
    return {
        'shortage_cost_per_unit': resolve_cost(
            config['costs']['shortage_cost_per_unit'], identity['sku'], identity['family']),
        'holding_cost_per_unit_per_day': resolve_cost(
            config['costs']['holding_cost_per_unit_per_day'], identity['sku'], identity['family']),
        'order_cost_per_order': config['costs']['order_cost_per_order'],
    }


def _realised_cost(quantity, demand, costs, lead_days):
    """Shortage + holding cost of one replenishment decision.

    The fixed order cost is deliberately excluded from the comparison: this
    backtest chooses a *quantity* for a single lead-time window, not an order
    frequency, so a constant per-order fee would cancel out of every candidate
    and only inflate the reported level.  It is still echoed in configuration
    and reported separately as ``order_cost_per_order``.
    """
    shortage_units = max(demand - quantity, 0.0)
    surplus_units = max(quantity - demand, 0.0)
    shortage_cost = shortage_units * costs['shortage_cost_per_unit']
    holding_cost = surplus_units * costs['holding_cost_per_unit_per_day'] * max(1, lead_days)
    return {'shortage_units': shortage_units, 'surplus_units': surplus_units,
            'shortage_cost': shortage_cost, 'holding_cost': holding_cost,
            'total_cost': shortage_cost + holding_cost}


# ---------------------------------------------------------------------------
# Family decisions
# ---------------------------------------------------------------------------

def _snap_quantile(critical, candidates):
    smaller = [q for q in candidates if q <= critical]
    return max(smaller) if smaller else min(candidates)


def _fallback_quantile(family_skus, identities, config):
    """Critical-ratio quantile using family-mean costs when evidence is thin."""
    settings = config['settings']
    costs = [_cost_context(config, identities[sku]) for sku in family_skus]
    mean_shortage = math.fsum(item['shortage_cost_per_unit'] for item in costs) / len(costs)
    mean_holding = math.fsum(item['holding_cost_per_unit_per_day'] for item in costs) / len(costs)
    mean_lead = math.fsum(identities[sku]['lead_days'] for sku in family_skus) / len(costs)
    denominator = mean_shortage + mean_holding * max(1.0, mean_lead)
    critical = mean_shortage / denominator if denominator > 0 else 0.5
    candidates = [q for q in settings['candidate_quantiles']
                  if settings['min_order_quantile'] <= q <= settings['max_order_quantile']]
    return _snap_quantile(critical, candidates)


def _family_evidence(evaluations, family, decision_origin):
    """Exact, exposed, non-promotion windows fully resolved by the decision."""
    return [evaluation for evaluation in evaluations
            if evaluation['family'] == family
            and evaluation['used_for_selection']
            and evaluation['origin'] + evaluation['lead_days'] <= decision_origin]


def _sku_equal_stockout_rate(evidence, model, quantile):
    per_sku = defaultdict(list)
    for evaluation in evidence:
        quantity = evaluation['forecast']['models'][model]['quantities'][quantile]['quantity']
        per_sku[evaluation['sku']].append(1.0 if evaluation['observed_units'] > quantity else 0.0)
    if not per_sku:
        return 0.0
    return math.fsum(math.fsum(values) / len(values) for values in per_sku.values()) / len(per_sku)


def _instance_stockout_rate(evidence, model, quantile):
    if not evidence:
        return 0.0
    hits = sum(1 for evaluation in evidence
               if evaluation['observed_units']
               > evaluation['forecast']['models'][model]['quantities'][quantile]['quantity'])
    return hits / len(evidence)


def _aggregate_cost(evidence, model, quantile):
    """Per-SKU mean cost, then an equal-weighted mean across SKUs.

    The equal weight is the point: a single expensive SKU cannot dominate the
    family decision merely because its unit costs are high.
    """
    per_sku = defaultdict(list)
    shortage_total = 0.0
    holding_total = 0.0
    for evaluation in evidence:
        quantity = evaluation['forecast']['models'][model]['quantities'][quantile]['quantity']
        result = _realised_cost(quantity, evaluation['observed_units'],
                                evaluation['costs'], evaluation['lead_days'])
        per_sku[evaluation['sku']].append(result['total_cost'])
        shortage_total += result['shortage_cost']
        holding_total += result['holding_cost']
    means = {sku: math.fsum(values) / len(values) for sku, values in per_sku.items()}
    total = shortage_total + holding_total
    return {
        'mean_cost_per_sku_origin': math.fsum(means.values()) / len(means) if means else 0.0,
        'shortage_cost_total': shortage_total,
        'holding_cost_total': holding_total,
        'total_cost': total,
        'cost_share_by_sku': ({sku: math.fsum(values) / total for sku, values in per_sku.items()}
                              if total > 0 else {sku: 0.0 for sku in per_sku}),
    }


def _candidate_table(evidence, candidates):
    table = {}
    for model in MODEL_NAMES:
        for quantile in candidates:
            table[(model, quantile)] = {
                **_aggregate_cost(evidence, model, quantile),
                'stockout_rate_sku_equal': _sku_equal_stockout_rate(evidence, model, quantile),
                'stockout_rate_instance': _instance_stockout_rate(evidence, model, quantile),
            }
    return table


def decide_family(family, skus, identities, evaluations, decision_origin, config):
    settings = config['settings']
    candidates = [q for q in settings['candidate_quantiles']
                  if settings['min_order_quantile'] <= q <= settings['max_order_quantile']]
    evidence = _family_evidence(evaluations, family, decision_origin)
    decision = {
        'family': family, 'decision_origin': decision_origin, 'skus': sorted(skus),
        'evidence_windows': len(evidence),
        'evidence_skus': len({evaluation['sku'] for evaluation in evidence}),
        'evidence_rule': ('只使用起点更早、lead_days 窗口在决策起点之前已完整结束、'
                          '且窗口与训练历史都有真实可售暴露的非促销窗口'),
        'candidate_quantiles': candidates,
        'guard': {'max_stockout_rate': settings['max_stockout_rate'],
                  'tolerance': settings['stockout_rate_tolerance'],
                  'measure': 'sku_equal_stockout_rate'},
    }
    if len(evidence) < settings['min_selection_windows']:
        decision.update({
            'selected_model': settings['default_model'],
            'selected_quantile': _fallback_quantile(sorted(skus), identities, config),
            'reason_code': 'insufficient_identifiable_evidence',
            'reason_text': ('可完整辨识且已结束的窗口不足，无法按成本选模型；'
                            '回退到配置默认模型，分位按族均成本的新svendor临界比'),
            'expected_cost_per_sku_origin': None,
            'stockout_rate_sku_equal': None,
            'stockout_rate_instance': None,
            'tie': None,
        })
        return decision

    scored = _candidate_table(evidence, candidates)
    ceiling = settings['max_stockout_rate'] + settings['stockout_rate_tolerance']
    feasible = {key: value for key, value in scored.items()
                if value['stockout_rate_sku_equal'] <= ceiling}
    pool = feasible or scored

    def ranking(key):
        return (scored[key]['mean_cost_per_sku_origin'], key[1], key[0])

    key = min(pool, key=ranking)
    if feasible:
        reason_code = 'cost_minimum_with_stockout_guard'
        reason_text = '在缺货率上限内选取每 SKU-起点期望成本最低的模型与分位'
    else:
        best_rate = min(value['stockout_rate_sku_equal'] for value in scored.values())
        pool = {candidate: value for candidate, value in scored.items()
                if value['stockout_rate_sku_equal'] == best_rate}
        key = min(pool, key=ranking)
        reason_code = 'stockout_guard_no_feasible_candidate'
        reason_text = ('没有候选满足缺货率上限，改选缺货率最低且成本最低的候选；'
                       '该决定需要人工复核')

    best_cost = scored[key]['mean_cost_per_sku_origin']
    tied = sorted(candidate for candidate in pool
                  if abs(scored[candidate]['mean_cost_per_sku_origin'] - best_cost) < 1e-9)
    chosen = scored[key]
    decision.update({
        'selected_model': key[0],
        'selected_quantile': key[1],
        'reason_code': reason_code,
        'reason_text': reason_text,
        'expected_cost_per_sku_origin': chosen['mean_cost_per_sku_origin'],
        'shortage_cost_total': chosen['shortage_cost_total'],
        'holding_cost_total': chosen['holding_cost_total'],
        'total_cost': chosen['total_cost'],
        'order_cost_per_order_excluded_from_selection': config['costs']['order_cost_per_order'],
        'stockout_rate_sku_equal': chosen['stockout_rate_sku_equal'],
        'stockout_rate_instance': chosen['stockout_rate_instance'],
        'cost_share_by_sku': chosen['cost_share_by_sku'],
        'aggregation': ('每个 SKU 先按起点取均值，再对 SKU 等权平均；'
                        '金额总额只作展示，不用于加权'),
        'tie': {
            'tied_candidates': len(tied),
            'breaking_rule': '取最低分位，再按模型名排序' if len(tied) > 1 else None,
            'best_cost': best_cost,
        },
        'alternatives': [
            {'model': candidate[0], 'quantile': candidate[1],
             'expected_cost_per_sku_origin': scored[candidate]['mean_cost_per_sku_origin'],
             'stockout_rate_sku_equal': scored[candidate]['stockout_rate_sku_equal'],
             'feasible': candidate in feasible,
             'rejected_because': ('缺货率超出上限' if candidate not in feasible
                                  else '成本高于所选候选')}
            for candidate in sorted(scored, key=ranking)[:3]
        ],
    })
    return decision


# ---------------------------------------------------------------------------
# Sensitivity for censored / unidentifiable demand
# ---------------------------------------------------------------------------

def _family_sensitivity(family, evaluations, identities, config, policy):
    """Cost sensitivity to the assumed demand inside non-exact windows.

    The family policy (model and quantile) is held fixed so the comparison
    isolates the demand assumption; a changing policy would confound the two.
    """
    settings = config['settings']
    factors = settings['demand_sensitivity_factors']
    relevant = [evaluation for evaluation in evaluations if evaluation['family'] == family]
    if not relevant:
        return []
    results = []
    for factor in factors:
        assumed = []
        unknown_upper = 0
        for evaluation in relevant:
            demand = evaluation['observed_units']
            bounds = evaluation['bounds']
            if bounds is not None:
                if bounds['ceiling_available']:
                    uppers = sorted(value for value in
                                    bounds['demand_estimate_ceiling_by_model'].values()
                                    if value is not None)
                    upper = uppers[len(uppers) // 2]
                    demand = demand + factor * max(0.0, upper - demand)
                else:
                    unknown_upper += 1
            assumed.append((evaluation, demand))
        per_sku = defaultdict(list)
        hits = defaultdict(lambda: [0.0, 0.0])
        shortage_units = 0.0
        surplus_units = 0.0
        for evaluation, demand in assumed:
            model = evaluation['forecast']['models'][policy['model']]
            quantity = model['quantities'][policy['quantile']]['quantity']
            result = _realised_cost(quantity, demand, evaluation['costs'], evaluation['lead_days'])
            per_sku[evaluation['sku']].append(result['total_cost'])
            hits[evaluation['sku']][1] += 1.0
            if demand > quantity:
                hits[evaluation['sku']][0] += 1.0
            shortage_units += result['shortage_units']
            surplus_units += result['surplus_units']
        means = [math.fsum(values) / len(values) for values in per_sku.values()]
        results.append({
            'factor': factor,
            'interpretation': ('0 表示只按已观测销量（丢失需求的下界）；1 表示叠加'
                               '模型对缺失暴露的敏感上界；这是敏感性区间，'
                               '不是恢复的真实订单'),
            'policy_held_fixed': {'model': policy['model'], 'quantile': policy['quantile']},
            'windows_without_upper_bound': unknown_upper,
            'expected_cost_per_sku_origin': math.fsum(means) / len(means) if means else 0.0,
            'stockout_rate_sku_equal': (math.fsum(a / b for a, b in hits.values()) / len(hits)
                                        if hits else 0.0),
            'shortage_units_total': shortage_units,
            'surplus_units_total': surplus_units,
        })
    return results


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------



def _validate_unique_days(records):
    seen = set()
    for row in records:
        key = (row['sku'], row['day'])
        if key in seen:
            raise ValueError(f'{row["sku"]} 在 day={row["day"]} 有重复观测，回测要求每日至多一行')
        seen.add(key)
        if row['sold'] < 0:
            raise ValueError(f'{row["sku"]} 在 day={row["day"]} 的 sold 不能为负')
        if row['available_hours'] < 0 or row['available_hours'] > 24:
            raise ValueError(
                f'{row["sku"]} 在 day={row["day"]} 的 available_hours 必须在 0 到 24 之间')


def run_backtest(records, config):
    settings = config['settings']
    _validate_unique_days(records)
    skus = defaultdict(list)
    for row in records:
        skus[row['sku']].append(row)
    for sku in skus:
        skus[sku].sort(key=lambda row: row['day'])
    identities = {}
    timeline = []
    for sku in sorted(skus):
        rows = skus[sku]
        identities[sku] = _sku_identity(rows)
        for origin in _candidate_origins(rows, settings):
            timeline.append((origin, sku))
    timeline.sort()

    forecasts = []
    for origin, sku in timeline:
        forecast = build_forecast(identities[sku], skus[sku], origin, config)
        forecast['costs'] = _cost_context(config, identities[sku])
        forecasts.append(forecast)

    # Future rows are read only from here on: first to score frozen forecasts,
    # then to decide family policy using windows resolved before each origin.
    evaluations = []
    for forecast in forecasts:
        rows = skus[forecast['sku']]
        scored = score_forecast(identities[forecast['sku']], rows,
                                forecast['origin'], forecast, config)
        evaluations.append({
            **scored,
            'sku': forecast['sku'], 'family': forecast['family'],
            'origin': forecast['origin'], 'lead_days': forecast['lead_days'],
            'costs': forecast['costs'],
            'forecast': {'models': {name: {key: value for key, value in model.items()}
                                    for name, model in forecast['models'].items()}},
        })
    families = sorted({identity['family'] for identity in identities.values()})
    members = defaultdict(list)
    for sku, identity in identities.items():
        members[identity['family']].append(sku)
    decision_origins = sorted({forecast['origin'] for forecast in forecasts})
    decisions = {(family, origin): decide_family(
        family, members[family], identities, evaluations, origin, config)
        for family in families for origin in decision_origins}

    for forecast in forecasts:
        decision = decisions[(forecast['family'], forecast['origin'])]
        model = forecast['models'][decision['selected_model']]
        forecast['recommendation'] = {
            'decision_origin': decision['decision_origin'],
            'model': decision['selected_model'],
            'quantile': decision['selected_quantile'],
            'quantity': model['quantities'][decision['selected_quantile']]['quantity'],
            'reason_code': decision['reason_code'],
            'history_has_exposure': forecast['exposure_evidence'],
            'demand_status': forecast['demand_status'],
        }

    final_origin = decision_origins[-1] if decision_origins else None
    decision_timeline = []
    for family in families:
        for origin in decision_origins:
            decision = decisions[(family, origin)]
            decision_timeline.append({
                'family': family, 'decision_origin': origin,
                'evidence_windows': decision['evidence_windows'],
                'selected_model': decision['selected_model'],
                'selected_quantile': decision['selected_quantile'],
                'reason_code': decision['reason_code'],
            })
    counts = defaultdict(int)
    for evaluation in evaluations:
        counts[evaluation['classification']] += 1

    reported = []
    for forecast, evaluation in zip(forecasts, evaluations):
        public = {key: forecast[key] for key in (
            'sku', 'family', 'origin', 'lead_days', 'known_days', 'training_rows',
            'training_events', 'promotion_rows_excluded_from_training',
            'training_all_promotion', 'exposure_evidence', 'demand_status', 'is_new_sku',
            'zero_event_upper_bound', 'exposure', 'training_exposure',
            'quantiles_considered', 'recommendation')}
        public['models'] = forecast['models']
        public['evaluation'] = {key: evaluation[key] for key in (
            'classification', 'classification_reason', 'window_days', 'missing_days',
            'observed_units', 'observed_available_hours', 'observed_units_are_lower_bound',
            'promotion_in_window', 'used_for_selection', 'selection_exclusion_reason',
            'demand_identifiable', 'bounds')}
        reported.append(public)

    return {
        'product': 'intermittent-demand', 'schema_version': SCHEMA_VERSION,
        'config': config, 'config_fingerprint': config_fingerprint(config),
        'summary': {
            'sku_count': len(identities), 'family_count': len(families),
            'origin_count': len(evaluations),
            'classification_counts': {key: counts.get(key, 0)
                                      for key in (EXACT, CENSORED, UNIDENTIFIABLE, MISSING)},
            'exact_windows': counts.get(EXACT, 0),
            'censored_windows': counts.get(CENSORED, 0),
            'unidentifiable_windows': counts.get(UNIDENTIFIABLE, 0),
            'windows_without_exposure_evidence': sum(
                1 for forecast in forecasts if not forecast['exposure_evidence']),
        },
        'assumptions': [
            '未来可售性按已知窗口的可售比例外推，只作为预测输入，不回填训练',
            '缺失暴露下的需求上界依赖模型假设，不是缺货损失的严格上限',
            'available_hours=0 的日不提供销售暴露，也不计为需求为零',
            '部分可售日的已售量只作为下界，未按小时比例无条件放大',
            '每个起点的训练与每个族决策只用当时已知且已完整结束的记录',
            '缺货期间需求只给上下界与敏感性区间，不声称恢复真实丢失订单',
        ],
        'family_decisions': ([decisions[(family, final_origin)] for family in families]
                             if final_origin is not None else []),
        'decision_timeline': decision_timeline,
        'sensitivity': {family: _family_sensitivity(
            family, evaluations, identities, config,
            {'model': decisions[(family, final_origin)]['selected_model'],
             'quantile': decisions[(family, final_origin)]['selected_quantile']}
            if final_origin is not None else
            {'model': settings['default_model'], 'quantile': settings['min_order_quantile']})
            for family in families},
        'unidentifiable': [{
            'sku': evaluation['sku'], 'family': evaluation['family'],
            'origin': evaluation['origin'], 'lead_days': evaluation['lead_days'],
            'reason': evaluation['classification_reason'],
            'observed_units_lower_bound': evaluation['observed_units'],
            'missing_days': evaluation['missing_days'],
            'bounds': evaluation['bounds'],
            'used_for_selection': evaluation['used_for_selection'],
        } for evaluation in evaluations if evaluation['classification'] != EXACT],
        'forecasts': reported,
    }
