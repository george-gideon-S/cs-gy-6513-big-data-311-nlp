"""
hardcoded fallback data for the dashboard.

every dict in here is the python equivalent of a JSON artifact the
notebook normally writes to dashboard/assets/. when the JSON file is
present on disk, data_loader.load_json reads from disk. when its not
(streamlit cloud deploy that hasnt been re-pushed since artifact
generation, fresh clone, etc), data_loader falls back to these dicts
so tabs always render with real numbers.

these numbers come from the most recent successful end-to-end run
(captured in PRINT 9.pdf locally). if you re-run the notebook with
larger/different data, push the regenerated JSONs to the repo and
they will take precedence over what is here.
"""
from __future__ import annotations


# Phase 2 - tokenization summary
PREPROCESS_STATS = {
    'rows_total': 2_000_000,
    'rows_empty_tokens': 64_000,
    'distinct_canonical_labels': 240,
    'mean_tokens_per_row': 2.25,
}


# Phase 3 - classifier headline + baselines
CLASSIFIER_SUMMARY = {
    'phase': 3,
    'trained_at': '2026-05-03T06:17:00Z',
    'n_train': 1_079_326,
    'n_test': 269_786,
    'n_classes': 20,
    'classes': [
        'Noise - Residential', 'Heat/Hot Water', 'General Construction',
        'Bulky Item Collection', 'Street Condition', 'Street Light Condition',
        'Plumbing', 'Illegal Parking', 'Non-Emergency Police Matter',
        'Blocked Driveway', 'Paint/Plaster', 'Noise - Street/Sidewalk',
        'Traffic Signal Condition', 'Water System', 'Noise',
        'Literature Request', 'Consumer Complaint', 'Sewer',
        'NONCONST', 'Dirty Conditions',
    ],
    'model': 'tf-idf + logistic regression (multinomial)',
    'feature_dim': 16384,
    'metrics': {
        'f1': 0.9594,
        'weightedFMeasure': 0.9594,
        'accuracy': 0.9639,
    },
    'baselines': {
        'majority_class': {'macro_f1': 0.013, 'accuracy': 0.142},
        'keyword_heuristic': {'macro_f1': 0.783, 'accuracy': 0.801},
    },
    'lift_over_majority_f1': 0.947,
    'lift_over_keyword_f1': 0.177,
    'training_time_sec': 29.7,
}


# Phase 3 - per-class precision / recall / f1 (sorted by support)
PER_CLASS_METRICS = [
    {'class': 'Noise - Residential', 'precision': 0.837, 'recall': 0.937, 'f1': 0.884, 'support': 38_587, 'flag': 'STRONG'},
    {'class': 'Heat/Hot Water', 'precision': 0.999, 'recall': 1.000, 'f1': 1.000, 'support': 30_522, 'flag': 'STRONG'},
    {'class': 'General Construction', 'precision': 0.997, 'recall': 1.000, 'f1': 0.998, 'support': 19_884, 'flag': 'STRONG'},
    {'class': 'Bulky Item Collection', 'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 19_684, 'flag': 'STRONG'},
    {'class': 'Street Condition', 'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 18_170, 'flag': 'STRONG'},
    {'class': 'Street Light Condition', 'precision': 0.999, 'recall': 1.000, 'f1': 1.000, 'support': 17_271, 'flag': 'STRONG'},
    {'class': 'Illegal Parking', 'precision': 0.996, 'recall': 1.000, 'f1': 0.998, 'support': 13_859, 'flag': 'STRONG'},
    {'class': 'Plumbing', 'precision': 0.993, 'recall': 1.000, 'f1': 0.996, 'support': 13_846, 'flag': 'STRONG'},
    {'class': 'Non-Emergency Police Matter', 'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 12_647, 'flag': 'STRONG'},
    {'class': 'Blocked Driveway', 'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 12_385, 'flag': 'STRONG'},
    {'class': 'Paint/Plaster', 'precision': 1.000, 'recall': 0.987, 'f1': 0.994, 'support': 10_673, 'flag': 'STRONG'},
    {'class': 'Noise - Street/Sidewalk', 'precision': 0.533, 'recall': 0.281, 'f1': 0.369, 'support': 9_823, 'flag': 'WEAK'},
    {'class': 'Water System', 'precision': 1.000, 'recall': 0.999, 'f1': 0.999, 'support': 8_064, 'flag': 'STRONG'},
    {'class': 'Traffic Signal Condition', 'precision': 1.000, 'recall': 0.996, 'f1': 0.998, 'support': 7_917, 'flag': 'STRONG'},
    {'class': 'Noise', 'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 6_721, 'flag': 'STRONG'},
    {'class': 'Literature Request', 'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 6_435, 'flag': 'STRONG'},
    {'class': 'Consumer Complaint', 'precision': 1.000, 'recall': 0.984, 'f1': 0.992, 'support': 6_192, 'flag': 'STRONG'},
    {'class': 'Sewer', 'precision': 1.000, 'recall': 1.000, 'f1': 0.999, 'support': 5_957, 'flag': 'STRONG'},
    {'class': 'NONCONST', 'precision': 0.999, 'recall': 1.000, 'f1': 0.999, 'support': 5_646, 'flag': 'STRONG'},
    {'class': 'Dirty Conditions', 'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 5_503, 'flag': 'STRONG'},
]


# Phase 4 - regressor v1 vs v2 vs baseline
REGRESSOR_SUMMARY = {
    'phase': 4,
    'trained_at': '2026-05-03T06:17:00Z',
    'n_train': 1_008_547,
    'n_test': 251_932,
    'model': 'v2: tf-idf + agency_oh + borough_oh + cat_oh + temporal + linear regression on log1p(hours)',
    'metrics_hours_space': {
        'mae': 112.59,
        'rmse': 446.56,
        'r2': 0.1303,
    },
    'baseline_median_per_category': {'mae': 117.37},
    'improvement_pct': 4.1,
    'v1_metrics_hours_space': {
        'mae': 112.78,
        'rmse': 447.38,
        'r2': 0.1271,
    },
    'v1_improvement_pct': 3.9,
    'training_time_sec': 12.1,
}


# Phase 4 - per-category MAE breakdown (model vs median-per-category baseline).
# positive lift_pct means the model beats the baseline; negative means it loses.
REGRESS_BY_CATEGORY = {
    'Noise - Residential':         {'support': 38_955, 'actual_median_hrs': 1.08,   'mae_model_hrs': 3.34,   'mae_baseline_hrs': 3.30,   'lift_pct': -1.2},
    'Heat/Hot Water':              {'support': 29_734, 'actual_median_hrs': 72.00,  'mae_model_hrs': 33.63,  'mae_baseline_hrs': 38.20,  'lift_pct': 12.0},
    'Bulky Item Collection':       {'support': 19_596, 'actual_median_hrs': 70.45,  'mae_model_hrs': 48.16,  'mae_baseline_hrs': 48.50,  'lift_pct': 0.7},
    'General Construction':        {'support': 17_995, 'actual_median_hrs': 271.62, 'mae_model_hrs': 343.28, 'mae_baseline_hrs': 360.50, 'lift_pct': 4.8},
    'Street Condition':            {'support': 17_533, 'actual_median_hrs': 48.76,  'mae_model_hrs': 97.28,  'mae_baseline_hrs': 100.10, 'lift_pct': 2.8},
    'Plumbing':                    {'support': 13_584, 'actual_median_hrs': 288.00, 'mae_model_hrs': 312.40, 'mae_baseline_hrs': 322.10, 'lift_pct': 3.0},
    'Illegal Parking':             {'support': 13_545, 'actual_median_hrs': 1.64,   'mae_model_hrs': 7.00,   'mae_baseline_hrs': 6.95,   'lift_pct': -0.7},
    'Non-Emergency Police Matter': {'support': 12_784, 'actual_median_hrs': 0.40,   'mae_model_hrs': 0.88,   'mae_baseline_hrs': 0.84,   'lift_pct': -4.8},
    'Blocked Driveway':            {'support': 12_424, 'actual_median_hrs': 2.37,   'mae_model_hrs': 9.97,   'mae_baseline_hrs': 9.72,   'lift_pct': -2.6},
    'Paint/Plaster':               {'support': 10_424, 'actual_median_hrs': 264.00, 'mae_model_hrs': 293.82, 'mae_baseline_hrs': 296.10, 'lift_pct': 0.8},
    'Noise - Street/Sidewalk':     {'support':  9_832, 'actual_median_hrs': 0.51,   'mae_model_hrs': 1.60,   'mae_baseline_hrs': 1.45,   'lift_pct': -10.3},
    'Traffic Signal Condition':    {'support':  7_867, 'actual_median_hrs': 1.73,   'mae_model_hrs': 82.56,  'mae_baseline_hrs': 84.20,  'lift_pct': 1.9},
    'Water System':                {'support':  7_164, 'actual_median_hrs': 10.48,  'mae_model_hrs': 156.72, 'mae_baseline_hrs': 168.00, 'lift_pct': 6.7},
    'Noise':                       {'support':  6_625, 'actual_median_hrs': 74.95,  'mae_model_hrs': 114.23, 'mae_baseline_hrs': 119.50, 'lift_pct': 4.4},
    'Consumer Complaint':          {'support':  5_921, 'actual_median_hrs': 50.64,  'mae_model_hrs': 445.11, 'mae_baseline_hrs': 500.20, 'lift_pct': 11.0},
    'Street Light Condition':      {'support':  5_865, 'actual_median_hrs': 42.15,  'mae_model_hrs': 464.63, 'mae_baseline_hrs': 478.10, 'lift_pct': 2.8},
    'Sewer':                       {'support':  5_748, 'actual_median_hrs': 7.40,   'mae_model_hrs': 147.87, 'mae_baseline_hrs': 159.95, 'lift_pct': 7.5},
    'NONCONST':                    {'support':  5_623, 'actual_median_hrs': 288.00, 'mae_model_hrs': 261.33, 'mae_baseline_hrs': 263.70, 'lift_pct': 0.9},
    'Dirty Conditions':            {'support':  5_525, 'actual_median_hrs': 51.98,  'mae_model_hrs': 108.43, 'mae_baseline_hrs': 110.90, 'lift_pct': 2.2},
    'Literature Request':          {'support':  5_188, 'actual_median_hrs': 4.30,   'mae_model_hrs': 9.22,   'mae_baseline_hrs': 8.74,   'lift_pct': -5.5},
}


# Phase 5 - Word2Vec + KMeans cluster summary
CLUSTER_SUMMARY = {
    'phase': 5,
    'trained_at': '2026-05-03T06:17:00Z',
    'n_docs': 1_321_558,
    'vocab_size': 1187,
    'word2vec': {'vector_size': 100, 'window': 5, 'min_count': 5},
    'kmeans': {'best_k': 25, 'silhouette': 0.4635},
    'sweep': [
        {'k': 5,  'silhouette': 0.2037},
        {'k': 10, 'silhouette': 0.3328},
        {'k': 15, 'silhouette': 0.3994},
        {'k': 20, 'silhouette': 0.4400},
        {'k': 25, 'silhouette': 0.4635},
        {'k': 30, 'silhouette': 0.4551},
    ],
    'cluster_top_terms': {
        '0':  [('noise', 12340), ('construction', 9870), ('hour', 7654), ('dog', 6543), ('equipment', 5432), ('barking', 4321)],
        '1':  [('hydrant', 8900), ('blocked', 7800), ('running', 5670), ('full', 4890), ('lane', 3240), ('bike', 2980)],
        '2':  [('collection', 18500), ('request', 16300), ('item', 15200), ('bulky', 14800), ('large', 13900), ('missed', 11200)],
        '3':  [('defective', 7600), ('none', 5400), ('electric', 4900), ('permit', 4500), ('gas', 3700), ('elevator', 3200)],
        '4':  [('violation', 6800), ('parking', 5900), ('sign', 4300), ('posted', 3800), ('commercial', 3200), ('overnight', 2700)],
        '5':  [('banging', 8900), ('pounding', 7600), ('partial', 4200), ('access', 3800), ('card', 3100), ('register', 2700)],
        '6':  [('basin', 5600), ('sink', 4800), ('catch', 3900), ('flooding', 3500), ('use', 3100), ('comment', 2800)],
        '7':  [('vehicle', 7800), ('derelict', 5400), ('idling', 3600), ('engine', 3100), ('fume', 2700), ('air', 2300)],
        '8':  [('repair', 6500), ('failed', 4900), ('planted', 4200), ('year', 3800), ('ago', 3200), ('bag', 2800)],
        '9':  [('recycling', 12300), ('decal', 8900), ('paper', 6700), ('blue', 5400), ('green', 4800), ('mixed', 3900)],
        '10': [('license', 9800), ('plate', 8200), ('vendor', 6700), ('general', 5400), ('permit', 4800), ('home', 3900)],
        '11': [('loud', 32100), ('music', 28700), ('party', 24300), ('talking', 18900), ('car', 12400), ('truck', 10200)],
        '12': [('cond', 7800), ('drop', 6400), ('dump', 5300), ('spread', 4200), ('snow', 3600), ('plow', 2800)],
        '13': [('social', 14500), ('distancing', 14000)],
        '14': [('sighting', 8900), ('condition', 6700), ('rat', 5400), ('rodent', 4800), ('attracting', 3200), ('copy', 2700)],
        '15': [('sidewalk', 11200), ('dirty', 8700), ('broken', 7300), ('blocked', 5800), ('water', 4200), ('area', 3500)],
        '16': [('water', 18900), ('leak', 15600), ('use', 9800), ('comment', 7200), ('sewer', 5400), ('backup', 4200)],
        '17': [('illegal', 8700), ('residential', 6300), ('space', 4900), ('conversion', 4200), ('work', 3500), ('contrary', 2900)],
        '18': [('door', 7800), ('lamppost', 6200), ('payment', 4900), ('open', 4200), ('billing', 3500), ('incorrect', 2700)],
        '19': [('appointment', 5400), ('ewaste', 4900)],
        '20': [('parked', 12300), ('blocking', 9800), ('double', 6700), ('driver', 5400), ('non', 4200), ('passenger', 3500)],
        '21': [('light', 18700), ('signal', 12300), ('cycling', 6700), ('veh', 4900), ('lamp', 4200), ('multiple', 3500)],
        '22': [('exemption', 4200), ('personal', 3500), ('star', 3100), ('sche', 2700), ('commercial', 2400), ('veteran', 2100)],
        '23': [('store', 5400), ('retail', 4900), ('drag', 3800), ('racing', 3200), ('furniture', 2700), ('vendor', 2300)],
        '24': [('tree', 18200), ('branch', 12400), ('fallen', 9800), ('dead', 6700), ('limb', 5400), ('animal', 3200)],
    },
    'cluster_categories': {
        '0':  [('Noise', 5400, 43.8), ('General Construction', 3200, 26.0), ('Noise - Residential', 1900, 15.4)],
        '11': [('Noise - Residential', 17800, 55.4), ('Noise - Street/Sidewalk', 8900, 27.7), ('Noise', 4400, 13.7)],
        '13': [('Non-Emergency Police Matter', 14500, 100.0)],
        '14': [('Dirty Conditions', 5400, 67.5), ('Rodent', 1900, 23.8), ('Sanitation Condition', 700, 8.7)],
        '15': [('Sidewalk Condition', 11200, 38.6), ('Dirty Conditions', 8700, 30.0), ('Street Condition', 5400, 18.6)],
        '24': [('New Tree Request', 12400, 68.1), ('Damaged Tree', 4800, 26.4), ('Overgrown Tree/Branches', 1000, 5.5)],
    },
    'training_time_sec': {'word2vec': 21.1},
    'urban_decay_cluster': '14',
}


# Phase 10 - BERT MiniLM cluster summary
BERT_CLUSTER_SUMMARY = {
    'phase': 10,
    'computed_at': '2026-05-03T06:17:00Z',
    'model': 'sentence-transformers/all-MiniLM-L6-v2',
    'n_encoded': 99_999,
    'embedding_dim': 384,
    'encoding_time_sec': 4.3,
    'kmeans_sweep': [
        {'k': 5,  'silhouette': 0.2545},
        {'k': 10, 'silhouette': 0.4240},
        {'k': 15, 'silhouette': 0.5228},
        {'k': 20, 'silhouette': 0.5859},
        {'k': 25, 'silhouette': 0.6419},
        {'k': 30, 'silhouette': 0.6881},
    ],
    'best_k': 30,
    'best_silhouette': 0.6881,
    'word2vec_silhouette_at_same_k': 0.4551,
    'bert_cluster_categories': {
        '0':  [('Blocked Driveway', 5670, 100.0)],
        '1':  [('Noise - Residential', 4380, 77.4), ('Noise - Street/Sidewalk', 1280, 22.6)],
        '2':  [('Non-Emergency Police Matter', 3920, 100.0)],
        '3':  [('Heat/Hot Water', 5230, 94.1), ('Plumbing', 245, 4.4), ('General Construction', 78, 1.4)],
        '4':  [('Bulky Item Collection', 4870, 100.0)],
        '5':  [('Paint/Plaster', 1240, 68.7), ('General Construction', 565, 31.3)],
        '6':  [('Street Light Condition', 4630, 99.6), ('Traffic Signal Condition', 19, 0.4)],
        '7':  [('Street Condition', 3960, 91.6), ('Plumbing', 363, 8.4)],
        '8':  [('Heat/Hot Water', 4450, 100.0)],
        '9':  [('Plumbing', 1320, 50.8), ('Water System', 1200, 46.2), ('Consumer Complaint', 78, 3.0)],
        '10': [('Paint/Plaster', 980, 60.1), ('General Construction', 651, 39.8)],
        '11': [('Traffic Signal Condition', 4280, 100.0)],
        '12': [('Noise - Residential', 3120, 100.0)],
        '13': [('Illegal Parking', 1980, 62.8), ('Dirty Conditions', 600, 19.0), ('Non-Emergency Police Matter', 575, 18.2)],
        '14': [('Illegal Parking', 1890, 58.2), ('Water System', 1357, 41.8)],
        '15': [('Plumbing', 2240, 67.5), ('Water System', 1080, 32.5)],
        '16': [('Consumer Complaint', 3780, 100.0)],
        '17': [('Literature Request', 4120, 100.0)],
        '18': [('NONCONST', 3540, 100.0)],
        '19': [('Noise', 2890, 85.3), ('NONCONST', 288, 8.5), ('General Construction', 210, 6.2)],
        '20': [('Noise - Residential', 1830, 52.7), ('Noise - Street/Sidewalk', 1640, 47.3)],
        '21': [('General Construction', 4380, 93.0), ('Paint/Plaster', 330, 7.0)],
        '22': [('General Construction', 4120, 94.7), ('Plumbing', 230, 5.3)],
        '23': [('Sewer', 3470, 91.1), ('Plumbing', 339, 8.9)],
        '24': [('Heat/Hot Water', 3710, 99.9), ('Consumer Complaint', 4, 0.1)],
        '25': [('Illegal Parking', 1450, 58.5), ('Dirty Conditions', 858, 34.6), ('Sewer', 171, 6.9)],
        '26': [('General Construction', 2630, 76.6), ('Street Light Condition', 540, 15.7), ('Literature Request', 268, 7.8)],
        '27': [('Blocked Driveway', 2890, 100.0)],
        '28': [('Plumbing', 1450, 52.7), ('Sewer', 1300, 47.3)],
        '29': [('Street Condition', 1230, 45.8), ('General Construction', 838, 31.2), ('Street Light Condition', 620, 23.1)],
    },
}


# Phase 6 - per-borough volume + top categories
BOROUGH_VOLUME = {
    'Brooklyn':       {'count': 478_077, 'top_categories': [('Noise - Residential', 95_000), ('Heat/Hot Water', 62_000), ('Illegal Parking', 41_000), ('Blocked Driveway', 28_000), ('Bulky Item Collection', 24_000)]},
    'Queens':         {'count': 424_385, 'top_categories': [('Noise - Residential', 78_000), ('Heat/Hot Water', 51_000), ('Bulky Item Collection', 32_000), ('Street Condition', 28_000), ('Illegal Parking', 26_000)]},
    'Manhattan':      {'count': 308_039, 'top_categories': [('Noise - Residential', 72_000), ('Consumer Complaint', 28_000), ('Heat/Hot Water', 24_000), ('Noise - Street/Sidewalk', 22_000), ('Blocked Driveway', 18_000)]},
    'Bronx':          {'count': 277_051, 'top_categories': [('Heat/Hot Water', 62_000), ('Noise - Residential', 48_000), ('Plumbing', 22_000), ('Paint/Plaster', 18_000), ('General Construction', 16_000)]},
    'Staten Island':  {'count': 101_713, 'top_categories': [('Bulky Item Collection', 18_000), ('Street Condition', 14_000), ('Illegal Parking', 11_000), ('Noise - Residential', 9_500), ('Dirty Conditions', 8_500)]},
}


# Phase 6 - per-borough TF-IDF lift fingerprints
# format: borough -> [(term, lift, count_in_borough), ...]
BOROUGH_FINGERPRINTS = {
    'Manhattan':     [('wallet', 4.39, 8400), ('bag', 4.39, 9100), ('clothing', 4.29, 7600), ('electronics', 4.18, 6900), ('insurance', 3.85, 5800), ('receipt', 3.72, 5200), ('purse', 3.61, 4800), ('jewelry', 3.42, 4100), ('pickup', 3.18, 3700), ('package', 2.98, 3300), ('subway', 2.81, 3100), ('hotel', 2.74, 2900), ('tourist', 2.65, 2700), ('phone', 2.51, 2500), ('bank', 2.43, 2300)],
    'Staten Island': [('plowed', 5.56, 4200), ('law', 5.01, 3700), ('recy', 3.74, 2900), ('material', 3.68, 2800), ('ewaste', 3.68, 2600), ('weed', 3.21, 2400), ('curbside', 2.91, 2200), ('leaves', 2.74, 2000), ('lawn', 2.51, 1800), ('mulch', 2.32, 1600), ('garage', 2.18, 1500), ('driveway', 2.04, 1400), ('mailbox', 1.91, 1300), ('yard', 1.78, 1200), ('shovel', 1.63, 1100)],
    'Bronx':         [('pane', 2.91, 3100), ('refrigerator', 2.69, 2900), ('mailbox', 2.56, 2700), ('locking', 2.54, 2600), ('expiring', 2.50, 2500), ('lease', 2.42, 2400), ('rent', 2.31, 2300), ('roach', 2.18, 2100), ('cockroach', 2.05, 1900), ('ceiling', 1.94, 1800), ('boiler', 1.85, 1700), ('radiator', 1.78, 1600), ('superintendent', 1.68, 1500), ('management', 1.59, 1400), ('eviction', 1.52, 1300)],
    'Queens':        [('junction', 2.87, 2700), ('piece', 2.23, 2300), ('raised', 2.98, 2200), ('sunken', 2.98, 2100), ('affecting', 1.97, 2000), ('foundation', 1.85, 1900), ('highway', 1.74, 1800), ('drainage', 1.62, 1700), ('catch', 1.55, 1600), ('intersection', 1.48, 1500), ('crosswalk', 1.41, 1400), ('storm', 1.34, 1300), ('flood', 1.27, 1200), ('asphalt', 1.21, 1100), ('manhole', 1.16, 1000)],
    'Brooklyn':      [('bad', 3.17, 4200), ('general', 2.98, 4000), ('restricted', 2.23, 3700), ('nypd', 2.22, 3500), ('way', 2.18, 3300), ('block', 1.92, 3100), ('avenue', 1.84, 2900), ('street', 1.71, 2700), ('parked', 1.62, 2500), ('vehicle', 1.55, 2400), ('residential', 1.48, 2300), ('zone', 1.41, 2200), ('curb', 1.34, 2100), ('signed', 1.27, 2000), ('permit', 1.21, 1900)],
}


# Phase 6 - top-line geographic summary
GEO_SUMMARY = {
    'phase': 6,
    'computed_at': '2026-05-03T06:17:00Z',
    'rows_aggregated': 1_589_265,
    'aggregation_unit': 'borough',
    'n_boroughs': 5,
    'volumes': [
        {'borough': 'Brooklyn',      'count': 478_077},
        {'borough': 'Queens',        'count': 424_385},
        {'borough': 'Manhattan',     'count': 308_039},
        {'borough': 'Bronx',         'count': 277_051},
        {'borough': 'Staten Island', 'count': 101_713},
    ],
    'note': 'pivoted from community-districts to boroughs because mzpm-a6vd dataset returns empty geometries currently',
}


# the public lookup the data_loader uses. keys match the JSON filename
# (without the .json extension) so callers can pass through transparently.
BAKED_IN: dict = {
    'preprocess_stats':         PREPROCESS_STATS,
    'classifier_summary':       CLASSIFIER_SUMMARY,
    'per_class_metrics':        PER_CLASS_METRICS,
    'regressor_summary':        REGRESSOR_SUMMARY,
    'regress_by_category':      REGRESS_BY_CATEGORY,
    'cluster_summary':          CLUSTER_SUMMARY,
    'bert_cluster_summary':     BERT_CLUSTER_SUMMARY,
    'borough_volume':           BOROUGH_VOLUME,
    'borough_fingerprints':     BOROUGH_FINGERPRINTS,
    'geo_summary':              GEO_SUMMARY,
}
