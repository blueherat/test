WITH report_evidence(
  section,
  row_order,
  group_name,
  role,
  rollback_step,
  pair_index,
  numerator,
  denominator,
  rate,
  difference,
  source_artifact_id,
  note
) AS (
  VALUES
    ('signal_cell', 1, 'E0B0', NULL, NULL, NULL, 1575, 1740, 1575.0 / 1740.0, NULL,
     'ded658ed3008f1639aca2671db9e5add48b3e9a27765fa870d8ec92aa81c3616',
     'Retrospective internal-only path count'),
    ('signal_cell', 2, 'E0B1', NULL, NULL, NULL, 139, 1740, 139.0 / 1740.0, NULL,
     'ded658ed3008f1639aca2671db9e5add48b3e9a27765fa870d8ec92aa81c3616',
     'Retrospective internal-only path count'),
    ('signal_cell', 3, 'E1B0', NULL, NULL, NULL, 18, 1740, 18.0 / 1740.0, NULL,
     'ded658ed3008f1639aca2671db9e5add48b3e9a27765fa870d8ec92aa81c3616',
     'Retrospective internal-only path count'),
    ('signal_cell', 4, 'E1B1', NULL, NULL, NULL, 8, 1740, 8.0 / 1740.0, NULL,
     'ded658ed3008f1639aca2671db9e5add48b3e9a27765fa870d8ec92aa81c3616',
     'Retrospective internal-only path count'),

    ('blind_alarm_audit', 10, 'E alarm: strict clear-bad', 'E alarm', NULL, NULL, 1, 26, 1.0 / 26.0, NULL,
     'd21e1b85d64cf990945eaae9ddab1349238fc6f4769a11672ba1c8d4722712fc',
     'Strict 2-of-2 score-blind consensus'),
    ('blind_alarm_audit', 11, 'Matched control: strict clear-bad', 'control', NULL, NULL, 2, 26, 2.0 / 26.0, NULL,
     'd21e1b85d64cf990945eaae9ddab1349238fc6f4769a11672ba1c8d4722712fc',
     'Same-class exact-start-schedule control'),
    ('blind_alarm_audit', 12, 'E alarm: blur/fusion', 'E alarm', NULL, NULL, 9, 26, 9.0 / 26.0, NULL,
     'd21e1b85d64cf990945eaae9ddab1349238fc6f4769a11672ba1c8d4722712fc',
     'Strict 2-of-2 score-blind consensus'),
    ('blind_alarm_audit', 13, 'Matched control: blur/fusion', 'control', NULL, NULL, 5, 26, 5.0 / 26.0, NULL,
     'd21e1b85d64cf990945eaae9ddab1349238fc6f4769a11672ba1c8d4722712fc',
     'Paired one-sided exact p for alarm enrichment = 0.0625'),

    ('opportunity_path', 20, 'Joint E+B', 'joint_E_and_B', NULL, NULL, 3, 8, 3.0 / 8.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Frozen guard requires at least four opportunity paths per role'),
    ('opportunity_path', 21, 'Matched B-only', 'B_only_exact_schedule_B_matched_control', NULL, NULL, 6, 8, 6.0 / 8.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Frozen guard fails because joint has only three paths'),

    ('strict_repair', 30, 'Joint E+B · all', 'joint_E_and_B', NULL, NULL, 10, 64, 10.0 / 64.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Protocol-literal blur reduction, preservation, and no fresh clear-bad'),
    ('strict_repair', 31, 'Matched B-only · all', 'B_only_exact_schedule_B_matched_control', NULL, NULL, 17, 64, 17.0 / 64.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Protocol-literal blur reduction, preservation, and no fresh clear-bad'),
    ('strict_repair', 32, 'Joint E+B · opportunities', 'joint_E_and_B', NULL, NULL, 10, 23, 10.0 / 23.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Unmatched pooled opportunity rate; three underlying paths'),
    ('strict_repair', 33, 'Matched B-only · opportunities', 'B_only_exact_schedule_B_matched_control', NULL, NULL, 17, 48, 17.0 / 48.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Unmatched pooled opportunity rate; six underlying paths'),
    ('strict_repair', 34, 'Joint E+B · step109 opportunities', 'joint_E_and_B', 109, NULL, 5, 11, 5.0 / 11.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Exploratory role-by-step result'),
    ('strict_repair', 35, 'Matched B-only · step109 opportunities', 'B_only_exact_schedule_B_matched_control', 109, NULL, 8, 24, 8.0 / 24.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Exploratory role-by-step result'),
    ('strict_repair', 36, 'Joint E+B · step149 opportunities', 'joint_E_and_B', 149, NULL, 5, 12, 5.0 / 12.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Exploratory role-by-step result'),
    ('strict_repair', 37, 'Matched B-only · step149 opportunities', 'B_only_exact_schedule_B_matched_control', 149, NULL, 9, 24, 9.0 / 24.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Exploratory role-by-step result'),

    ('matched_path_difference', 40, 'Pair 0', NULL, NULL, 0, 0, 8, NULL, -0.375,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Joint strict repair rate minus matched B-only strict repair rate'),
    ('matched_path_difference', 41, 'Pair 1', NULL, NULL, 1, 0, 8, NULL, 0.0,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Joint strict repair rate minus matched B-only strict repair rate'),
    ('matched_path_difference', 42, 'Pair 2', NULL, NULL, 2, 0, 8, NULL, 0.0,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Joint strict repair rate minus matched B-only strict repair rate'),
    ('matched_path_difference', 43, 'Pair 3', NULL, NULL, 3, 0, 8, NULL, -0.125,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Joint strict repair rate minus matched B-only strict repair rate'),
    ('matched_path_difference', 44, 'Pair 4', NULL, NULL, 4, 0, 8, NULL, 0.0,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Joint strict repair rate minus matched B-only strict repair rate'),
    ('matched_path_difference', 45, 'Pair 5', NULL, NULL, 5, 0, 8, NULL, -0.25,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Joint strict repair rate minus matched B-only strict repair rate'),
    ('matched_path_difference', 46, 'Pair 6', NULL, NULL, 6, 0, 8, NULL, -0.125,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Joint strict repair rate minus matched B-only strict repair rate'),
    ('matched_path_difference', 47, 'Pair 7', NULL, NULL, 7, 0, 8, NULL, 0.0,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Joint strict repair rate minus matched B-only strict repair rate'),

    ('analysis_correction', 50, 'Joint v1-only successes', 'joint_E_and_B', NULL, NULL, 4, 14, 4.0 / 14.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Preferred-fresh successes that did not meet literal blur reduction'),
    ('analysis_correction', 51, 'B-only v1-only successes', 'B_only_exact_schedule_B_matched_control', NULL, NULL, 13, 30, 13.0 / 30.0, NULL,
     'f608e77afd72bbb6921720eaf92840c519528697ea0dd1712bac8dd147d6a358',
     'Preferred-fresh successes that did not meet literal blur reduction')
),
base_output AS (
SELECT
  report_evidence.*,
  CASE
    WHEN section = 'matched_path_difference' THEN
      CASE pair_index
        WHEN 0 THEN 0 WHEN 1 THEN 0 WHEN 2 THEN 0 WHEN 3 THEN 6
        WHEN 4 THEN 4 WHEN 5 THEN 0 WHEN 6 THEN 0 WHEN 7 THEN 0
      END
  END AS joint_success_count,
  CASE
    WHEN section = 'matched_path_difference' THEN
      CASE pair_index
        WHEN 0 THEN 3 WHEN 1 THEN 0 WHEN 2 THEN 0 WHEN 3 THEN 7
        WHEN 4 THEN 4 WHEN 5 THEN 2 WHEN 6 THEN 1 WHEN 7 THEN 0
      END
  END AS b_only_success_count,
  CASE
    WHEN section = 'matched_path_difference' THEN 8
  END AS comparisons_per_path,
  CASE
    WHEN section IN ('opportunity_path', 'strict_repair', 'matched_path_difference') THEN 0
  END AS opportunity_guard_satisfied,
  source_artifact_id AS result_id
FROM report_evidence
),
branch_evidence(
  section,
  row_order,
  group_name,
  role,
  rollback_step,
  pair_index,
  numerator,
  denominator,
  rate,
  difference,
  source_artifact_id,
  note,
  horizon,
  centrality_order,
  comparator_rate
) AS (
  VALUES
    ('branch_rank', 60, 'h5 · rank1 medoid', 'rank1_medoid', NULL, NULL, 7, 18, 7.0 / 18.0, 7.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc full-rank audit; same-four-scout uniform-random baseline', 5, 1, 0.375),
    ('branch_rank', 61, 'h5 · rank2', 'rank2', NULL, NULL, 7, 18, 7.0 / 18.0, 7.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc full-rank audit; same-four-scout uniform-random baseline', 5, 2, 0.375),
    ('branch_rank', 62, 'h5 · rank3', 'rank3', NULL, NULL, 5, 18, 5.0 / 18.0, 5.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc full-rank audit; same-four-scout uniform-random baseline', 5, 3, 0.375),
    ('branch_rank', 63, 'h5 · rank4 max-outlier', 'rank4_max', NULL, NULL, 8, 18, 8.0 / 18.0, 8.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc reverse-direction clue', 5, 4, 0.375),

    ('branch_rank', 64, 'h10 · rank1 medoid', 'rank1_medoid', NULL, NULL, 5, 18, 5.0 / 18.0, 5.0 / 18.0 - 0.375,
     '00838715cf95869cf4ac788226b05d3a1cf2714f0caa12e17c83b1db7c49e703',
     'Frozen primary medoid rule failed', 10, 1, 0.375),
    ('branch_rank', 65, 'h10 · rank2', 'rank2', NULL, NULL, 7, 18, 7.0 / 18.0, 7.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc full-rank audit', 10, 2, 0.375),
    ('branch_rank', 66, 'h10 · rank3', 'rank3', NULL, NULL, 5, 18, 5.0 / 18.0, 5.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc full-rank audit', 10, 3, 0.375),
    ('branch_rank', 67, 'h10 · rank4 max-outlier', 'rank4_max', NULL, NULL, 10, 18, 10.0 / 18.0, 10.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc reverse-direction clue selected after medoid failure; not confirmatory', 10, 4, 0.375),

    ('branch_rank', 68, 'h20 · rank1 medoid', 'rank1_medoid', NULL, NULL, 5, 18, 5.0 / 18.0, 5.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc full-rank audit; same-four-scout uniform-random baseline', 20, 1, 0.375),
    ('branch_rank', 69, 'h20 · rank2', 'rank2', NULL, NULL, 7, 18, 7.0 / 18.0, 7.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc full-rank audit; same-four-scout uniform-random baseline', 20, 2, 0.375),
    ('branch_rank', 70, 'h20 · rank3', 'rank3', NULL, NULL, 8, 18, 8.0 / 18.0, 8.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc full-rank audit; same-four-scout uniform-random baseline', 20, 3, 0.375),
    ('branch_rank', 71, 'h20 · rank4 max-outlier', 'rank4_max', NULL, NULL, 7, 18, 7.0 / 18.0, 7.0 / 18.0 - 0.375,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Posthoc reverse-direction clue is nearly null at h20', 20, 4, 0.375),

    ('branch_detection', 80, 'higher-O opportunity AUC', 'higher_O', NULL, NULL, 18, 63, 18.0 / 63.0, 18.0 / 63.0 - 0.5,
     '00838715cf95869cf4ac788226b05d3a1cf2714f0caa12e17c83b1db7c49e703',
     'Frozen higher-O detection prediction failed; low-O 0.714 is only its algebraic sign flip', 10, NULL, 0.5),
    ('branch_safety', 81, 'non-opportunity clear-bad', 'max_outlier', NULL, NULL, 2, 14, 2.0 / 14.0, 2.0 / 14.0 - 6.0 / 56.0,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Max-outlier is worse than uniform-random expectation on this safety readout', 10, 4, 6.0 / 56.0),
    ('branch_safety', 82, 'non-opportunity preservation', 'max_outlier', NULL, NULL, 12, 14, 12.0 / 14.0, 12.0 / 14.0 - 49.0 / 56.0,
     '7d0a0094db136c2c6d644733b71b4cd634ad11bf7b9738151771d360c947508a',
     'Max-outlier is worse than uniform-random expectation on this safety readout', 10, 4, 49.0 / 56.0)
),
all_output AS (
  SELECT
    base_output.*,
    NULL AS horizon,
    NULL AS centrality_order,
    NULL AS comparator_rate
  FROM base_output
  UNION ALL
  SELECT
    section,
    row_order,
    group_name,
    role,
    rollback_step,
    pair_index,
    numerator,
    denominator,
    rate,
    difference,
    source_artifact_id,
    note,
    NULL AS joint_success_count,
    NULL AS b_only_success_count,
    NULL AS comparisons_per_path,
    NULL AS opportunity_guard_satisfied,
    source_artifact_id AS result_id,
    horizon,
    centrality_order,
    comparator_rate
  FROM branch_evidence
)
SELECT *
FROM all_output
ORDER BY section, row_order;
