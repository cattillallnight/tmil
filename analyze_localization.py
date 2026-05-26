import csv, statistics

rows = []
with open('results/step12_human_localization_metrics.csv', newline='') as f:
    reader = csv.DictReader(f)
    for r in reader:
        gt = r['human_gt_burst']
        ai_idx = int(r['ai_max_attn_idx'])
        hit = int(r['hit_at_1'])
        iou = float(r['iou'])
        start, end = map(int, gt.split('-'))
        rows.append({'gt_start': start, 'gt_end': end, 'ai_idx': ai_idx, 'hit': hit, 'iou': iou})

n_windows_list = [max(r['ai_idx'], r['gt_end']) + 1 for r in rows]
hits = [r['hit'] for r in rows]
ious = [r['iou'] for r in rows]

random_hit1 = statistics.mean([1.0/n for n in n_windows_list]) * 100

gt_positions = [(r['gt_start'], r['gt_end']) for r in rows]
early = sum(1 for s,e in gt_positions if e <= 10)
mid   = sum(1 for s,e in gt_positions if 10 < s < 100)
late  = sum(1 for s,e in gt_positions if s >= 100)

fail_rows = [r for r in rows if r['hit'] == 0]
succ_rows = [r for r in rows if r['hit'] == 1]
fail_n = [max(r['ai_idx'], r['gt_end']) + 1 for r in fail_rows]
succ_n = [max(r['ai_idx'], r['gt_end']) + 1 for r in succ_rows]

nonzero_iou = [r for r in rows if r['iou'] > 0]

print(f"Total accounts: {len(rows)}")
print(f"Hits: {sum(hits)} ({sum(hits)/len(rows)*100:.2f}%)")
print(f"Mean IoU: {statistics.mean(ious):.4f}")
print()
print("--- N_windows stats ---")
print(f"Mean N_windows: {statistics.mean(n_windows_list):.2f}")
print(f"Median N_windows: {statistics.median(n_windows_list):.2f}")
print(f"Min: {min(n_windows_list)}, Max: {max(n_windows_list)}")
print()
print("--- Random Hit@1 baseline ---")
print(f"E[random Hit@1] = mean(1/N) = {random_hit1:.4f}%")
print()
print("--- GT burst position ---")
print(f"Early (end<=10): {early}")
print(f"Mid (10<start<100): {mid}")
print(f"Late (start>=100): {late}")
print()
print("--- Failure vs Success N_windows ---")
print(f"Failure cases ({len(fail_rows)}): mean N_windows = {statistics.mean(fail_n):.2f}")
print(f"Success cases ({len(succ_rows)}): mean N_windows = {statistics.mean(succ_n):.2f}")
print()
print("--- Non-zero IoU (partial overlap) ---")
print(f"Partial overlap (IoU>0): {len(nonzero_iou)} accounts ({len(nonzero_iou)/len(rows)*100:.1f}%)")
iou_vals = [r['iou'] for r in nonzero_iou]
print(f"Mean IoU among partial: {statistics.mean(iou_vals):.4f}")
