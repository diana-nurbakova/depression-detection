
import json, os, sys, statistics
from collections import defaultdict

def load_labels(path):
    labels = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                labels[parts[0]] = int(parts[1])
    return labels

def analyze(data_dir, labels_path):
    labels = load_labels(labels_path)
    n_dep = sum(1 for v in labels.values() if v == 1)
    n_ctrl = sum(1 for v in labels.values() if v == 0)
    
    print("=" * 60)
    print("eRisk 2025 Task 2 Training Data Statistics (v2)")
    print("=" * 60)
    print(f"\nUsers: {len(labels)}")
    print(f"  Depressed: {n_dep} ({100*n_dep/len(labels):.1f}%)")
    print(f"  Control:   {n_ctrl} ({100*n_ctrl/len(labels):.1f}%)")
    print(f"  Ratio:     1:{n_ctrl/max(n_dep,1):.1f}")
    
    # Accumulators
    threads_per_user = []
    comments_per_thread = []
    target_posts_per_thread = []
    other_posts_per_thread = []
    replies_to_target_per_thread = []
    target_text_lengths = []
    target_is_author_count = 0
    total_threads = 0
    unique_other_users_per_thread = []
    deleted_submissions = 0
    empty_target_threads = 0
    
    # Per-class
    class_threads = defaultdict(list)
    class_target_texts = defaultdict(list)
    
    json_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    print(f"JSON files: {len(json_files)}")
    
    errors = 0
    for fi, fname in enumerate(json_files):
        # The filename IS the target subject ID
        target_id = fname.replace('.json', '')
        label = labels.get(target_id, -1)
        
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            try:
                threads = json.load(f)
            except json.JSONDecodeError:
                errors += 1
                continue
        
        if not isinstance(threads, list):
            threads = [threads]
        
        threads_per_user.append(len(threads))
        if label >= 0:
            class_threads[label].append(len(threads))
        
        user_target_text_count = 0
        
        for item in threads:
            total_threads += 1
            sub = item.get('submission', {})
            comments = item.get('comments', [])
            
            sub_author = sub.get('user_id', '')
            sub_id = sub.get('submission_id', '')
            
            # Target user identification:
            # 1. filename = target_id
            # 2. Verify: comments with target=True should have user_id == target_id
            
            # Is target the submission author?
            is_author = (sub_author == target_id)
            if is_author:
                target_is_author_count += 1
            
            if sub_author == 'deleted_user':
                deleted_submissions += 1
            
            # Collect target user's post IDs and text
            target_post_ids = set()
            target_count = 0
            other_count = 0
            
            # Check submission
            if is_author:
                target_post_ids.add(sub_id)
                body = sub.get('body', '') or ''
                if body.strip() and not body.startswith('[deleted') and not body.startswith('[removed'):
                    target_text_lengths.append(len(body.split()))
                    target_count += 1
            
            # Check comments
            for c in comments:
                c_uid = c.get('user_id', '')
                c_id = c.get('comment_id', '')
                c_body = c.get('body', '') or ''
                
                if c_uid == target_id:
                    target_post_ids.add(c_id)
                    if c_body.strip() and not c_body.startswith('[deleted') and not c_body.startswith('[removed'):
                        target_text_lengths.append(len(c_body.split()))
                    target_count += 1
                else:
                    other_count += 1
            
            if target_count == 0:
                empty_target_threads += 1
            
            # Direct replies to target
            reply_count = sum(
                1 for c in comments
                if c.get('parent_id', '') in target_post_ids
                and c.get('user_id', '') != target_id
            )
            
            # Unique other users
            other_users = set()
            for c in comments:
                uid = c.get('user_id', '')
                if uid and uid != target_id and uid != 'deleted_user':
                    other_users.add(uid)
            if sub_author and sub_author != target_id and sub_author != 'deleted_user':
                other_users.add(sub_author)
            
            comments_per_thread.append(len(comments))
            target_posts_per_thread.append(target_count)
            other_posts_per_thread.append(other_count)
            replies_to_target_per_thread.append(reply_count)
            unique_other_users_per_thread.append(len(other_users))
            user_target_text_count += target_count
        
        if label >= 0:
            class_target_texts[label].append(user_target_text_count)
        
        # Progress
        if (fi + 1) % 100 == 0:
            print(f"  Processed {fi+1}/{len(json_files)} users...")
    
    def pstats(values, name):
        if not values:
            print(f"  {name}: no data")
            return
        s = statistics.stdev(values) if len(values) > 1 else 0
        print(f"  {name}: mean={statistics.mean(values):.1f}, "
              f"median={statistics.median(values):.1f}, "
              f"std={s:.1f}, "
              f"min={min(values)}, max={max(values)}")
    
    print(f"\n--- Overall ---")
    print(f"Total threads: {total_threads}")
    print(f"Parse errors: {errors}")
    print(f"Target is submission author: {target_is_author_count}/{total_threads} "
          f"({100*target_is_author_count/max(total_threads,1):.1f}%)")
    print(f"Threads with NO target user text: {empty_target_threads}/{total_threads} "
          f"({100*empty_target_threads/max(total_threads,1):.1f}%)")
    print(f"Deleted submissions: {deleted_submissions}")
    
    print(f"\n--- Per User ---")
    pstats(threads_per_user, "Threads (rounds)")
    
    print(f"\n--- Per Thread ---")
    pstats(comments_per_thread, "Total comments")
    pstats(target_posts_per_thread, "Target user contributions")
    pstats(other_posts_per_thread, "Other user contributions")
    pstats(replies_to_target_per_thread, "Direct replies to target")
    pstats(unique_other_users_per_thread, "Unique other users")
    
    print(f"\n--- Target User Text ---")
    pstats(target_text_lengths, "Words per post")
    print(f"  Total text segments: {len(target_text_lengths)}")
    
    # Distribution of threads per user
    print(f"\n--- Threads-per-user distribution ---")
    buckets = [(1,5), (6,10), (11,25), (26,50), (51,100), (101,250), (251,500), (501,1000), (1001,9999)]
    for lo, hi in buckets:
        count = sum(1 for t in threads_per_user if lo <= t <= hi)
        if count > 0:
            print(f"  {lo}-{hi}: {count} users ({100*count/len(threads_per_user):.1f}%)")
    
    print(f"\n--- Per Class ---")
    for lbl, name in [(1, "Depressed"), (0, "Control")]:
        t = class_threads.get(lbl, [])
        txt = class_target_texts.get(lbl, [])
        n = len(t)
        print(f"\n  {name} ({n} users):")
        if t:
            pstats(t, "Threads (rounds)")
            pstats(txt, "Total target texts per user")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python dataset_stats.py <data_dir> <labels_file>")
        print("  data_dir:    path to all_combined/ folder")
        print("  labels_file: path to shuffled_ground_truth_labels.txt")
        sys.exit(1)
    
    all_combined_dir = "data\eRisk-2025\eRisk25-datasets\t2-early-contextualized-depression\final-eriskt2-dataset-with-ground-truth\final-eriskt2-dataset-with-ground-truth\all_combined"
    shuffled_labels_path = "data\eRisk-2025\eRisk25-datasets\t2-early-contextualized-depression\final-eriskt2-dataset-with-ground-truth\final-eriskt2-dataset-with-ground-truth\shuffled_ground_truth_labels.txt"
    
    analyze(sys.argv[1], sys.argv[2])
    #analyze_dataset(all_combined_dir, shuffled_labels_path)
    