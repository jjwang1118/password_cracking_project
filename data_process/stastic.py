# 計算重複數量
# 不符合規定數量
# 存成 csv
#path,orignal_num,filter_num,removed_num,finished

import os
import subprocess
import multiprocessing
import pandas as pd
import json
from itertools import groupby
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from data_process.path_config import config


def _write_tsv_worker(args: tuple) -> Path | None:
    """
    Worker（在獨立 process 中執行）：
    讀取單一 CSV 檔，將 (account_prefix \t password) 寫入獨立暫存 TSV。
    回傳暫存檔路徑；若發生例外則回傳 None。
    """
    fpath, tmp_file_str = args
    tmp_file = Path(tmp_file_str)
    try:
        with open(tmp_file, 'w', encoding='utf-8') as out:
            for chunk in pd.read_csv(
                fpath, chunksize=50000,
                usecols=['account', 'password'],
                dtype=str, na_filter=False,
                engine='c',
            ):
                prefix = (
                    chunk['account']
                    .str.rsplit('@', n=1).str[0]
                    .str.strip()
                    .str.replace(r'[\t\n\r]', ' ', regex=True)
                )
                pw = (
                    chunk['password']
                    .str.strip()
                    .str.replace(r'[\t\n\r]', ' ', regex=True)
                )
                out.writelines(prefix + '\t' + pw + '\n')
        return tmp_file
    except Exception:
        if tmp_file.exists():
            tmp_file.unlink()
        return None

def cac_not_valid_password_num(path):
    data=pd.read_csv(path)
    for idx,row in data.iterrows():
        if row['finished'] == False:
            print(f"檔案 {row['path']} 尚未完成處理，請先完成處理後再進行統計。")
            return None, None, None, None
    
    total_orign=data['orignal_num'].astype(int).sum()
    total_filter=data['filter_num'].astype(int).sum()
    toal_not_deplicate=data['removed_num'].astype(int).sum()
    
    return total_orign, total_orign-total_filter, total_filter-toal_not_deplicate, toal_not_deplicate


def org_domain_analysis(data_root:str=None, batch_size:int=20):
    if data_root is None:
        cfg = config()
        data_root = cfg['output_path']

    out_dir = Path.home() / "projects" / "password_cracking_project" / "datasets_filtered"
    org_path  = out_dir / "org.json"
    dm_path   = out_dir / "domain.json"

    # 初始化空檔（清掉舊結果）
    for p in [org_path, dm_path]:
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    store_dm  = {}   # domain -> count (本批次暫存)
    store_org = {}   # org    -> count (本批次暫存)
    file_count = 0

    def flush():
        """將本批次計數 merge 進磁碟檔案，然後清空記憶體 dict。"""
        for path, buf in [(org_path, store_org), (dm_path, store_dm)]:
            with open(path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            for k, v in buf.items():
                existing[k] = existing.get(k, 0) + v
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=4)
        store_org.clear()
        store_dm.clear()

    for root, dir, file in os.walk(data_root):
        for fname in file:
            path = os.path.join(root, fname)
            for chunk in pd.read_csv(path, chunksize=10000, usecols=['account']):
                for account in chunk['account']:
                    parts = str(account).rsplit("@", 1)
                    if len(parts) < 2:
                        continue
                    domain_parts = parts[-1].split('.')
                    if len(domain_parts) < 2:
                        continue
                    org, domain = domain_parts[0], domain_parts[1]
                    store_dm[domain]  = store_dm.get(domain, 0) + 1
                    store_org['@' + org]  = store_org.get('@' + org, 0) + 1

            file_count += 1
            if file_count % batch_size == 0:
                flush()

    # 處理最後不足一個 batch 的剩餘資料
    if store_dm or store_org:
        flush()



def len_analysis(root: str = None):
    if root is None:
        cfg = config()
        root = cfg['output_path']

    out_dir = Path.home() / "projects" / "password_cracking_project" / "datasets_filtered"
    out_path = out_dir / "length_distribution.json"

    # key 只有密碼長度（通常 < 100 種），整個 dict 極小，全程放記憶體即可
    length_distribution = {}

    for dirpath, dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(dirpath, fname)
            for chunk in pd.read_csv(path, chunksize=10000, usecols=['password']):
                counts = chunk['password'].astype(str).str.len().value_counts()
                for pw_len, count in counts.items():
                    length_distribution[pw_len] = length_distribution.get(pw_len, 0) + int(count)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(length_distribution, f, ensure_ascii=False, indent=4)


def _shard_key(account: str) -> str:
    """依 account 首字元決定分片檔名 key（a-z / 0-9 / _other）。"""
    if not account:
        return "_other"
    ch = account[0].lower()
    if ch.isalpha() or ch.isdigit():
        return ch
    return "_other"


def _shard_jsonl(src_path: Path, shard_dir: Path, flush_interval: int = 5000):

    shard_dir.mkdir(exist_ok=True)
    handles: dict[str, object] = {}

    def get_handle(key: str):
        if key not in handles:
            handles[key] = open(
                shard_dir / f"{key}.jsonl", "w",
                encoding="utf-8", buffering=1 << 20,
            )
        return handles[key]

    total = 0
    errors = 0

    try:
        with open(src_path, "r", encoding="utf-8") as fin:
            for line in fin:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                key = _shard_key(obj.get("account", ""))
                fout = get_handle(key)
                fout.write(line + "\n")

                total += 1
                if total % flush_interval == 0:
                    fout.flush()
                if total % 1_000_000 == 0:
                    print(f"  [shard] 已處理 {total:,} 行 ...", flush=True)
    finally:
        for fh in handles.values():
            fh.close()

    print(f"  [shard] 完成！共 {total:,} 行，跳過 {errors:,} 行解析錯誤。")
    print(f"  [shard] 分片輸出至: {shard_dir}")
    for p in sorted(shard_dir.glob("*.jsonl")):
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb >= 1024:
            print(f"    {p.name:>12s}  {size_mb / 1024:.2f} GB")
        else:
            print(f"    {p.name:>12s}  {size_mb:.1f} MB")


def pair_sister(data_root: str = None, n_workers: int = 4,
                sort_buffer: str = '512M', shard: bool = False):
    """
    三階段 disk-based 實作，記憶體用量 O(1)：
      1. 並行讀取 CSV → 各自寫成獨立暫存 TSV（ProcessPoolExecutor）
      2. 外部多核排序（sort --parallel）
      3. 串流 groupby → 批次寫出 JSONL
      4.（可選）依 account 首字元分片，避免單檔過大 OOM

    Parameters
    ----------
    n_workers   : 並行 worker 數，預設 CPU 核心數 - 1
    sort_buffer : 傳給 sort --buffer-size，預設 '512M'
    shard       : 若為 True，完成後將 JSONL 按 account 首字元分片至
                  sister_password_shards/ 目錄（a-z, 0-9, _other）
    """
    if data_root is None:
        cfg = config()
        data_root = cfg['output_path']

    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    out_dir     = Path.home() / "projects" / "password_cracking_project" / "datasets_filtered"
    out_path    = out_dir / "sister_password_count.jsonl"
    sorted_path = out_dir / "_tmp_pairs_sorted.tsv"
    tmp_dir     = out_dir / "_tmp_tsv_parts"
    tmp_dir.mkdir(exist_ok=True)

    # ── 第一步：並行讀取多個 CSV，各自寫暫存 TSV ────────────────────
    all_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(data_root)
        for fname in files
    ]

    worker_args = [
        (fpath, str(tmp_dir / f"_part_{i}.tsv"))
        for i, fpath in enumerate(all_files)
    ]

    tmp_files: list[Path] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_write_tsv_worker, worker_args, chunksize=4):
            if result is not None:
                tmp_files.append(result)

    if not tmp_files:
        print("警告：沒有成功處理任何檔案。")
        return

    # ── 第二步：外部多核排序，直接接受多個輸入檔 ────────────────────
    subprocess.run(
        [
            'sort', '-t', '\t', '-k1,1',
            f'--parallel={n_workers}',
            f'--buffer-size={sort_buffer}',
            '-o', str(sorted_path),
        ] + [str(f) for f in tmp_files],
        check=True,
    )

    for f in tmp_files:
        f.unlink()
    tmp_dir.rmdir()

    # ── 第三步：串流 groupby → 批次 writelines → JSONL ──────────────
    _WRITE_BATCH = 2000          # 每累積 2000 行才呼叫一次 writelines

    with open(sorted_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8', buffering=1 << 20) as fout:
        buf: list[str] = []
        for account, group_iter in groupby(
            fin, key=lambda line: line.split('\t', 1)[0]
        ):
            passwords: set[str] = set()
            count = 0
            for line in group_iter:
                parts = line.rstrip('\n').split('\t', 1)
                if len(parts) == 2:
                    passwords.add(parts[1])
                    count += 1
            buf.append(
                json.dumps(
                    {"account": account, "passwords": list(passwords), "count": count},
                    ensure_ascii=False,
                ) + '\n'
            )
            if len(buf) >= _WRITE_BATCH:
                fout.writelines(buf)
                buf.clear()
        if buf:
            fout.writelines(buf)

    sorted_path.unlink()

    # ── 第四步（可選）：依首字元分片，避免單檔 OOM ──────────────────
    if shard:
        print("開始分片 sister_password_count.jsonl ...")
        shard_dir = out_dir / "sister_password_shards"
        _shard_jsonl(out_path, shard_dir)
                    


