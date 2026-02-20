# pairwise_ranking_randomized.py
# 随机化两两比较的顺序，并加入“气味冷却”避免连续重复嗅闻

import random
import sys
from collections import deque

RECENT_COOLDOWN = 2   # 最近出现的对象在接下来这几次比较中尽量回避
PROBE_TRIES = 8       # 为了避开冷却对象，随机探测索引的重试次数上限

def ask_preference(a: int, b: int) -> bool | None:
    """
    询问用户在 a 和 b 中更偏好哪个。
    返回 True 表示 a 优于 b；False 表示 b 优于 a；None 表示用户想查看进度（输入 s）。
    """
    pair = [a, b]
    random.shuffle(pair)

    while True:
        print(f"\n请选择你更偏好的对象：")
        print(f"[1] {pair[0]}")
        print(f"[2] {pair[1]}")
        choice = input("请输入 1 或 2（s 查看当前排序，q 退出）：").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("已退出。")
            sys.exit(0)
        if choice == "s":
            return None
        if choice == "1":
            winner = pair[0]; break
        if choice == "2":
            winner = pair[1]; break
        print("无效输入，请重新输入。")

    return winner == a

def compare(a: int, b: int, memo: dict[tuple[int,int], bool], show_progress, recent: deque[int]) -> bool:
    """
    带缓存的比较：返回 True 表示 a 优于 b。
    """
    key = (a, b)
    if key in memo:
        return memo[key]
    if (b, a) in memo:
        return not memo[(b, a)]

    while True:
        res = ask_preference(a, b)
        if res is None:
            show_progress()
            continue
        memo[key] = res
        # 记录最近被嗅闻的对象（两个都加上）
        recent.append(a)
        recent.append(b)
        while len(recent) > 2 * RECENT_COOLDOWN:
            recent.popleft()
        return res

def pick_probe_index(lo: int, hi: int, sorted_list: list[int], recent: deque[int]) -> int:
    """
    在 [lo, hi) 内随机选择一个探测索引，尽量避开最近出现的对象。
    """
    if hi - lo <= 1:
        return lo
    candidates = list(range(lo, hi))
    # 多次尝试随机挑一个不在冷却列表里的对象
    for _ in range(PROBE_TRIES):
        mid = random.choice(candidates)
        if sorted_list[mid] not in recent:
            return mid
    # 如果尝试多次仍避不开，就退而求其次：用常规的中点
    return (lo + hi) // 2

def binary_insert_random(sorted_list: list[int], item: int,
                         memo: dict[tuple[int,int], bool], show_progress, recent: deque[int]):
    """
    随机探测的“二分插入”，同时结合冷却机制。
    """
    if not sorted_list:
        sorted_list.append(item)
        return

    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = pick_probe_index(lo, hi, sorted_list, recent)
        if compare(item, sorted_list[mid], memo, show_progress, recent):
            hi = mid
        else:
            lo = mid + 1
    sorted_list.insert(lo, item)

def main():
    print("=== 随机化两两对比偏好排序（对象 1–12）===")
    # 1) 随机化插入顺序
    objects = list(range(1, 13))
    random.shuffle(objects)

    memo: dict[tuple[int,int], bool] = {}
    ranking: list[int] = []
    recent = deque()  # 冷却队列：记录最近出现的对象编号

    def show_progress():
        if not ranking:
            print("\n[进度] 尚未有排序结果。")
        else:
            print("\n[进度] 当前排序（高偏好在前）：")
            print(" > ".join(map(str, ranking)))

    print("提示：输入 s 可查看当前排序，q 可退出程序。")
    print(f"本轮随机插入顺序：{objects}\n")

    total = len(objects)
    for idx, obj in enumerate(objects, 1):
        print(f"—— 插入第 {idx}/{total} 个对象：{obj}")
        binary_insert_random(ranking, obj, memo, show_progress, recent)

    print("\n=== 最终偏好排序（高偏好在前）===")
    print(" > ".join(map(str, ranking)))

if __name__ == "__main__":
    main()
