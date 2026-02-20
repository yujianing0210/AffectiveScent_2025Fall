"""
基于排序插入法的气味完全排序脚本（鲁棒版本）。

核心思路：
1) 维持一个"已排序"的列表，初始为空
2) 逐个将未排序的气味插入已排序列表
3) 对于每个待插入的气味，通过二分查找定位其正确位置
4) 在定位过程中，仅进行"待插入气味 vs 中点气味"的比较
5) 比较结果直接决定二分方向，无需传递推导
6) 最终得到完全排序，无任何矛盾可能
"""

from __future__ import annotations

import os
import random
import sys
from collections import deque
from pathlib import Path

# 可调参数
SCENT_COUNT = 10
GROUP_SIZE = 2
DEFAULT_TARGET_SECONDS = 60
RECENT_COOLDOWN = 2  # 冷却轮数：最近这些轮出现过的气味尽量回避
PROBE_TRIES = 8      # 随机探测重试次数


def build_scent_list(total: int) -> list[str]:
    """生成气味名称列表。"""
    return [f"Scent-{i+1}" for i in range(total)]


class RobustRankingEngine:
    """
    鲁棒的排序引擎：基于排序插入法。
    - 维持一个已排序列表
    - 逐个插入未排序气味
    - 每次插入通过二分查找定位位置
    - 保证绝不产生矛盾
    """
    
    def __init__(self, scent_count: int):
        self.n = scent_count
        self.ranked = []           # 已排序列表（从高到低）
        self.unranked = set(range(scent_count))  # 未排序的气味
        self.memo = {}             # 比较缓存：(i, j) -> True 表示 i > j
        self.recent = deque()      # 冷却队列
        self.round_count = 0       # 总比较轮数
    
    def compare(self, a: int, b: int) -> bool:
        """
        Ask user which of a and b ranks higher.
        Returns True if a > b, False if b > a.
        """
        # Check cache
        if (a, b) in self.memo:
            return self.memo[(a, b)]
        if (b, a) in self.memo:
            return not self.memo[(b, a)]
        
        self.round_count += 1
        pair = [a, b]
        random.shuffle(pair)
        scents = build_scent_list(self.n)
        
        while True:
            print(f"\n—— Round {self.round_count} ——")
            print("\nPlease select the scent with higher ranking:")
            for idx, scent_id in enumerate(pair, 1):
                print(f"[{idx}] {scents[scent_id]}")
            
            choice = input("Enter number (s to view progress, q to quit): ").strip().lower()
            
            if choice in ("q", "quit", "exit"):
                print("Exiting.")
                sys.exit(0)
            
            if choice == "s":
                self.show_progress()
                continue
            
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(pair):
                    winner = pair[idx - 1]
                    result = (winner == a)
                    self.memo[(a, b)] = result
                    
                    # Update cooldown queue
                    self.recent.append(a)
                    self.recent.append(b)
                    while len(self.recent) > 2 * RECENT_COOLDOWN:
                        self.recent.popleft()
                    
                    return result
            
            print("Invalid input, please try again.")
    
    def pick_probe_position(self, lo: int, hi: int) -> int:
        """
        在 [lo, hi) 范围内选择探测位置，尽量避开最近出现的气味。
        """
        if hi - lo <= 1:
            return lo
        
        candidates = list(range(lo, hi))
        
        # 多次尝试找一个不在冷却列表的位置
        for _ in range(PROBE_TRIES):
            mid = random.choice(candidates)
            if self.ranked[mid] not in self.recent:
                return mid
        
        # 如果尝试多次仍避不开，返回中点
        return (lo + hi) // 2
    
    def insert_scent(self, scent_id: int):
        """
        Insert a scent into the ranked list using binary search.
        Uses random probing + cooldown mechanism.
        """
        scents = build_scent_list(self.n)
        
        # Display progress before insertion
        print(f"\n{'='*50}")
        print(f"【Inserting {self.n - len(self.unranked) + 1}/{self.n}】Ranking {scents[scent_id]}...")
        print(f"{'='*50}")
        print(f"Ranked scents: {len(self.ranked)}/{self.n}")
        if self.ranked:
            print(f"Current ranking: {' > '.join([scents[i] for i in self.ranked])}")
        print()
        
        if not self.ranked:
            self.ranked.append(scent_id)
            return
        
        lo, hi = 0, len(self.ranked)
        
        while lo < hi:
            # Random probe position (avoid cooldown objects)
            mid = self.pick_probe_position(lo, hi)
            
            # Compare: scent_id vs ranked[mid]
            if self.compare(scent_id, self.ranked[mid]):
                # scent_id is higher, insert before mid
                hi = mid
            else:
                # ranked[mid] is higher, insert after mid
                lo = mid + 1
        
        self.ranked.insert(lo, scent_id)
        
        # Display updated ranking after insertion
        print(f"\n✓ {scents[scent_id]} inserted, current ranking: {' > '.join([scents[i] for i in self.ranked])}")
    
    def show_progress(self):
        """Display current ranking progress."""
        scents = build_scent_list(self.n)
        print("\n[Progress] Ranked scents (high to low):")
        for pos, scent_id in enumerate(self.ranked, 1):
            print(f"  {pos}. {scents[scent_id]}")
        
        if self.unranked:
            print(f"\nUnranked scents: {[scents[i] for i in sorted(self.unranked)]}")
        
        pct = 100 * len(self.ranked) / self.n if self.n else 0
        print(f"\nRanked: {len(self.ranked)}/{self.n} ({pct:.1f}%)")
        print(f"Total comparison rounds: {self.round_count}")
    
    def run_ranking(self):
        """
        Execute complete ranking process.
        """
        # Randomize insertion order
        order = list(self.unranked)
        random.shuffle(order)
        
        print("=" * 50)
        print("=== Robust Scent Ranking System (Sorting Insertion) ===")
        print("=" * 50)
        print(f"Total scents: {self.n}")
        print(f"Random insertion order: {[f'Scent-{i+1}' for i in order]}\n")
        print("Tips: Enter 's' to view progress, 'q' to quit.\n")
        
        for scent_id in order:
            self.insert_scent(scent_id)
            self.unranked.remove(scent_id)
        
        print("\n" + "=" * 50)
        print("✓ Ranking complete!")
        print("=" * 50)
    
    def get_final_ranking(self) -> list[int]:
        """Return final ranking (high to low)."""
        return self.ranked[:]


def load_api_key_from_env() -> str | None:
    """From environment variables or .env file read OPENAI_API_KEY."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return None

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return key
    except ImportError:
        pass

    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("OPENAI_API_KEY"):
                    _, val = line.split("=", 1)
                    val = val.strip().strip('"').strip("'")
                    if val:
                        os.environ["OPENAI_API_KEY"] = val
                        return val
    except OSError:
        return None

    return os.getenv("OPENAI_API_KEY")


def generate_sequence_with_gpt(
    ranking: list[int],
    scents: list[str],
    user_brief: str = "",
    target_seconds: int = 55,
):
    """Call OpenAI to generate scent sequence for racing track (English version)."""
    api_key = load_api_key_from_env()
    if not api_key:
        print("\n[Tip] OPENAI_API_KEY not detected. Please set it in .env file or export it first.")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("\n[Tip] openai package not installed. Run: pip install openai")
        return

    client = OpenAI(api_key=api_key)

    # Build scent details
    scent_details = {
        'Scent-1': 'Iso E Super (high volatility, strong stimulation)',
        'Scent-2': 'Ambroxan (low volatility, warm base)',
        'Scent-3': 'Ocean (medium volatility, fresh sensation)',
        'Scent-4': 'Lily (medium-high volatility, floral)',
        'Scent-5': 'Hedione (medium volatility, green sensation)',
        'Scent-6': 'Coriander (medium-high volatility, spicy)',
        'Scent-7': 'Cypress (medium volatility, woody)',
        'Scent-8': 'Bergamot (high volatility, citrus brightness)',
        'Scent-9': 'Orange (high volatility, citrus vitality)',
        'Scent-10': 'Grapefruit (high volatility, citrus impact)'
    }

    # Build ranking text (high to low)
    lines = []
    for pos, idx in enumerate(ranking, 1):
        scent_name = scents[idx]
        detail = scent_details.get(scent_name, '')
        lines.append(f"{pos}. {scent_name} - {detail}")
    ranking_text = "\n".join(lines)

    # Track timeline
    track_timeline = """
Track Timeline (55-second lap):
00:00 - 00:05	Acceleration Straight: Obvious acceleration phase
00:05 - 00:08	Right Turn Entry: Braking, turning right
00:08 - 00:13	Continuous S-Curves (R-L-R): Narrow and winding, frequent directional changes
00:13 - 00:16	Exit Curve/Short Straight: Accelerating toward tent area
00:16 - 00:20	Long Right Curve: Extended arc, maintaining high cornering speed
00:20 - 00:23	Exit Curve/Short Straight: Light acceleration transition
00:23 - 00:26	Sharp Right Turn: Clear deceleration, large angle right turn
00:26 - 00:32	Long Straight: Most obvious acceleration, longest duration
00:32 - 00:36	Sharp Left Hairpin (U-Turn): Heavy braking, lowest speed, left turnaround
00:36 - 00:41	Exit Curve/Long Straight: Immediate acceleration toward tent area
00:41 - 00:44	Right Turn Entry: High speed entry into medium-arc right turn
00:44 - 00:47	Sharp Left Turn: Quick left turn
00:47 - 00:51	Exit Curve/Gentle Right: Gentle right curve transition
00:51 - 00:54	Final Straight Sprint: Full acceleration crossing finish
00:54 - 00:55	Crossing Line: Completing lap successfully
"""

    prompt = f"""
You are a professional fragrance consultant and racing experience design expert. Your task is to design a 55-second scent sequence for a professional racing driver to support focus, motivation, and controlled arousal on the track.

【USER-SPECIFIC SCENT RANKING】(Based on arousal effect, high to low):
{ranking_text}

【VOLATILITY GUIDELINES】(Professional perfumer recommendations):
- Low volatility materials: Establish stable ambient base (Ambroxan, Cypress)
- Medium volatility materials: Create perceptual transitions and coherence (Hedione, Ocean, Coriander)
- High volatility materials: Generate noticeable interventions and impact (Iso E Super, Bergamot, Orange, Grapefruit)

【DESIGN PRINCIPLES】:
1. Volatility Stratification: Progressive strategy from low → medium → high
2. Base Scaffold: Maintain stable low-volatility base throughout
3. Smooth Transitions: Use crossfades, avoid abrupt switches
4. High-Volatility Bursts: Use only as short interventions (3-5 seconds) for perceptual impact
5. Coherence Priority: Goal is perceptual continuity, not chaotic shocks

【FIVE-PHASE SEQUENCE STRUCTURE】:
- Phase 1 (Ambient Base): Low volatility, establish stable foundation (8-10 seconds)
- Phase 2 (Airy Lift): Medium-high volatility, enhance perception (8-10 seconds)
- Phase 3 (Luminous Expansion): Medium volatility, expand sensory range (12-15 seconds)
- Phase 4 (Centered Confidence): Medium-low volatility, restore control sense (12-15 seconds)
- Phase 5 (Discrete High-Volatility Interventions): 1-2 short high-volatility bursts (3-5 seconds each)

【TRACK TIMELINE & DRIVING DYNAMICS】:
{track_timeline}

【DESIGN OBJECTIVES】:
Design a scent sequence for this 55-second lap that:
1. Reflects the user's arousal ranking (high-ranked scents for critical moments)
2. Syncs with track dynamics (acceleration → high volatility, deceleration → low volatility)
3. Maintains controlled arousal throughout (neither overstimulation nor drowsiness)
4. Enhances focus, motivation, and driving confidence

【OUTPUT FORMAT】:
1. Sequence: Use `scent X x Ys` format (X=scent number, Y=seconds), connected with ` + `
   Example: scent 2 x 8s + scent 5 x 10s + scent 1 x 5s + ...
   
2. Design Reasoning (English, 200-300 words):
   - Why this order was chosen
   - How it corresponds with track phases
   - How volatility layering supports driving experience
   - When and why high-volatility interventions are used

【CRITICAL CONSTRAINTS】:
- Total duration must be exactly 55 seconds
- Each scent duration range: 3-12 seconds
- Maximum 1-2 high-volatility bursts (each 3-5 seconds)
- Avoid abrupt switches, prioritize smooth transitions
- Output only sequence and reasoning, no extra explanation

Begin design:
"""
    if user_brief:
        prompt += f"\n【USER ADDITIONAL PREFERENCES】: {user_brief}\n"

    print("\n[GPT] Designing racing track scent sequence...")
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional fragrance consultant, perfumer, and racing experience designer. You must generate a scientifically sound, coherent, and efficient scent sequence that strictly adheres to time constraints, volatility principles, and track dynamics."
                },
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=0.8,
            max_tokens=600,
        )
        result = resp.choices[0].message.content.strip()
        
        print("\n" + "="*70)
        print("✨ RACING TRACK SCENT SEQUENCE GENERATED BY GPT")
        print("="*70)
        print(result)
        print("="*70)
        
    except Exception as exc:
        print(f"\n[Error] Failed to call GPT: {exc}")


def main():
    if SCENT_COUNT < 2:
        print("Parameter error: SCENT_COUNT must be >= 2.")
        sys.exit(1)

    scents = build_scent_list(SCENT_COUNT)
    engine = RobustRankingEngine(SCENT_COUNT)

    # Execute ranking
    engine.run_ranking()

    # Display final result
    ranking = engine.get_final_ranking()
    
    print("\n=== FINAL COMPLETE RANKING (High to Low) ===")
    for pos, idx in enumerate(ranking, 1):
        print(f"{pos:2d}. {scents[idx]}")
    
    print(f"\nTotal comparison rounds: {engine.round_count}")
    print(f"Theoretical minimum rounds: {SCENT_COUNT - 1}")
    print(f"Theoretical maximum rounds: ~{SCENT_COUNT * 4} (worst case log insertion)")

    # Optional: Call GPT to generate scent sequence
    try_generate = input("\nGenerate racing track scent sequence with GPT? (y/n): ").strip().lower()
    if try_generate == "y":
        brief = input("Optional: Enter additional preferences for track experience (e.g., 'enhance mid-section sprint feeling'): ").strip()
        generate_sequence_with_gpt(ranking, scents, brief, target_seconds=55)


if __name__ == "__main__":
    main()
