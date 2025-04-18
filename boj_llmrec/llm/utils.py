from pyparsing import Word, alphas, infixNotation, opAssoc, Literal
import pandas as pd

def level_to_tier(level: int) -> str:
    level_map = {
        0: "Bronze",
        1: "Silver",
        2: "Gold",
        3: "Platinum",
        4: "Diamond",
        5: "Ruby",
    }
    level -= 1
    tier = level_map.get(level // 5, "Unknown")
    tier += f" {5 - level % 5}"
    return tier

def tier_to_level(tier: str) -> int:
    level_map = {
        "Bronze": 0,
        "Silver": 1,
        "Gold": 2,
        "Platinum": 3,
        "Diamond": 4,
        "Ruby": 5,
    }
    return level_map[tier[:-2]] * 5 + (5 - int(tier[-1])) + 1

var = Word(alphas + '_')
expr = infixNotation(
    var,
    [(Literal("&&"), 2, opAssoc.LEFT),
     (Literal("||"), 2, opAssoc.LEFT)]
)

def get_recommended_problems(sorted_problem_info: pd.DataFrame,
                             topk: int = 10,
                             tags: str = "",
                             max_difficulty: str = "",
                             min_difficulty: str = "",
                             alternative: int = 0,
                             ) -> str:
    def tag_mask(tag: str) -> pd.Series:
        return sorted_problem_info["tags"].str.contains(tag, regex=False).convert_dtypes().fillna(False)

    def evaluate(cond) -> pd.Series:
        if isinstance(cond, str):
            return tag_mask(cond)
        if not isinstance(cond, list):
            cond = list(cond)
        if not cond:
            raise ValueError("Empty sub-expression detected")

        res = evaluate(cond[0])
        i = 1
        while i < len(cond):
            op, right_raw = cond[i], cond[i + 1]
            right = evaluate(right_raw)

            if op == "&&":
                res = res & right
            elif op == "||":
                res = res | right
            else:
                raise ValueError(f"Unknown operator: {op}")
            i += 2
        return res

    if not tags:
        mask = pd.Series(True, index=sorted_problem_info.index)
    else:
        parsed = expr.parseString(tags, parseAll=True).asList()[0]
        mask = evaluate(parsed)

    min_level = tier_to_level(min_difficulty) if min_difficulty else 0
    max_level = tier_to_level(max_difficulty) if max_difficulty else 1000
    sorted_problem_info = sorted_problem_info[
        (sorted_problem_info["level"] >= min_level) &
        (sorted_problem_info["level"] <= max_level)
    ]
    mask = mask.reindex(sorted_problem_info.index, fill_value=False)
    filtered = sorted_problem_info[mask].iloc[topk * alternative: topk * (alternative + 1)]
    return "\n".join(
        f"ID: {row.problemId}, Title: {row.titleKo}, Tags: {row.tags}, Difficulty: {level_to_tier(row.level)}"
        for _, row in filtered.iterrows()
    )

if __name__ == "__main__":
    condition = "dp || greedy || math"

    sorted_problem_info = pd.DataFrame({
        "problemId": [1, 2, 3],
        "titleKo":   ["문제1", "문제2", "문제3"],
        "tags":      ["dp", "greedy", "math"],
    })

    topk = 2
    result = get_recommended_problems(condition, sorted_problem_info, topk)
    print(result)
