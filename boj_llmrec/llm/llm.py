from openai import OpenAI
import json
import pandas as pd

from .utils import get_recommended_problems, level_to_tier, tier_to_level

class LLM:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.prompt = """
        당신은 Baekjoon Online Judge의 알고리즘 문제를 추천해주는 친절한 대화형 추천 시스템입니다.
        유저가 문제를 요청하면, 기계적으로 문제 목록만 나열하지 말고, 대화하며 추천해 주세요.
        또한 당신은 문제의 구체적인 내용은 알지 못하므로, 유저가 이를 묻는다면 알지 못한다고 답변해 주세요.

        문제의 난이도는 'Bronze 5'부터 'Ruby 1'까지의 범위로 설정되어 있습니다.
        예시는 다음과 같습니다: 'Bronze 5', 'Silver 2', 'Ruby 2', 'Platinum 1'.
        티어 뒤의 숫자는 1에서 5까지의 숫자로, 5는 해당 분류 내에서 가장 쉬운 문제를 의미합니다.

        추천할 때는 각 문제마다 아래의 형식을 따라 주세요:

        출력 형식:
        🔹 [{문제 제목} ({문제 번호}번)]({문제 링크}) - {문제 난이도}
        📌 {간단한 설명}

        문제 제목은 **그대로, 정확히** 전달하세요.

        조건:
        - 문제는 2~4개 정도 추천하며, 시각적으로 보기 좋게 이모지를 적절히 활용해 주세요.
        - 문제의 난이도 제한은 사용자의 요구가 있지 않은 한 설정하지 않습니다.
        """
        possible_tags = (
            "math, implementation, dp, data_structures, graphs, greedy, string, bruteforcing, " +
            "graph_traversal, sorting, geometry, ad_hoc, number_theory, trees, segtree, binary_search, " +
            "arithmetic, simulation, constructive, bfs, prefix_sum, combinatorics, case_work, dfs, " +
            "shortest_path, bitmask, hash_set, dijkstra, backtracking, tree_set, sweeping, disjoint_set, " +
            "parsing, priority_queue, dp_tree, divide_and_conquer, two_pointer, stack, parametric_search, " +
            "game_theory, flow, primality_test, probability, lazyprop, dp_bitfield, knapsack, recursion"
        )
        self.functions = [{
            "type": "function",
            "name": "get_recommended_problems",
            "description": "개인화된 백준 알고리즘 문제들을 주어진 조건에 맞게 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "string",
                        "description": (
                            "문제 유형에 대한 조건입니다.\n" +
                            "사용 가능한 유형들은 다음과 같습니다: " + possible_tags + "\n" +
                            "유형은 &&(AND) 연산자나 ||(OR) 연산자로 묶을 수 있습니다.\n" +
                            "예시는 다음과 같습니다: 'dp && segtree', 'implementation || greedy', 'math && geometry'"
                        )
                    },
                    "max_difficulty": {
                        "type": "string",
                        "description": (
                            "문제의 최대 난이도입니다.\n" +
                            "유저의 요구가 있지 않은 이상, 이 값은 명시하지 마세요.\n" +
                            "예시는 다음과 같습니다: 'Bronze 5', 'Silver 2', 'Ruby 1', 'Platinum 3'."
                        )
                    },
                    "min_difficulty": {
                        "type": "string",
                        "description": (
                            "문제의 최소 난이도입니다.\n" +
                            "유저의 요구가 있지 않은 이상, 이 값은 명시하지 마세요.\n" +
                            "예시는 다음과 같습니다: 'Silver 4', 'Gold 5', 'Platinum 2', 'Platinum 5'."
                        )
                    },
                    "alternative": {
                        "type": "integer",
                        "description": (
                            "동일한 조건 하에 다른 문제를 받고 싶다면, 이 값을 명시하세요.\n" +
                            "이 값은 0부터 시작하며, 0은 기본 추천 문제를 의미합니다.\n" +
                            "예시는 다음과 같습니다: 0, 1, 2, 3."
                        )
                    }
                },
                "required": [],
                "additionalProperties": False
            }
        }]

    def chat(self, user_input: str, prev_msgs: list, sorted_problem_info: pd.DataFrame) -> tuple[str, list]:
        if not prev_msgs:
            prev_msgs = [{"role": "developer", "content": self.prompt}]

        prev_msgs.append({
            "role": "user",
            "content": user_input
        })
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=prev_msgs,
            functions=self.functions,
            function_call="auto",
        )
        if response.choices[0].message.function_call:
            args = json.loads(response.choices[0].message.function_call.arguments)
            print(f"Tags: {args.get('tags', 'None')}, Min: {args.get('min_difficulty', 'None')}, Max: {args.get('max_difficulty', 'None')}")
            print(f"Alternative: {args.get('alternative', 'None')}")
            args['sorted_problem_info'] = sorted_problem_info
            result = get_recommended_problems(**args)
            prev_msgs.append(response.choices[0].message)
            prev_msgs.append({
                "role": "function",
                "name": response.choices[0].message.function_call.name,
                "content": json.dumps(result),
            })
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=prev_msgs
            )
        prev_msgs.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        return response.choices[0].message.content, prev_msgs