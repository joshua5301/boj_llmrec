from openai import OpenAI
import json
import re
import pandas as pd

from ..recommender.recommender import Recommender
from .llm_utils import get_filtered_problems

class LLM:
    def __init__(self, api_key: str, recommender: Recommender = None) -> None:
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.recommender = recommender

        possible_tags = (
            "math, implementation, dp, data_structures, graphs, greedy, string, bruteforcing, " +
            "graph_traversal, sorting, geometry, ad_hoc, number_theory, trees, segtree, binary_search, " +
            "arithmetic, simulation, constructive, bfs, prefix_sum, combinatorics, case_work, dfs, " +
            "shortest_path, bitmask, hash_set, dijkstra, backtracking, tree_set, sweeping, disjoint_set, " +
            "parsing, priority_queue, dp_tree, divide_and_conquer, two_pointer, stack, parametric_search, " +
            "game_theory, flow, primality_test, probability, lazyprop, dp_bitfield, knapsack, recursion"
        )
        possible_tags = [
            "math", "implementation", "dp", "data_structures", "graphs", "greedy", "string", "bruteforcing",
            "graph_traversal", "sorting", "geometry", "ad_hoc", "number_theory", "trees", "segtree", "binary_search",
            "arithmetic", "simulation", "constructive", "bfs", "prefix_sum", "combinatorics", "case_work", "dfs",
            "shortest_path", "bitmask", "hash_set", "dijkstra", "backtracking", "tree_set", "sweeping", "disjoint_set",
            "parsing", "priority_queue", "dp_tree", "divide_and_conquer", "two_pointer", "stack", "parametric_search",
            "game_theory", "flow", "primality_test", "probability", "lazyprop", "dp_bitfield", "knapsack", "recursion"
        ]
        possible_tags_str = ', '.join(possible_tags)
        possible_tags_explained_str = ', '.join([f"{tag}_explained" for tag in possible_tags])
        self.tts_prompt = """
        당신은 LLM의 출력을 음성으로 전환할 때, 출력에서 불필요한 부분을 제거해주는 시스템입니다.
        LLM이 출력한 내용 도중 코드 블럭이나, 링크, 이모티콘이 포함되어 있다면, 이를 제거하세요.
        불필요한 부분을 제외한 나머지 부분은 그대로 유지하세요.
        다음은 LLM의 출력입니다.

        """
        self.title_prompt = """
        당신은 LLM과 유저의 대화 기록을 바탕으로 적절한 세션 제목을 생성하는 시스템입니다.
        짧고 간결하게 대화 내용을 요약하여, 세션의 주제를 나타내는 제목을 생성하세요.
        다음은 대화 기록입니다.

        """
        self.keyword_prompt = ("""
        당신은 LLM과 유저의 대화 기록을 바탕으로 적절한 키워드를 생성하는 시스템입니다.
        키워드는 크게 개념 키워드와 코드 키워드로 나뉩니다.
        만약 LLM이 개념 설명을 하였다면, 그 개념에 대한 키워드를 생성하세요.
        가능한 개념 키워드는 다음과 같습니다:
        """
        + possible_tags_explained_str +
        """
        LLM이 유저의 코드를 분석하거나, 오류를 지적했다면, 그 코드에 대한 키워드를 생성하세요.
        가능한 코드 키워드는 다음과 같습니다:
        time_complexity_over, space_complexity_over, syntax_error, edge_case_error, readability_issue, off_by_one_error
        코드 키워드의 경우, **오직 명확하게 분류가능할 때에만** 키워드를 생성하세요.
        키워드는 쉼표로 구분하여 나열하세요.
        다음은 대화 기록입니다.

        """
        )
        self.prompt = """
        당신은 Baekjoon Online Judge에 특화된 대화형 알고리즘 문제 풀이 도우미입니다.
        유저가 문제를 요청하면, 기계적으로 문제 목록만 나열하지 말고, 대화하며 추천해 주세요.
        또한 당신은 문제의 구체적인 내용은 알지 못하므로, 유저가 이를 묻는다면 알지 못한다고 답변해 주세요.
        만약 tool 호출의 결과가 비어있는 경우, 유저의 핸들이 존재하지 않거나, solved.ac 서버의 문제인 경우가 많습니다.
        이 경우, 유저에게 핸들을 확인해 달라고 요청하세요.

        문제의 난이도는 'Bronze 5'부터 'Ruby 1'까지의 범위로 설정되어 있습니다.
        예시는 다음과 같습니다: 'Bronze 5', 'Silver 2', 'Ruby 2', 'Platinum 1'.
        티어 뒤의 숫자는 1에서 5까지의 숫자로, 5는 해당 분류 내에서 가장 쉬운 문제를 의미합니다.

        문제를 제공할 때는 각 문제마다 아래의 형식을 따라 주세요:

        출력 형식:
        🔹 [{문제 제목} ({문제 번호}번)]({문제 링크}) - {문제 난이도}
        📌 {간단한 설명}

        문제 제목은 **그대로, 정확히** 전달하세요.

        조건:
        - 문제는 2~4개 정도 제공하며, 시각적으로 보기 좋게 이모지를 적절히 활용해 주세요.
        - 문제의 난이도 제한은 사용자의 요구가 있지 않은 한 설정하지 않습니다.
        """
        self.functions = [{
            "type": "function",
            "name": "get_filtered_problems",
            "description": "백준 알고리즘 문제들을 주어진 조건에 맞게 가져옵니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": (
                            "함수 호출의 목적입니다.\n" +
                            "개인화된 문제 추천을 원한다면 'recommend'를 명시하세요.\n" +
                            "유사한 문제를 제공받고 싶다면 'similar'를 명시하세요.\n" +
                            "어떤 유저가 풀었던 문제를 제공받고 싶다면 'user'를 명시하세요.\n"
                            "무조건 'recommend', 'similar', 'user' 중 하나만을 명시해야 합니다."
                        )
                    },
                    "target_problem_id": {
                        "type": "integer",
                        "description": (
                            "type가 'similar'일 때, 유사한 문제를 찾기 위한 기준 문제의 ID입니다.\n" +
                            "예시는 다음과 같습니다: 1000, 1234, 5678.\n"
                        )
                    },
                    "target_user_handle": {
                        "type": "string",
                        "description": (
                            "type가 'user'일 때, 해당 유저가 푼 문제를 가져오기 위한 유저 핸들입니다.\n" +
                            "예시는 다음과 같습니다: '37aster', 'baekjoon', 'user123'.\n"
                        )
                    },
                    "tags": {
                        "type": "string",
                        "description": (
                            "문제 유형에 대한 조건입니다.\n" +
                            "사용 가능한 유형들은 다음과 같습니다: " + possible_tags_str + "\n" +
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
                            "이 값은 0부터 시작하며, 0은 기본을 의미합니다.\n" +
                            "예시는 다음과 같습니다: 0, 1, 2, 3."
                        )
                    }
                },
                "required": ['type'],
                "additionalProperties": False
            }
        }]

    def _get_profile_prompt(self, profile: dict) -> str:
        profile_prompt = "사용자 프로필 정보:\n"
        level_desc = {
            "very low":  "사용자는 프로그래밍 경험이 거의 없으며, 기본 문법 정도만 알고 있습니다.",
            "low":       "사용자는 간단한 입출력·자료형을 다룰 수 있지만 알고리즘 경험이 많지 않습니다.",
            "medium":    "사용자는 정렬·구현·기초 자료구조 문제를 무리 없이 해결할 수 있습니다.",
            "high":      "사용자는 그래프·DP·그리디 등 중급 알고리즘을 습득했고, 중~고난도 문제 경험이 있습니다.",
            "very high": "사용자는 복잡한 알고리즘/자료구조를 능숙히 사용하며, 대회 수준 문제도 해결 가능합니다.",
        }
        if (lvl := profile.get("user_level")) in level_desc:
            profile_prompt += level_desc[lvl] + "\n"
        goal_desc = {
            "coding test": "주요 목표는 취업 코딩 테스트 대비입니다.",
            "contest":     "주요 목표는 알고리즘 대회(ICPC·PS 대회) 준비입니다.",
            "learning":    "주요 목표는 알고리즘 지식 확장 및 실력 향상입니다.",
            "hobby":       "주요 목표는 취미로 문제 풀이를 즐기는 것입니다.",
        }
        if (goal := profile.get("goal")) in goal_desc:
            profile_prompt += goal_desc[goal] + "\n"

        if tags := profile.get("interested_tags"):
            # ex) ['dp', 'graph']
            tag_list = ", ".join(tags)
            profile_prompt += f"사용자는 다음 주제에 특히 흥미가 있습니다: {tag_list}.\n"
        return profile_prompt

    def chat(self, user_input: str, prev_msgs: list, user_handle: str, profile: dict) -> tuple[str, list]:
        keywords = []
        recommended_problems = self.recommender.get_recommended_problems(user_handle)
        if not prev_msgs:
            profile_prompt = self._get_profile_prompt(profile)
            prev_msgs = [{"role": "developer", "content": self.prompt + profile_prompt}]

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
            if args.get('type') == 'recommend':
                args['sorted_problem_info'] = recommended_problems
            elif args.get('type') == 'similar':
                target_problem_id = args.get('target_problem_id')
                similar_problems = self.recommender.get_similar_problems(target_problem_id)
                args['sorted_problem_info'] = similar_problems
            elif args.get('type') == 'user':
                target_user_handle = args.get('target_user_handle')
                user_problems = self.recommender.get_other_user_problems(recommended_problems, user_handle, target_user_handle)
                args['sorted_problem_info'] = user_problems
            else:
                raise ValueError(f"Invalid type: {args.get('type')}. Must be 'recommend', 'similar', or 'user'.")
            
            if tags := args.get('tags'):
                tags = [t.strip() for t in re.split(r'\|\||&&', tags)]
                tags = [t + '_recommended' for t in tags]
                keywords.extend(tags)
            if target_problem_id := args.get('target_problem_id'):
                keywords.append(f"problem_{target_problem_id}")
            if target_user_handle := args.get('target_user_handle'):
                keywords.append(f"user_{target_user_handle}")
            if not keywords:
                keywords.append("none")

            result = get_filtered_problems(**args)
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
        text_output = response.choices[0].message.content
        if not keywords:
            keyword = self.get_chat_keywords(user_input, text_output)
            keywords.extend(keyword)
        speech_output = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": self.tts_prompt + text_output
            }]
        ).choices[0].message.content
        return text_output, speech_output, prev_msgs, keywords
    
    def get_session_title(self, message: str, response: str) -> str:
        prompt = self.title_prompt + f"User: {message}\nAssistant: {response}"
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    
    def get_chat_keywords(self, message: str, response: str) -> list[str]:
        prompt = self.keyword_prompt + f"User: {message}\nAssistant: {response}"
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
        )
        keywords = response.choices[0].message.content.strip()
        keywords = keywords.split(',')
        keywords = [k.strip() for k in keywords]
        return keywords
        