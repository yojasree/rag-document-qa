"""
evaluate.py - Tests the pipeline's accuracy against known Q&A pairs.
Run: python src/evaluate.py
"""
from qa_chain import answer_question

TEST_SET = [
    {
        "question": "What is this document about?",
        "expected_keywords": ["topic", "subject"],
    },
]

def simple_keyword_score(answer, expected_keywords):
    answer_lower = answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matches / len(expected_keywords) if expected_keywords else 0.0

def run_evaluation():
    print(f"Running evaluation on {len(TEST_SET)} test case(s)...\n")
    total_score = 0.0
    for i, test_case in enumerate(TEST_SET, start=1):
        question = test_case["question"]
        expected_keywords = test_case["expected_keywords"]
        answer, sources = answer_question(question)
        score = simple_keyword_score(answer, expected_keywords)
        total_score += score
        print(f"Test {i}: {question}")
        print(f"  Generated answer: {answer[:150]}...")
        print(f"  Score: {score:.0%}\n")
    overall_accuracy = (total_score / len(TEST_SET)) * 100 if TEST_SET else 0
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    return overall_accuracy

if __name__ == "__main__":
    run_evaluation()
