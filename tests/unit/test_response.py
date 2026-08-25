from compasscart.models import Candidate, QuestionDecision
from compasscart.response import ResponseBuilder


def test_response_explains_relaxed_recommendations_without_changing_contract_keys():
    response = ResponseBuilder({"A"}, ["A"]).build(
        [Candidate("A", relaxed=True, violations=("budget<=80",))],
        QuestionDecision(None),
        top_k=1,
    )

    assert set(response) == {"message", "ask_attribute", "recommendations", "usage"}
    assert "close alternatives" in response["message"].lower()
    assert "relaxing budget<=80" in response["message"].lower()


def test_response_accepts_explicit_relaxed_constraint_context():
    response = ResponseBuilder({"A"}, ["A"]).build(
        [Candidate("A")],
        QuestionDecision(None),
        top_k=1,
        relaxed=True,
        relaxed_constraints=("material=leather",),
    )

    assert "relaxing material=leather" in response["message"].lower()
