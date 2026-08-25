from compasscart.parser import MessageParser


def test_parser_marks_override_and_new_constraints():
    parser = MessageParser()

    result = parser.parse("Actually, ignore red. What I need is blue leather.", turn=3)

    assert result.is_override is True
    assert {(item.attribute, item.value) for item in result.constraints} >= {
        ("color", "blue"),
        ("material", "leather"),
    }


def test_parser_uses_expected_attribute_for_clarification():
    parser = MessageParser()

    result = parser.parse(
        "For that, what matters is: 100% cotton.",
        turn=2,
        expected_attribute="material",
    )

    assert [(item.attribute, item.value) for item in result.constraints] == [
        ("material", "cotton")
    ]


def test_parser_detects_no_preference_and_browsing_route():
    parser = MessageParser()

    no_preference = parser.parse(
        "I don't have a preference for color; please use your judgment.",
        turn=2,
        expected_attribute="color",
    )
    browsing = parser.parse("I'm looking for shoes, but I'm still exploring.", turn=1)

    assert no_preference.no_preference_attribute == "color"
    assert browsing.route_hint == "browsing"


def test_parser_detects_buying_requirement_and_budget():
    parser = MessageParser()

    result = parser.parse(
        "I'm looking for running shoes. A key requirement is: under $80.", turn=1
    )

    assert result.route_hint == "buying"
    assert ("budget", "80.00") in {
        (item.attribute, item.value) for item in result.constraints
    }
