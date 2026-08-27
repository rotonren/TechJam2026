import pytest

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
    assert no_preference.route_hint is None
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


def test_override_is_not_parsed_as_previous_question_answer():
    result = MessageParser().parse(
        "Actually, ignore my earlier preference. What I need is: leather.",
        turn=3,
        expected_attribute="category",
    )

    assert result.is_override is True
    assert result.replace_preferences is True
    assert [(item.attribute, item.value) for item in result.constraints] == [
        ("material", "leather")
    ]


def test_attribute_correction_does_not_replace_unmentioned_preferences():
    result = MessageParser().parse("Actually, make it blue.", turn=2)

    assert result.is_override is True
    assert result.replace_preferences is False


def test_parser_assigns_budget_comparison_operators():
    parser = MessageParser()

    under = parser.parse("Shoes under $80", turn=1).constraints[0]
    over = parser.parse("Shoes over $80", turn=1).constraints[0]
    between = parser.parse("Shoes between $50 and $80", turn=1).constraints[0]

    assert (under.operator, under.value) == ("lte", "80.00")
    assert (over.operator, over.value) == ("gte", "80.00")
    assert (between.operator, between.value, between.upper_value) == (
        "between",
        "50.00",
        "80.00",
    )


def test_parser_keeps_explicit_or_as_one_in_constraint():
    result = MessageParser().parse("I need black or blue shoes", turn=1)

    colors = [item for item in result.constraints if item.attribute == "color"]

    assert len(colors) == 1
    assert (colors[0].operator, colors[0].value, colors[0].alternatives) == (
        "in",
        "black",
        ("black", "blue"),
    )


def test_parser_marks_negative_known_values_as_not_in():
    result = MessageParser().parse("I need shoes without leather", turn=1)

    assert (
        "material",
        "leather",
        "not_in",
        ("leather",),
    ) in {
        (item.attribute, item.value, item.operator, item.alternatives)
        for item in result.constraints
    }


def test_pending_attribute_does_not_capture_an_explicit_negative_preference():
    result = MessageParser().parse(
        "I don't want leather", turn=2, expected_attribute="use_case"
    )

    assert ("material", "leather", "not_in") in {
        (item.attribute, item.value, item.operator) for item in result.constraints
    }
    assert all(item.attribute != "use_case" for item in result.constraints)


def test_parser_combines_negative_or_values_into_one_not_in_constraint():
    result = MessageParser().parse("I don't want black or blue shoes", turn=1)

    colors = [item for item in result.constraints if item.attribute == "color"]
    assert len(colors) == 1
    assert (colors[0].operator, colors[0].alternatives) == (
        "not_in",
        ("black", "blue"),
    )


def test_parser_does_not_treat_excluded_catalog_terms_as_categories():
    parser = MessageParser({"category": ("Leather", "Shoes")})

    result = parser.parse("I do not want leather shoes", turn=1)

    assert ("material", "leather", "not_in") in {
        (item.attribute, item.value, item.operator) for item in result.constraints
    }
    assert ("category", "leather") not in {
        (item.attribute, item.value) for item in result.constraints
    }


def test_parser_does_not_turn_a_negated_object_into_a_category():
    result = MessageParser().parse("I do not want leather", turn=1)

    assert ("material", "leather", "not_in") in {
        (item.attribute, item.value, item.operator) for item in result.constraints
    }
    assert ("category", "leather") not in {
        (item.attribute, item.value) for item in result.constraints
    }


def test_parser_uses_catalog_vocabulary_for_brand_size_and_category():
    parser = MessageParser(
        {
            "brand": ("Acme Fashion",),
            "size": ("Medium",),
            "category": ("Dresses",),
        }
    )

    result = parser.parse("I want Acme's Fashion dresses in medium", turn=1)

    assert {(item.attribute, item.value) for item in result.constraints} == {
        ("brand", "acme fashion"),
        ("size", "medium"),
        ("category", "dresses"),
    }


def test_parser_recognizes_changed_mind_and_continuation_language():
    changed = MessageParser().parse("I've changed my mind: blue shoes", turn=2)
    continuation = MessageParser().parse("Show me more options", turn=2)

    assert changed.is_override is True
    assert continuation.is_continuation is True


def test_continuation_command_is_not_assigned_to_pending_attribute():
    result = MessageParser({"feature": ("breathable",)}).parse(
        "show me more", turn=2, expected_attribute="feature"
    )

    assert result.is_continuation is True
    assert result.constraints == ()


def test_semantic_fixed_term_is_not_inferred_as_brand_without_brand_cue():
    parser = MessageParser(
        {"brand": ("Waterproof",), "category": ("Jackets",)}
    )

    result = parser.parse("I need a waterproof jacket", turn=1)
    pairs = {(item.attribute, item.value) for item in result.constraints}

    assert ("feature", "waterproof") in pairs
    assert ("category", "jackets") in pairs
    assert ("brand", "waterproof") not in pairs


def test_budget_phrase_is_not_inferred_as_catalog_category():
    parser = MessageParser({"category": ("Shoes", "Over 50")})

    result = parser.parse("I need shoes over $50", turn=1)
    pairs = {(item.attribute, item.value) for item in result.constraints}

    assert ("category", "shoes") in pairs
    assert ("category", "over 50") not in pairs
    assert ("budget", "50.00") in pairs


def test_pending_question_gates_fixed_aliases_to_the_requested_attribute():
    result = MessageParser().parse(
        "Blue leather, please.", turn=2, expected_attribute="color"
    )

    assert {(item.attribute, item.value) for item in result.constraints} == {
        ("color", "blue"),
    }


def test_pending_question_suppresses_fixed_aliases_for_another_attribute():
    parser = MessageParser(
        {"category": ("Shoes",), "use_case": ("Running", "Work")}
    )

    result = parser.parse("Blue.", turn=2, expected_attribute="use_case")

    assert result.constraints == ()


def test_pending_question_suppresses_fixed_and_dynamic_aliases_for_other_attributes():
    parser = MessageParser({"category": ("Shoes",)})

    result = parser.parse("blue shoes", turn=2, expected_attribute="material")

    assert result.constraints == ()


def test_pending_style_accepts_matching_dynamic_alias_as_hard_clarification():
    parser = MessageParser(
        {
            "style": ("Adjustable",),
            "size": ("Adjustable",),
            "feature": ("Adjustable",),
        }
    )

    result = parser.parse("Adjustable", turn=2, expected_attribute="style")

    assert [
        (item.attribute, item.value, item.is_hard, item.source)
        for item in result.constraints
    ] == [("style", "adjustable", True, "clarification")]


def test_pending_size_suppresses_category_alias_without_soft_fallback():
    parser = MessageParser({"category": ("Boots",)})

    result = parser.parse("Boots", turn=2, expected_attribute="size")

    assert result.constraints == ()


def test_pending_size_does_not_treat_a_trailing_category_word_as_an_explicit_cue():
    parser = MessageParser({"category": ("Boots",)})

    result = parser.parse("Boots category", turn=2, expected_attribute="size")

    assert not any(item.attribute == "category" for item in result.constraints)


@pytest.mark.parametrize(
    ("message", "expected_attribute", "vocabulary", "expected"),
    (
        ("category: Boots", "size", {}, ("category", "boots")),
        ("size: Adjustable", "style", {"size": ("Adjustable",)}, ("size", "adjustable")),
        ("style: Adjustable", "size", {"style": ("Adjustable",)}, ("style", "adjustable")),
    ),
)
def test_explicit_attribute_cues_allow_only_the_cued_cross_attribute_alias_while_pending(
    message, expected_attribute, vocabulary, expected
):
    result = MessageParser(vocabulary).parse(
        message, turn=2, expected_attribute=expected_attribute
    )

    assert [(item.attribute, item.value) for item in result.constraints] == [expected]


@pytest.mark.parametrize(
    ("message", "expected_attribute", "expected"),
    (
        ("size: Adjustable", "style", ("size", "adjustable")),
        ("style: Adjustable", "size", ("style", "adjustable")),
    ),
)
def test_explicit_cue_excludes_same_span_aliases_for_the_pending_attribute(
    message, expected_attribute, expected
):
    result = MessageParser(
        {"size": ("Adjustable",), "style": ("Adjustable",)}
    ).parse(message, turn=2, expected_attribute=expected_attribute)

    assert [(item.attribute, item.value) for item in result.constraints] == [expected]


@pytest.mark.parametrize("message", ("I need boots", "I want boots", "looking for boots"))
def test_pending_goal_phrase_allows_category_replacement_without_override_word(message):
    result = MessageParser({"category": ("Boots",)}).parse(
        message, turn=2, expected_attribute="size"
    )

    assert result.is_override is False
    assert result.is_goal_replacement is True
    assert [(item.attribute, item.value) for item in result.constraints] == [
        ("category", "boots")
    ]


def test_pending_ordinary_category_word_does_not_become_goal_replacement():
    result = MessageParser({"category": ("Boots",)}).parse(
        "boots please", turn=2, expected_attribute="size"
    )

    assert result.is_goal_replacement is False
    assert result.constraints == ()


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("category: boots", ("category", "boots")),
        ("color: blue", ("color", "blue")),
    ),
)
def test_explicit_cue_span_does_not_fallback_to_pending_soft_text(message, expected):
    result = MessageParser({"category": ("Boots",)}).parse(
        message, turn=2, expected_attribute="size"
    )

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        (*expected, True)
    ]


def test_residual_open_text_after_explicit_cue_remains_bounded_pending_soft_text():
    result = MessageParser({"category": ("Boots",)}).parse(
        "category: boots for snow", turn=2, expected_attribute="size"
    )

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        ("category", "boots", True),
        ("size", "for snow", False),
    ]


@pytest.mark.parametrize(
    "message", ("I need blue leather boots", "shopping for blue leather boots")
)
def test_pending_goal_clause_allows_attributes_until_the_category_head(message):
    result = MessageParser(
        {"category": ("Boots",), "size": ("Boots",)}
    ).parse(message, turn=2, expected_attribute="size")

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        ("color", "blue", True),
        ("material", "leather", True),
        ("category", "boots", True),
    ]


def test_pending_goal_clause_keeps_only_unrecognized_suffix_as_soft_evidence():
    result = MessageParser({"category": ("Boots",), "size": ("Boots",)}).parse(
        "I need boots for snow", turn=2, expected_attribute="size"
    )

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        ("category", "boots", True),
        ("size", "for snow", False),
    ]


@pytest.mark.parametrize("message", ("I need handbags", "shopping for handbags"))
def test_pending_goal_uses_dynamic_category_head_not_just_fixed_categories(message):
    result = MessageParser({"category": ("Handbags",)}).parse(
        message, turn=2, expected_attribute="size"
    )

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        ("category", "handbags", True)
    ]


def test_pending_goal_prefers_full_dynamic_category_head_over_overlapping_attribute():
    result = MessageParser(
        {"category": ("Hiking Boots",), "use_case": ("Hiking",)}
    ).parse("I need hiking boots", turn=2, expected_attribute="size")

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        ("category", "hiking boots", True)
    ]


def test_pending_goal_allows_recognized_attributes_after_the_category_head():
    result = MessageParser({"category": ("Boots",)}).parse(
        "I need boots in blue leather", turn=2, expected_attribute="size"
    )

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        ("category", "boots", True),
        ("color", "blue", True),
        ("material", "leather", True),
    ]


@pytest.mark.parametrize("message", ("I don't want leather boots", "I do not want leather boots"))
def test_pending_negative_goal_does_not_start_a_positive_category_replacement(message):
    result = MessageParser({"category": ("Boots",)}).parse(
        message, turn=2, expected_attribute="size"
    )

    assert result.is_goal_replacement is False
    assert [(item.attribute, item.value, item.operator) for item in result.constraints] == [
        ("material", "leather", "not_in")
    ]


def test_explicit_category_replacement_is_a_pending_goal_replacement():
    result = MessageParser({"category": ("Boots",)}).parse(
        "category: boots", turn=2, expected_attribute="size"
    )

    assert result.is_goal_replacement is True
    assert [(item.attribute, item.value) for item in result.constraints] == [
        ("category", "boots")
    ]


@pytest.mark.parametrize("joiner", ("with", "that has", "but with"))
def test_pending_goal_clauses_allow_feature_after_the_category_head(joiner):
    result = MessageParser({"category": ("Boots",), "feature": ("Waterproof",)}).parse(
        f"I need boots {joiner} waterproof lining",
        turn=2,
        expected_attribute="size",
    )

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        ("category", "boots", True),
        ("feature", "waterproof", True),
    ]


def test_override_no_preference_retains_the_pending_attribute():
    result = MessageParser().parse(
        "Actually, no preference", turn=2, expected_attribute="size"
    )

    assert result.is_override is True
    assert result.no_preference_attribute == "size"
    assert result.constraints == ()


@pytest.mark.parametrize(
    "message",
    (
        "I'm not looking for boots",
        "I never want boots",
        "I no longer want boots",
        "I don't really want boots",
    ),
)
def test_pending_bounded_negative_goal_phrases_do_not_create_category_or_soft_size(message):
    result = MessageParser({"category": ("Boots",)}).parse(
        message, turn=2, expected_attribute="size"
    )

    assert result.is_goal_replacement is False
    assert result.constraints == ()


@pytest.mark.parametrize(
    "message",
    (
        "I don't currently want boots",
        "I do not particularly want boots",
        "I am definitely not looking for boots",
    ),
)
def test_pending_structured_negative_goal_with_bounded_adverbs_has_no_category_or_soft_size(
    message,
):
    result = MessageParser({"category": ("Boots",)}).parse(
        message, turn=2, expected_attribute="size"
    )

    assert result.is_goal_replacement is False
    assert result.constraints == ()


def test_no_preference_override_does_not_replace_preferences():
    result = MessageParser().parse(
        "I've changed my mind, no preference", turn=2, expected_attribute="size"
    )

    assert result.is_override is True
    assert result.no_preference_attribute == "size"
    assert result.replace_preferences is False


@pytest.mark.parametrize("expected_attribute", ("feature", "other"))
def test_lining_remains_soft_text_when_it_is_the_pending_answer(expected_attribute):
    result = MessageParser().parse("lining", turn=2, expected_attribute=expected_attribute)

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        (expected_attribute, "lining", False)
    ]


@pytest.mark.parametrize(
    ("message", "vocabulary", "expected"),
    (
        (
            "I need boots with breathable lining",
            {"category": ("Boots",), "feature": ("Breathable",)},
            [("category", "boots", True), ("feature", "breathable", True)],
        ),
        (
            "I need boots with waterproof padding",
            {"category": ("Boots",), "feature": ("Waterproof",)},
            [("category", "boots", True), ("feature", "waterproof", True)],
        ),
        (
            "I need boots with waterproof inner lining",
            {"category": ("Boots",), "feature": ("Waterproof",)},
            [("category", "boots", True), ("feature", "waterproof", True)],
        ),
    ),
)
def test_goal_modifier_residual_is_derived_from_recognized_attribute_spans(
    message, vocabulary, expected
):
    result = MessageParser(vocabulary).parse(
        message, turn=2, expected_attribute="size"
    )

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == expected


def test_unknown_goal_requirement_after_a_category_remains_soft_evidence():
    result = MessageParser({"category": ("Boots",)}).parse(
        "I need boots with a magnetic clasp", turn=2, expected_attribute="size"
    )

    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == [
        ("category", "boots", True),
        ("size", "with a magnetic clasp", False),
    ]


@pytest.mark.parametrize(
    ("message", "vocabulary", "expected"),
    (
        (
            "category: boots with waterproof lining",
            {"category": ("Boots",), "feature": ("Waterproof",)},
            [("category", "boots", True), ("feature", "waterproof", True)],
        ),
        (
            "category: boots in blue leather",
            {"category": ("Boots",)},
            [
                ("category", "boots", True),
                ("color", "blue", True),
                ("material", "leather", True),
            ],
        ),
    ),
)
def test_explicit_category_replacement_allows_same_line_attributes(
    message, vocabulary, expected
):
    result = MessageParser(vocabulary).parse(
        message, turn=2, expected_attribute="size"
    )

    assert result.is_goal_replacement is True
    assert [(item.attribute, item.value, item.is_hard) for item in result.constraints] == expected


def test_parse_caches_raw_candidates_and_goal_heads_once(monkeypatch):
    parser = MessageParser({"category": ("Hiking Boots",), "use_case": ("Hiking",)})
    raw_calls = 0
    goal_calls = 0
    scan = parser._scan_raw_vocabulary_matches
    compute = parser._compute_goal_category_head_spans

    def count_raw(text):
        nonlocal raw_calls
        raw_calls += 1
        return scan(text)

    def count_goal(text):
        nonlocal goal_calls
        goal_calls += 1
        return compute(text)

    monkeypatch.setattr(parser, "_scan_raw_vocabulary_matches", count_raw)
    monkeypatch.setattr(parser, "_compute_goal_category_head_spans", count_goal)

    result = parser.parse(
        "I need hiking boots " + "with waterproof lining " * 128,
        2,
        "size",
    )

    assert any(item.attribute == "category" for item in result.constraints)
    assert (raw_calls, goal_calls) == (1, 1)


@pytest.mark.parametrize(
    ("message", "expected_attribute", "vocabulary", "expected"),
    (
        ("brand Acme", "style", {"brand": ("Acme",)}, ("brand", "acme")),
        ("by Acme", "style", {"brand": ("Acme",)}, ("brand", "acme")),
        (
            "casual style",
            "use_case",
            {"style": ("Casual",), "use_case": ("Casual",)},
            ("style", "casual"),
        ),
    ),
)
def test_natural_cues_are_exclusive_and_do_not_soft_fill_the_pending_attribute(
    message, expected_attribute, vocabulary, expected
):
    result = MessageParser(vocabulary).parse(
        message, turn=2, expected_attribute=expected_attribute
    )

    assert [(item.attribute, item.value) for item in result.constraints] == [expected]


def test_goal_override_allows_category_replacement_while_pending():
    result = MessageParser({"category": ("Boots",)}).parse(
        "Actually, I need boots", turn=2, expected_attribute="size"
    )

    assert result.is_override is True
    assert [(item.attribute, item.value) for item in result.constraints] == [
        ("category", "boots")
    ]


def test_no_preference_without_pending_attribute_does_not_reject_mentioned_attribute():
    result = MessageParser().parse("No preference for color", turn=2)

    assert result.no_preference_attribute is None


def test_catalog_vocabulary_keeps_or_grouping_separate_from_other_attributes():
    parser = MessageParser({"category": ("Shoes",)})

    result = parser.parse("black or blue leather shoes", turn=1)

    assert {(item.attribute, item.value) for item in result.constraints} == {
        ("color", "black"),
        ("material", "leather"),
        ("category", "shoes"),
    }
    color = next(item for item in result.constraints if item.attribute == "color")
    assert (color.operator, color.alternatives) == ("in", ("black", "blue"))


def test_catalog_plural_and_lexical_category_detection_do_not_duplicate_constraint():
    parser = MessageParser({"category": ("Belts",)})

    result = parser.parse("I need a black leather belt", turn=1)

    categories = [item for item in result.constraints if item.attribute == "category"]
    assert len(categories) == 1
    assert categories[0].value == "belts"


def test_catalog_vocabulary_matching_does_not_compile_regex_per_message(
    monkeypatch,
):
    parser = MessageParser({"brand": tuple(f"brand {index}" for index in range(200))})

    def fail_if_compiled(_value):
        raise AssertionError("catalog phrase regex should be compiled at construction")

    monkeypatch.setattr(MessageParser, "_phrase_pattern", staticmethod(fail_if_compiled))

    result = parser.parse("I want brand 123", turn=1)

    assert any(
        item.attribute == "brand" and item.value == "brand 123"
        for item in result.constraints
    )


def test_catalog_aliases_ignore_instruction_stopwords_and_later_description_text():
    parser = MessageParser(
        {
            "brand": ("key", "pants", "Acme Fashion"),
            "category": ("casual", "shoes", "bands"),
            "size": ("adjustable", "medium"),
        }
    )

    result = parser.parse(
        "I'm looking for Dresses Casual. A key requirement: fabric. Stainless Steel Band.",
        turn=1,
    )
    pairs = {(item.attribute, item.value) for item in result.constraints}

    assert ("brand", "key") not in pairs
    assert ("brand", "pants") not in pairs
    assert ("category", "casual") not in pairs
    assert ("category", "bands") not in pairs


def test_catalog_description_aliases_do_not_create_adjustable_hard_constraints():
    parser = MessageParser(
        {
            "size": ("adjustable",),
            "style": ("adjustable",),
            "feature": ("adjustable",),
        }
    )

    result = parser.parse(
        "I need a leather belt. Adjustable strap and buckle closure.", turn=1
    )

    assert "adjustable" not in {
        item.value for item in result.constraints if item.is_hard
    }


def test_catalog_category_spans_protect_nested_style_and_brand_aliases():
    parser = MessageParser(
        {
            "category": ("Fashion Sneakers", "Boy Shorts"),
            "style": ("Sneaker",),
            "brand": ("Boy",),
        }
    )

    fashion = parser.parse("I want Fashion Sneakers", turn=1)
    boy = parser.parse("I want Boy Shorts", turn=1)

    assert ("category", "fashion sneakers") in {
        (item.attribute, item.value) for item in fashion.constraints
    }
    assert not any(
        item.attribute == "style" and item.value == "sneaker" and item.is_hard
        for item in fashion.constraints
    )
    assert ("category", "boy shorts") in {
        (item.attribute, item.value) for item in boy.constraints
    }
    assert not any(
        item.attribute == "brand" and item.value == "boy" and item.is_hard
        for item in boy.constraints
    )


def test_explicit_style_and_brand_cues_survive_category_span_protection():
    parser = MessageParser(
        {
            "category": ("Fashion Sneakers", "Boy Shorts"),
            "style": ("Sneaker",),
            "brand": ("Boy",),
        }
    )

    fashion = parser.parse("I want sneaker style Fashion Sneakers", turn=1)
    boy = parser.parse("I want Boy brand shorts", turn=1)

    assert ("style", "sneaker") in {
        (item.attribute, item.value) for item in fashion.constraints
    }
    assert ("brand", "boy") in {
        (item.attribute, item.value) for item in boy.constraints
    }


def test_pending_attribute_is_an_explicit_cue_for_dynamic_catalog_aliases():
    parser = MessageParser(
        {"brand": ("Acme Fashion",), "style": ("Sneaker",)}
    )

    brand = parser.parse("Acme Fashion", turn=2, expected_attribute="brand")
    style = parser.parse("Sneaker", turn=2, expected_attribute="style")

    assert [
        (item.attribute, item.value, item.confidence, item.is_hard, item.source)
        for item in (*brand.constraints, *style.constraints)
    ] == [
        ("brand", "acme fashion", 1.0, True, "clarification"),
        ("style", "sneaker", 1.0, True, "clarification"),
    ]


def test_dynamic_catalog_aliases_accept_explicit_label_punctuation():
    parser = MessageParser(
        {"brand": ("Acme Fashion",), "style": ("Sneaker",)}
    )

    brand = parser.parse("brand: Acme Fashion", turn=1)
    style = parser.parse("style: Sneaker", turn=1)

    assert ("brand", "acme fashion") in {
        (item.attribute, item.value) for item in brand.constraints
    }
    assert ("style", "sneaker") in {
        (item.attribute, item.value) for item in style.constraints
    }


@pytest.mark.parametrize(
    "message",
    (
        "LOOK: Sneaker",
        "design - Sneaker",
        "Sneaker, LOOK",
        "Sneaker; DESIGN",
    ),
)
def test_dynamic_style_aliases_accept_nearby_look_and_design_cues(message):
    result = MessageParser({"style": ("Sneaker",)}).parse(message, turn=1)

    assert ("style", "sneaker") in {
        (item.attribute, item.value) for item in result.constraints
    }


def test_uncued_dynamic_style_alias_remains_suppressed():
    result = MessageParser({"style": ("Sneaker",)}).parse(
        "I want Sneaker shoes", turn=1
    )

    assert not any(
        item.attribute == "style" and item.is_hard
        for item in result.constraints
    )


@pytest.mark.parametrize(
    "taxonomy",
    (
        "Shoe & Jewelry",
        "Shoes and Jewelry",
        "shoe AND jewelry Men",
        "SHOES & JEWELRY WOMEN",
        "Clothing Shoe & Jewelry Men",
        "CLOTHING, SHOES and JEWELRY Women",
    ),
)
def test_amazon_root_taxonomy_variants_are_not_hard_categories(taxonomy):
    parser = MessageParser(
        {"category": ("Shoes", "Jewelry", "Fashion Sneakers")}
    )

    result = parser.parse(
        f"I'm looking for {taxonomy}, but I'm still exploring.", turn=1
    )

    assert not any(
        item.attribute == "category" and item.is_hard
        for item in result.constraints
    )
