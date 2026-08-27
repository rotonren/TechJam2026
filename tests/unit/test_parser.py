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


def test_pending_question_does_not_hide_other_recognized_attributes():
    result = MessageParser().parse(
        "Blue leather, please.", turn=2, expected_attribute="color"
    )

    assert {(item.attribute, item.value) for item in result.constraints} == {
        ("color", "blue"),
        ("material", "leather"),
    }


def test_pending_question_does_not_relabel_a_known_other_attribute():
    parser = MessageParser(
        {"category": ("Shoes",), "use_case": ("Running", "Work")}
    )

    result = parser.parse("Blue.", turn=2, expected_attribute="use_case")

    assert [(item.attribute, item.value) for item in result.constraints] == [
        ("color", "blue")
    ]


def test_pending_question_keeps_recognized_color_and_category_together():
    parser = MessageParser({"category": ("Shoes",)})

    result = parser.parse("blue shoes", turn=2, expected_attribute="material")

    assert {(item.attribute, item.value) for item in result.constraints} == {
        ("color", "blue"),
        ("category", "shoes"),
    }


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


def test_pending_size_question_accepts_bare_known_numeric_size_as_hard():
    parser = MessageParser({"size": ("10",)})

    assert parser.supports("size", "10") is True

    result = parser.parse("10", turn=2, expected_attribute="size")

    assert [
        (item.attribute, item.value, item.confidence, item.is_hard, item.source)
        for item in result.constraints
    ] == [("size", "10", 1.0, True, "clarification")]


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


@pytest.mark.parametrize(
    ("attribute", "value", "expected"),
    (
        (" color ", " GRAY ", True),
        ("color", "grey", False),
        ("material", "COTTON", True),
        ("brand", " Acme Fashion ", True),
        ("category", "dress", True),
        ("budget", "$79.5", True),
        ("budget", "79.50", False),
        ("budget", "79.500", False),
        ("budget", "not a number", False),
        ("unknown", "blue", False),
        ("color", "ultraviolet", False),
        ("color", "", False),
    ),
)
def test_parser_reports_normalized_attribute_value_support(
    attribute, value, expected
):
    parser = MessageParser(
        {"brand": ("Acme Fashion",), "category": ("Dresses",)}
    )

    assert parser.supports(attribute, value) is expected


@pytest.mark.parametrize(
    ("attribute", "value"),
    (
        ("color", "gray"),
        ("material", "cotton"),
        ("brand", "Acme Fashion"),
        ("category", "dress"),
        ("budget", "$79.5"),
    ),
)
def test_reported_support_is_consumable_as_a_clarification(attribute, value):
    parser = MessageParser(
        {"brand": ("Acme Fashion",), "category": ("Dresses",)}
    )

    assert parser.supports(attribute, value) is True
    parsed = parser.parse(value, turn=2, expected_attribute=attribute)
    assert any(item.attribute == attribute for item in parsed.constraints)


def test_parser_support_check_is_side_effect_free():
    parser = MessageParser({"brand": ("Acme Fashion",)})
    before = parser.parse(
        "brand: Acme Fashion in blue", turn=2, expected_attribute="brand"
    )

    assert parser.supports("brand", "Acme Fashion") is True
    assert parser.supports("brand", "Unlisted Brand") is False

    after = parser.parse(
        "brand: Acme Fashion in blue", turn=2, expected_attribute="brand"
    )
    assert after == before
    assert parser.supports("brand", "Unlisted Brand") is False
