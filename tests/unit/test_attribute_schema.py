from __future__ import annotations

from types import MappingProxyType

import pytest

from compasscart.attribute_schema import AttributeSchema


def _inverted():
    return {
        "closure": {
            "zipper": {"P1", "P2", "P3"},
            "pull on": {"P4", "P5", "P6"},
        },
        "item_model_number": {
            "ABC-1": {"P1"},
            "ABC-2": {"P2"},
            "ABC-3": {"P3"},
        },
        "material": {"cotton": {"P1", "P4"}},
    }


def test_catalog_schema_exposes_dynamic_values_to_parser():
    schema = AttributeSchema.from_catalog(_inverted(), product_count=6)

    assert "closure" not in schema.parser_vocabulary()
    assert schema.parser_vocabulary(include_discovered=True)["closure"] == (
        "pull on",
        "zipper",
    )
    assert "closure" in schema.discovered_attributes
    assert schema.specifications["closure"].layer == "category"
    assert schema.specifications["closure"].source == "catalog"
    assert schema.cues_for("closure") == ("closure type", "closure")


def test_dynamic_question_attributes_use_evaluator_safe_other_slot():
    schema = AttributeSchema.from_catalog(_inverted(), product_count=6)
    closure = next(
        spec
        for spec in schema.question_specs(include_dynamic=True)
        if spec.name == "closure"
    )

    assert closure.ask_attribute == "other"
    assert closure.question_eligible is True


def test_schema_keeps_global_category_and_dynamic_layers_separate():
    schema = AttributeSchema.from_layers(
        {
            "global": {"brand": {"acme": {"P1", "P2"}}},
            "category": {"closure": {"zipper": {"P1"}, "pull on": {"P2"}}},
            "dynamic": {"occasion": {"work": {"P1"}, "party": {"P2"}}},
        },
        product_count=2,
        category_scopes={"closure": {"footwear"}},
    )

    assert schema.specifications["brand"].layer == "global"
    assert schema.specifications["closure"].category_scopes == ("footwear",)
    assert schema.specifications["occasion"].layer == "dynamic"
    assert "occasion" in schema.dynamic_attributes


def test_catalog_schema_is_read_only():
    schema = AttributeSchema.from_catalog(_inverted(), product_count=6)

    assert isinstance(schema.specifications, MappingProxyType)
    with pytest.raises(TypeError):
        schema.specifications["closure"] = schema.specifications["closure"]
