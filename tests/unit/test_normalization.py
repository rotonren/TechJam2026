from compasscart.normalization import extract_attributes, terms


def test_extracts_material_color_budget_and_use_case():
    product = {
        "title": "Blue mesh running shoes",
        "price": 79.99,
        "features": ["Breathable mesh for gym and outdoor running"],
        "details": {"Color": "Navy Blue", "Material": "Mesh"},
    }

    attributes = extract_attributes(product)

    assert "blue" in attributes["color"]
    assert "mesh" in attributes["material"]
    assert "running" in attributes["use_case"]
    assert attributes["budget"] == ("79.99",)


def test_terms_are_normalized_and_deduplicated():
    assert terms("Blue BLUE running-shoes") == ["blue", "running", "shoes"]
