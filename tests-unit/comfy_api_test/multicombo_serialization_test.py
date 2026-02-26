from comfy_api.latest._io import MultiCombo


def test_multicombo_serializes_multi_select_as_object():
    multi_combo = MultiCombo.Input(
        id="providers",
        options=["a", "b", "c"],
        default=["a"],
    )

    serialized = multi_combo.as_dict()

    assert serialized["multiselect"] is True
    assert "multi_select" in serialized
    assert serialized["multi_select"] == {}


def test_multicombo_serializes_multi_select_with_placeholder_and_chip():
    multi_combo = MultiCombo.Input(
        id="providers",
        options=["a", "b", "c"],
        default=["a"],
        placeholder="Select providers",
        chip=True,
    )

    serialized = multi_combo.as_dict()

    assert serialized["multiselect"] is True
    assert serialized["multi_select"] == {
        "placeholder": "Select providers",
        "chip": True,
    }
    assert serialized["placeholder"] == "Select providers"
    assert serialized["chip"] is True
