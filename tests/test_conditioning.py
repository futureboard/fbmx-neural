"""The conditioning API is the contract between data, model and runtime."""

from __future__ import annotations

import pytest
import torch

from fbmx.conditioning import (
    CategoricalParam,
    ConditioningSchema,
    ContinuousParam,
    ParamBatch,
)


def test_continuous_normalisation_round_trip():
    p = ContinuousParam("attack", 20.0, 800.0, 100.0, "us")
    assert p.normalize(20.0) == pytest.approx(-1.0)
    assert p.normalize(800.0) == pytest.approx(1.0)
    assert p.denormalize(p.normalize(357.0)) == pytest.approx(357.0)
    # out-of-range values clamp rather than extrapolate
    assert p.normalize(-5.0) == pytest.approx(-1.0)


def test_continuous_rejects_bad_range():
    with pytest.raises(ValueError):
        ContinuousParam("x", 1.0, 0.0)
    with pytest.raises(ValueError):
        ContinuousParam("x", 0.0, 1.0, default=2.0)


def test_categorical_is_not_an_ordinal():
    p = CategoricalParam("mode", ("4:1", "8:1", "12:1", "20:1", "all_buttons"))
    assert p.num_categories == 5
    assert p.index("all_buttons") == 4
    with pytest.raises(ValueError):
        p.index("nope")
    # embeddings, not a scalar axis
    assert p.embedding_dim >= 1


def test_schema_encode_decode():
    schema = ConditioningSchema(
        continuous=(ContinuousParam("drive", 0.0, 10.0, 5.0),),
        categorical=(CategoricalParam("rev", ("D", "E"), "E"),),
    )
    batch = schema.encode({"drive": 2.5, "rev": "D"})
    assert batch.continuous.shape == (1, 1)
    assert batch.categorical.shape == (1, 1)
    decoded = schema.decode(batch)
    assert decoded["drive"] == pytest.approx(2.5)
    assert decoded["rev"] == "D"


def test_schema_defaults_and_strictness():
    schema = ConditioningSchema(continuous=(ContinuousParam("mix", 0.0, 1.0, 0.75),))
    assert schema.decode(schema.encode({}))["mix"] == pytest.approx(0.75)
    with pytest.raises(ValueError):
        schema.encode({"mixx": 0.5})


def test_cond_dim_accounts_for_embeddings():
    schema = ConditioningSchema(
        continuous=(ContinuousParam("a"), ContinuousParam("b")),
        categorical=(CategoricalParam("c", ("x", "y"), embedding_dim=4),),
    )
    assert schema.cond_dim == 2 + 4


def test_empty_schema_is_usable():
    schema = ConditioningSchema()
    assert schema.is_empty and schema.cond_dim == 0
    batch = schema.empty_batch(3)
    assert batch.is_empty and batch.batch_size == 3


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        ConditioningSchema(
            continuous=(ContinuousParam("x"),),
            categorical=(CategoricalParam("x", ("a", "b")),),
        )


def test_param_batch_expand_and_collate():
    schema = ConditioningSchema(continuous=(ContinuousParam("g", 0.0, 1.0, 0.5),))
    one = schema.encode({"g": 0.2})
    many = one.expand_to(4)
    assert many.batch_size == 4
    assert torch.allclose(many.continuous[0], many.continuous[3])
    merged = ParamBatch.collate([one, schema.encode({"g": 0.8})])
    assert merged.batch_size == 2
    with pytest.raises(ValueError):
        merged.expand_to(5)


def test_schema_serialisation_round_trip():
    schema = ConditioningSchema(
        continuous=(ContinuousParam("input", 0.0, 10.0, 5.0, "dial"),),
        categorical=(CategoricalParam("buttons", ("off", "all"), embedding_dim=3),),
    )
    restored = ConditioningSchema.from_dict(schema.to_dict())
    assert restored == schema
    assert restored.cond_dim == schema.cond_dim
