import torch

from models.encoding import encode_query
from models.neugn import NeuGN, NeuGNConfig
from models.residual_arc import ResidualArcHead, collate_residual, encode_residual
from queries.euler import euler_sequence
from search.graph import Graph


def test_residual_head_preserves_assignment_and_handles_padding():
    q = Graph.from_edges(4, [(0, 1), (1, 2), (2, 3)], [0] * 4)
    qenc = encode_query(q, euler_sequence(q))
    # Residual tokens exchange edge-aware information before CHECK reads them.
    model = NeuGN(NeuGNConfig(num_g=12, num_labels=1, d=16, heads=2, layers=2, ffn=32))
    model.eval().requires_grad_(False)
    head = ResidualArcHead(model.cfg.d).eval()
    first = encode_residual(qenc, [1, 2, -1, -1], {2: frozenset({3, 4, 5}), 3: frozenset({6})},
                            assigned=1, num_g=12, candidates_per_domain=2)
    second = encode_residual(qenc, [1, 2, 3, -1], {3: frozenset({6})},
                             assigned=2, num_g=12, candidates_per_domain=2)
    assert (first["state"]["content"][qenc["seq"] == 1] == 2).all()
    assert first["widths"] == [3, 1]
    assert first["candidates"][0] == [3, 4]
    batch = collate_residual([first, second], "cpu")
    outputs = head(model, batch)
    with torch.no_grad():
        single = head(model, collate_residual([second], "cpu"))
        assert torch.allclose(outputs[1:], single, atol=1e-6)
        assert torch.allclose(outputs, head(model, batch, head.query_context(model, batch)))
    assert outputs.shape == (2,) and outputs.isfinite().all()
    outputs.sum().backward()
    assert head.edge_bias.grad is not None and head.edge_bias.grad.abs() > 0
    assert head.projection.weight.grad is not None
    assert all(p.grad is None for p in model.parameters())
