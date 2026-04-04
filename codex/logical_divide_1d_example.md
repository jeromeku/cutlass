# Trace: `logical_divide_1d_example`

## Code Map

- [logical_divide_1d_example](../cuteDSL-examples/notebooks/layout-algebra.py#L5)
- [run_logical_divide_1d](../cuteDSL-examples/notebooks/layout-algebra.py#L26)
- [logical_divide](../python/pycute/layout.py#L297)
- [composition](../python/pycute/layout.py#L190)
- [complement](../python/pycute/layout.py#L232)
- [crd2idx](../python/pycute/int_tuple.py#L160)

## Key Functions Index

| Function | File | Purpose |
|----------|------|---------|
| `logical_divide` | [layout.py](../python/pycute/layout.py#L297) | Builds the divided layout as `composition(A, make_layout(B, complement(B, size(A))))`. |
| `complement` | [layout.py](../python/pycute/layout.py#L232) | Computes the "rest" layout that fills whatever the tiler does not cover. |
| `composition` | [layout.py](../python/pycute/layout.py#L190) | Pushes the tile/rest layout through the original layout mapping. |
| `crd2idx` | [int_tuple.py](../python/pycute/int_tuple.py#L160) | Explains how a single integer coordinate is interpreted against a hierarchical shape. |

## Example Setup

From [layout-algebra.py](../cuteDSL-examples/notebooks/layout-algebra.py#L11):

```python
layout = cute.make_layout((4, 2, 3), stride=(2, 1, 8))
tiler = cute.make_layout(4, stride=2)
result = cute.logical_divide(layout, tiler=tiler)
```

Write the original layout as:

- `A = (4, 2, 3):(2, 1, 8)`
- `B = 4:2`

The original layout maps a coordinate `(i, j, k)` to:

```text
A(i, j, k) = 2*i + 1*j + 8*k
```

So the domain has size:

```text
size(A) = 4 * 2 * 3 = 24
```

## Big Picture

The reference implementation says:

```python
logical_divide(layoutA, layoutB) =
    composition(layoutA, make_layout(layoutB, complement(layoutB, size(layoutA))))
```

For this example, CuTe computes:

```text
logical_divide(A, B) = composition(A, make_layout(B, complement(B, 24)))
```

So there are three steps:

1. Compute the complement of the tiler under the 24-element domain.
2. Form a new hierarchical layout `(tile, rest)`.
3. Compose that layout through `A`.

## Step 1: Compute `complement(B, 24)`

The tiler is:

```text
B = 4:2
```

That means tile coordinate `t in [0, 4)` maps to the domain ordinal:

```text
B(t) = 2*t = {0, 2, 4, 6}
```

The complement algorithm in [layout.py](../python/pycute/layout.py#L232) walks strides in increasing order.
For `4:2` there is one nontrivial `(stride, shape)` pair: `(2, 4)`.

Initial state:

```text
current_idx = 1
result = []
```

Process `(stride=2, shape=4)`:

```text
append shape  = stride // current_idx = 2 // 1 = 2
append stride = current_idx           = 1
current_idx   = shape * stride        = 4 * 2 = 8
```

Finish with the tail factor needed to cover all 24 elements:

```text
append shape  = ceil_div(24, 8) = 3
append stride = 8
```

So:

```text
complement(B, 24) = (2, 3):(1, 8)
```

Interpretation:

- the tiler `4:2` picks positions separated by stride 2
- the complement fills in the leftover inner offset `0/1` and the outer block `0/1/2`

## Step 2: Form the hierarchical `(tile, rest)` layout

Now combine the tiler and its complement:

```text
make_layout(B, complement(B, 24))
= make_layout(4:2, (2, 3):(1, 8))
= (4, (2, 3)):(2, (1, 8))
```

This means every domain ordinal `n` in `[0, 24)` can be written uniquely as:

```text
n = 2*t + u + 8*v
```

with:

- `t in [0, 4)`
- `u in [0, 2)`
- `v in [0, 3)`

So the new logical coordinates are:

```text
(tile_coord, rest_coord) = (t, (u, v))
```

## Step 3: Compose Through the Original Layout

Now apply [composition](../python/pycute/layout.py#L190):

```text
result = composition(A, (4, (2, 3)):(2, (1, 8)))
```

Because the right-hand side is hierarchical, composition is applied to each part:

```text
result =
(
  composition(A, 4:2),
  composition(A, (2, 3):(1, 8))
)
```

### Step 3a: Tile part `composition(A, 4:2)`

By definition:

```text
composition(A, B)(c) = A(B(c))
```

Here `B(c) = 2*c`, so we evaluate `A` at domain ordinals `{0, 2, 4, 6}`.

Using [crd2idx](../python/pycute/int_tuple.py#L160), a single integer ordinal is unpacked against the shape `(4,2,3)` with the first mode varying fastest. For this layout:

```text
A(0) = 0
A(2) = 4
A(4) = 1
A(6) = 5
```

So the tile part enumerates:

```text
{0, 4, 1, 5}
```

CuTe represents that sequence as:

```text
(2, 2):(4, 1)
```

Why this shape and stride?

- shape `(2,2)` because the four tile points are factored into two fast positions and two slow positions
- stride `(4,1)` because moving along the first tile axis jumps by 4, while moving along the second tile axis jumps by 1

You can verify the mapping:

```text
(0,0) -> 0
(1,0) -> 4
(0,1) -> 1
(1,1) -> 5
```

### Step 3b: Rest part `composition(A, (2, 3):(1, 8))`

This part can be read directly as:

```text
A(u + 8*v)
```

Evaluate a few points:

```text
A(0)  = 0
A(1)  = 2
A(8)  = 8
A(9)  = 10
A(16) = 16
A(17) = 18
```

That is exactly:

```text
(2, 3):(2, 8)
```

Why?

- the first rest axis still has extent 2, but now each step moves by 2 in the final codomain
- the second rest axis has extent 3 and each step moves by 8

## Final Result

Putting the tile and rest pieces together:

```text
logical_divide((4,2,3):(2,1,8), 4:2)
= ((2,2), (2,3)):((4,1), (2,8))
```

This is the exact result produced by the Python reference implementation.

## Intuition

The important part is that `logical_divide` does not just return:

```text
(4, (2,3)):(2, (1,8))
```

That object describes how to split the **domain ordinal space** into tile and rest coordinates.
`composition(A, ...)` then pushes that split through the original layout `A`, which re-expresses both pieces in the final codomain.

That is why:

- the tile `4:2` becomes `(2,2):(4,1)`
- the rest `(2,3):(1,8)` becomes `(2,3):(2,8)`

The 4 tile positions are not physically laid out as `0,2,4,6` after passing through `A`; they become `0,4,1,5`, and CuTe captures that with the 2D tile sublayout `(2,2):(4,1)`.
