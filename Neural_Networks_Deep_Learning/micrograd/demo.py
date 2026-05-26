"""
Micrograd demo: raw Value autograd + MLP training on a toy dataset.
"""
import random
from micrograd.engine import Value
from micrograd.nn import MLP

# ── 1. Raw autograd ──────────────────────────────────────────────────────────

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

# Build a small expression: e = relu((a * b + c) ** 2)
d = a * b + c     # 2*(-3) + 10 = 4
e = (d ** 2).relu()  # relu(16) = 16

print("=== Raw Value autograd ===")
print(f"d = {d}")
print(f"e = {e}")

e.backward()

print(f"a.grad = {a.grad:.4f}  (expected: 2 * d * b = 2*4*(-3) = -24)")
print(f"b.grad = {b.grad:.4f}  (expected: 2 * d * a = 2*4*2   =  16)")
print(f"c.grad = {c.grad:.4f}  (expected: 2 * d * 1 = 2*4*1   =   8)")
print()

# ── 2. MLP binary classification ─────────────────────────────────────────────

random.seed(42)

# Toy XOR-like dataset: 4 points, labels +1 / -1
xs = [
    [2.0,  3.0],
    [-1.0, -2.0],
    [1.5, -1.0],
    [-2.0,  1.0],
]
ys = [1.0, 1.0, -1.0, -1.0]   # desired targets

# 2-input → hidden layer of 4 → hidden layer of 4 → 1 output (linear)
model = MLP(2, [4, 4, 1])
print("=== MLP structure ===")
print(model)
print(f"Total parameters: {len(model.parameters())}")
print()

# Training loop — SGD with hinge loss: loss = mean(max(0, 1 - y*ypred))
learning_rate = 0.05
for step in range(50):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((1 + -yi * ypi).relu() for yi, ypi in zip(ys, ypred)) * (1 / len(ys))

    # Backward pass
    model.zero_grad()
    loss.backward()

    # SGD update
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0 or step == 49:
        preds = [round(ypi.data) for ypi in ypred]
        correct = sum(int((p > 0) == (y > 0)) for p, y in zip(preds, ys))
        print(f"step {step:3d} | loss = {loss.data:.4f} | accuracy = {correct}/{len(ys)}")

print()
print("=== Final predictions ===")
for x, y, yp in zip(xs, ys, ypred):
    sign = "+" if yp.data > 0 else "-"
    print(f"  input={x}  target={int(y):+d}  raw={yp.data:+.3f}  predicted={sign}1")
