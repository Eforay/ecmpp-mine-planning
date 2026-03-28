"""
Z3 TUTORIAL FOR YOUR PAPER
===========================
Run each section one at a time to understand what Z3 is doing.
Copy this file to your machine and run: python3 z3_tutorial.py
"""

from z3 import (Bool, And, Or, Not, Implies, If,
                Optimize, sat, unsat, unknown)

print("=" * 60)
print("STEP 1: Boolean variables")
print("=" * 60)

# Z3 Boolean variables work exactly like your x_{B_i} variables
x1 = Bool('x1')   # represents: block B1 is extracted (True) or not (False)
x2 = Bool('x2')
print(f"Created Boolean variable: {x1}")
print(f"Created Boolean variable: {x2}")

print()
print("=" * 60)
print("STEP 2: A simple satisfiability check")
print("=" * 60)

# Create a solver
from z3 import Solver
s = Solver()

# Add a constraint: x1 OR x2 must be true (at least one block extracted)
s.add(Or(x1, x2))

# Add another constraint: NOT both can be true (mutual exclusivity)
s.add(Not(And(x1, x2)))

result = s.check()
print(f"Satisfiable? {result}")   # Should print: sat

if result == sat:
    m = s.model()
    print(f"Solution: x1={m[x1]}, x2={m[x2]}")
    # Z3 will find one valid assignment, e.g. x1=True, x2=False

print()
print("=" * 60)
print("STEP 3: An unsatisfiable problem")
print("=" * 60)

s2 = Solver()
s2.add(x1)           # x1 must be True
s2.add(Not(x1))      # x1 must be False
result2 = s2.check()
print(f"Satisfiable? {result2}")   # Should print: unsat
# This is like trying to extract and not-extract the same block — impossible

print()
print("=" * 60)
print("STEP 4: The Optimize solver (for NPV maximisation)")
print("=" * 60)

# For your paper you need to MAXIMISE NPV subject to constraints.
# Z3's Optimize solver does exactly this.
from z3 import Int, Real

# Simple example: two blocks with values, maximise total
b1 = Bool('b1')
b2 = Bool('b2')
val_b1 = 5.3  # net NPV contribution of block 1
val_b2 = 4.3  # net NPV contribution of block 2

opt = Optimize()

# NPV = val_b1 * b1 + val_b2 * b2
# Z3 works with integers or reals natively.
# For Boolean * number, use If(bool_var, value, 0)
npv = If(b1, val_b1, 0.0) + If(b2, val_b2, 0.0)

# Constraint: cannot extract both (mutual exclusivity example)
opt.add(Not(And(b1, b2)))

# Maximise NPV
opt.maximize(npv)

result3 = opt.check()
print(f"Satisfiable? {result3}")
if result3 == sat:
    m3 = opt.model()
    print(f"b1={m3[b1]}, b2={m3[b2]}")
    # Should choose b1=True (higher value) b2=False
    solved_npv = (val_b1 if m3[b1] else 0) + (val_b2 if m3[b2] else 0)
    print(f"Achieved NPV = ${solved_npv:.1f}M")

print()
print("=" * 60)
print("STEP 5: Your paper's clause structure in Z3")
print("=" * 60)

# This is exactly how your environmental clauses translate to Z3
# Using two blocks as a mini-example

x_B17 = Bool('x_B17')   # True = B17 is extracted
x_B22 = Bool('x_B22')
x_B23 = Bool('x_B23')

# Your clause C3a: NOT (B17 AND B22 AND B23) — three-way exclusion
# In Z3: Or(Not(x_B17), Not(x_B22), Not(x_B23))
C3a = Or(Not(x_B17), Not(x_B22), Not(x_B23))

# Your clause C3b: NOT (B17 AND B22) — pairwise exclusion
# In Z3: Or(Not(x_B17), Not(x_B22))
C3b = Or(Not(x_B17), Not(x_B22))

s3 = Solver()
s3.add(C3a)
s3.add(C3b)

# Ask: can we extract all three?
s3.add(x_B17, x_B22, x_B23)
result4 = s3.check()
print(f"Can we extract B17+B22+B23? {result4}")  # unsat

s3.reset()
s3.add(C3a)
s3.add(C3b)
# Ask: can we extract B17+B22 but NOT B23?
s3.add(x_B17, x_B22, Not(x_B23))
result5 = s3.check()
print(f"Can we extract B17+B22 (not B23)? {result5}")  # unsat — C3b catches this

s3.reset()
s3.add(C3a)
s3.add(C3b)
# Ask: can we extract just B17?
s3.add(x_B17, Not(x_B22), Not(x_B23))
result6 = s3.check()
print(f"Can we extract only B17? {result6}")  # sat

print()
print("All tutorial steps complete.")
print("Now run z3_ecmpp_experiment.py for the full paper experiment.")
