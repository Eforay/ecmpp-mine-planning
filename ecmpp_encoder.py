"""ECMPP to Z3 SMT encoding.

This module encodes the Environmentally Constrained Mine Planning Problem
as a Z3 SMT formula.
"""

from z3 import *


class ECMPPEncoder:
    """Encodes ECMPP instance as Z3 SMT formula."""
    
    def __init__(self, blocks, clauses, npv_min):
        """
        Args:
            blocks: List of block objects with attributes:
                    - id: block identifier
                    - value: economic value
                    - precedence: list of predecessor block ids
            clauses: List of CNF clauses, each a list of literals
            npv_min: Minimum NPV threshold
        """
        self.blocks = blocks
        self.clauses = clauses
        self.npv_min = npv_min
        self.solver = Solver()
        self.block_vars = {}
        
    def encode(self):
        """Encode ECMPP as Z3 constraints."""
        # Create Boolean variables for each block
        for block in self.blocks:
            self.block_vars[block.id] = Bool(f"x_{block.id}")
        
        # Precedence constraints (conjunctive)
        for block in self.blocks:
            for pred_id in block.precedence:
                self.solver.add(
                    Implies(self.block_vars[block.id], 
                           self.block_vars[pred_id])
                )
        
        # Environmental clauses (CNF)
        for clause in self.clauses:
            clause_literals = []
            for lit in clause:
                if lit.startswith('~'):
                    var = lit[1:]
                    clause_literals.append(Not(self.block_vars[var]))
                else:
                    clause_literals.append(self.block_vars[lit])
            self.solver.add(Or(*clause_literals))
        
        # NPV constraint (pseudo-Boolean)
        total_value = Sum([If(self.block_vars[b.id], b.value, 0) 
                          for b in self.blocks])
        self.solver.add(total_value >= self.npv_min)
        
    def solve(self, timeout=10000):
        """Solve the encoded problem.
        
        Returns:
            tuple: (sat_status, model, npv)
        """
        self.encode()
        self.solver.set("timeout", timeout)
        
        if self.solver.check() == sat:
            model = self.solver.model()
            npv = sum([b.value for b in self.blocks 
                       if model[self.block_vars[b.id]]])
            return ("sat", model, npv)
        else:
            return ("unsat", None, None)