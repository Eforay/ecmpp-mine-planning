"""ECMPP instance generator with tunable complexity.

Generates synthetic mine planning instances with environmental constraints
ranging from pedagogical (25 blocks) to industrial-scale (10^5 blocks).
"""

import csv
import os
import random
import re
import numpy as np
from typing import List, Tuple, Dict


class ECMPPGenerator:
    """Generate ECMPP instances with controllable parameters."""
    
    def __init__(self, seed=42):
        self.random = random.Random(seed)
        np.random.seed(seed)
    
    def _get_precedence(self, block_id: str) -> List[str]:
        """Get precedence constraints based on layer structure."""
        precedence_map = {
            # Layer L1: no predecessors
            'B1': [], 'B2': [], 'B3': [], 'B4': [], 'B5': [],
            # Layer L2: require blocks above
            'B6': ['B1'], 'B7': ['B2'], 'B8': ['B3'], 'B9': ['B4'], 'B10': ['B5'],
            # Layer L3: require blocks above
            'B11': ['B6'], 'B12': ['B7'], 'B13': ['B8'], 'B14': ['B9'], 'B15': ['B10'],
            # Layer L4: require blocks above
            'B16': ['B11'], 'B17': ['B12'], 'B18': ['B13'], 'B19': ['B14'], 'B20': ['B15'],
            # Layer L5: require blocks above
            'B21': ['B16'], 'B22': ['B17'], 'B23': ['B18'], 'B24': ['B19'], 'B25': ['B20'],
        }
        return precedence_map.get(block_id, [])
    
    def _parse_clause(self, clause_str: str) -> List[str]:
        """Parse a clause string into a list of literals."""
        # Remove parentheses and split by ∨
        clause_str = clause_str.strip('()')
        literals = re.split(r'\s*∨\s*', clause_str)
        return [lit.strip() for lit in literals]
    
    def generate_pedagogical(self, data_path: str = '25block_example') -> Tuple[List[Dict], List[List[str]], int]:
        """Generate the 25-block pedagogical instance from Appendix A."""
        blocks = []
        
        # Load block data from CSV
        csv_path = os.path.join(data_path, 'block_model.csv')
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                blocks.append({
                    'id': row['block'],
                    'value': float(row['net']),
                    'layer': row['layer'],
                    'energy': float(row['energy']),
                    'water': float(row['water']),
                    'land': float(row['land']),
                    'noise': float(row['noise']),
                    'sulfide': row['sulfide'] == 'yes',
                    'precedence': self._get_precedence(row['block'])
                })
        
        # Load clauses
        clauses = []
        clauses_path = os.path.join(data_path, 'clauses.txt')
        with open(clauses_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line:
                    clause_str = line.split('=')[1].strip()
                    clause = self._parse_clause(clause_str)
                    clauses.append(clause)
        
        npv_min = 28
        return (blocks, clauses, npv_min)
    
    def generate_synthetic(self, n_blocks: int, n_clauses: int,
                           clause_density: float = 0.1) -> Tuple[List[Dict], List[List[str]], int]:
        """Generate a synthetic ECMPP instance.
        
        Args:
            n_blocks: Number of mining blocks
            n_clauses: Number of environmental clauses
            clause_density: Average number of literals per clause
        
        Returns:
            Tuple of (blocks, clauses, npv_min)
        """
        blocks = []
        # Generate blocks with log-normal value distribution
        values = np.random.lognormal(mean=5, sigma=2, size=n_blocks)
        
        for i in range(n_blocks):
            blocks.append({
                'id': f'B_{i+1}',
                'value': float(values[i]),
                'precedence': []  # Simplified: no precedence for synthetic
            })
        
        # Generate random 3-CNF clauses
        clauses = []
        for _ in range(n_clauses):
            # Select 1-3 random blocks for the clause
            n_lits = min(3, max(1, int(clause_density * n_blocks)))
            vars_in_clause = self.random.sample(blocks, n_lits)
            clause = []
            for var in vars_in_clause:
                if self.random.random() < 0.5:
                    clause.append(var['id'])
                else:
                    clause.append(f"~{var['id']}")
            # Pad to exactly 3 literals if needed
            while len(clause) < 3:
                clause.append(clause[0])  # repeat literal
            clauses.append(clause)
        
        npv_min = sum(b['value'] for b in blocks) * 0.3  # 30% threshold
        
        return (blocks, clauses, npv_min)
    
    def generate_scaling_set(self, sizes: List[int]) -> Dict:
        """Generate a set of instances for scaling experiments."""
        results = {}
        for size in sizes:
            n_clauses = max(10, int(size * 0.1))  # 10% of blocks
            results[size] = self.generate_synthetic(size, n_clauses)
        return results


# Convenience function for quick generation
def generate_pedagogical_instance():
    """Quick function to generate the pedagogical 25-block instance."""
    generator = ECMPPGenerator()
    return generator.generate_pedagogical()