#!/usr/bin/env python3
"""
Базовая реализация Merkle Tree.
"""

import hashlib
from typing import List

class FractalMerkleTree:
    """Упрощенная реализация Merkle Tree для тестирования."""
    
    def __init__(self):
        self.leaves = []
        self.tree = []
        self.root = None
        
    def add_leaf(self, data: str):
        """Добавление листа в дерево."""
        leaf_hash = hashlib.sha256(data.encode()).hexdigest()
        self.leaves.append(leaf_hash)
        
    def build_tree(self):
        """Построение Merkle Tree."""
        if not self.leaves:
            return
        
        current_level = self.leaves.copy()
        self.tree = [current_level]
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    combined = current_level[i] + current_level[i + 1]
                else:
                    combined = current_level[i] + current_level[i]
                
                node_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(node_hash)
            
            self.tree.append(next_level)
            current_level = next_level
        
        self.root = current_level[0] if current_level else None
    
    def get_root(self) -> str:
        """Возвращает корень дерева."""
        if self.root is None:
            self.build_tree()
        return self.root or ""
    
    def get_depth(self) -> int:
        """Возвращает глубину дерева."""
        return len(self.tree) if self.tree else 0
    
    def get_node_count(self) -> int:
        """Возвращает количество узлов в дереве."""
        return sum(len(level) for level in self.tree) if self.tree else 0