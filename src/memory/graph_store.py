"""
Local Graph Store - Persistence layer for the patient knowledge graph
Handles offline storage of the GraphRAG data
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle


class LocalGraphStore:
    """
    Local storage backend for the patient knowledge graph.
    
    Features:
    - JSON-based persistence for portability
    - Incremental updates
    - Backup and recovery
    - 100% offline operation
    """
    
    def __init__(self, base_dir: str = "./data/graph_store"):
        """
        Initialize the local graph store.
        
        Args:
            base_dir: Base directory for storing graph data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage paths
        self.nodes_dir = self.base_dir / "nodes"
        self.edges_file = self.base_dir / "edges.json"
        self.index_file = self.base_dir / "index.json"
        self.metadata_file = self.base_dir / "metadata.json"
        
        # Create directories
        self.nodes_dir.mkdir(exist_ok=True)
        
        # In-memory cache
        self._node_cache: Dict[str, Dict[str, Any]] = {}
        self._edge_cache: List[Dict[str, Any]] = []
        self._index: Dict[str, Any] = {}
        
        # Load existing data
        self._load_index()
        self._load_edges()
    
    def _load_index(self) -> None:
        """Load the node index from disk"""
        if self.index_file.exists():
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self._index = json.load(f)
        else:
            self._index = {
                "patients": {},
                "node_count": 0,
                "edge_count": 0,
                "last_updated": None
            }
    
    def _save_index(self) -> None:
        """Save the node index to disk"""
        self._index["last_updated"] = datetime.now().isoformat()
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self._index, f, indent=2)
    
    def _load_edges(self) -> None:
        """Load edges from disk"""
        if self.edges_file.exists():
            with open(self.edges_file, 'r', encoding='utf-8') as f:
                self._edge_cache = json.load(f)
        else:
            self._edge_cache = []
    
    def _save_edges(self) -> None:
        """Save edges to disk"""
        with open(self.edges_file, 'w', encoding='utf-8') as f:
            json.dump(self._edge_cache, f, indent=2)
    
    # ==================== Node Operations ====================
    
    def save_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """Save a single node to disk"""
        # Update cache
        self._node_cache[node_id] = node_data
        
        # Save to file
        node_file = self.nodes_dir / f"{node_id}.json"
        with open(node_file, 'w', encoding='utf-8') as f:
            json.dump(node_data, f, indent=2, default=str)
        
        # Update index
        self._index["node_count"] = len(list(self.nodes_dir.glob("*.json")))
        
        # Track patient nodes
        if node_data.get("node_type") == "patient":
            patient_id = node_data.get("data", {}).get("patient_id", node_id)
            self._index["patients"][patient_id] = node_id
        
        self._save_index()
    
    def load_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Load a single node from disk"""
        # Check cache first
        if node_id in self._node_cache:
            return self._node_cache[node_id]
        
        # Load from file
        node_file = self.nodes_dir / f"{node_id}.json"
        if node_file.exists():
            with open(node_file, 'r', encoding='utf-8') as f:
                node_data = json.load(f)
                self._node_cache[node_id] = node_data
                return node_data
        
        return None
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a node from storage"""
        # Remove from cache
        if node_id in self._node_cache:
            del self._node_cache[node_id]
        
        # Remove file
        node_file = self.nodes_dir / f"{node_id}.json"
        if node_file.exists():
            node_file.unlink()
            
            # Update index
            self._index["node_count"] = len(list(self.nodes_dir.glob("*.json")))
            self._save_index()
            
            return True
        
        return False
    
    def get_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Load all nodes from disk"""
        nodes = {}
        for node_file in self.nodes_dir.glob("*.json"):
            node_id = node_file.stem
            with open(node_file, 'r', encoding='utf-8') as f:
                nodes[node_id] = json.load(f)
        
        self._node_cache = nodes
        return nodes
    
    # ==================== Edge Operations ====================
    
    def save_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save an edge relationship"""
        edge = {
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "data": data or {},
            "created_at": datetime.now().isoformat()
        }
        
        # Check for duplicates
        existing = [
            e for e in self._edge_cache
            if e["source"] == source_id and e["target"] == target_id and e["relation"] == relation
        ]
        
        if not existing:
            self._edge_cache.append(edge)
            self._index["edge_count"] = len(self._edge_cache)
            self._save_edges()
            self._save_index()
    
    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get edges with optional filters"""
        edges = self._edge_cache
        
        if source_id:
            edges = [e for e in edges if e["source"] == source_id]
        
        if target_id:
            edges = [e for e in edges if e["target"] == target_id]
        
        if relation:
            edges = [e for e in edges if e["relation"] == relation]
        
        return edges
    
    def delete_edge(
        self,
        source_id: str,
        target_id: str,
        relation: Optional[str] = None
    ) -> int:
        """Delete edge(s) matching the criteria. Returns count deleted."""
        initial_count = len(self._edge_cache)
        
        if relation:
            self._edge_cache = [
                e for e in self._edge_cache
                if not (e["source"] == source_id and e["target"] == target_id and e["relation"] == relation)
            ]
        else:
            self._edge_cache = [
                e for e in self._edge_cache
                if not (e["source"] == source_id and e["target"] == target_id)
            ]
        
        deleted_count = initial_count - len(self._edge_cache)
        
        if deleted_count > 0:
            self._index["edge_count"] = len(self._edge_cache)
            self._save_edges()
            self._save_index()
        
        return deleted_count
    
    # ==================== Patient Operations ====================
    
    def get_patient_node_id(self, patient_id: str) -> Optional[str]:
        """Get the node ID for a patient"""
        return self._index.get("patients", {}).get(patient_id)
    
    def list_patients(self) -> Dict[str, str]:
        """Get all registered patients"""
        return self._index.get("patients", {})
    
    # ==================== Backup/Recovery ====================
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """Create a backup of the entire graph store"""
        backup_name = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir = self.base_dir / "backups" / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all data
        import shutil
        
        # Copy nodes
        nodes_backup = backup_dir / "nodes"
        if self.nodes_dir.exists():
            shutil.copytree(self.nodes_dir, nodes_backup)
        
        # Copy edges
        if self.edges_file.exists():
            shutil.copy(self.edges_file, backup_dir / "edges.json")
        
        # Copy index
        if self.index_file.exists():
            shutil.copy(self.index_file, backup_dir / "index.json")
        
        # Save backup metadata
        metadata = {
            "backup_name": backup_name,
            "created_at": datetime.now().isoformat(),
            "node_count": self._index.get("node_count", 0),
            "edge_count": self._index.get("edge_count", 0),
            "patient_count": len(self._index.get("patients", {}))
        }
        
        with open(backup_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return str(backup_dir)
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore from a backup"""
        backup_dir = self.base_dir / "backups" / backup_name
        
        if not backup_dir.exists():
            return False
        
        import shutil
        
        # Clear current data
        if self.nodes_dir.exists():
            shutil.rmtree(self.nodes_dir)
        
        # Restore nodes
        nodes_backup = backup_dir / "nodes"
        if nodes_backup.exists():
            shutil.copytree(nodes_backup, self.nodes_dir)
        else:
            self.nodes_dir.mkdir()
        
        # Restore edges
        edges_backup = backup_dir / "edges.json"
        if edges_backup.exists():
            shutil.copy(edges_backup, self.edges_file)
        
        # Restore index
        index_backup = backup_dir / "index.json"
        if index_backup.exists():
            shutil.copy(index_backup, self.index_file)
        
        # Reload caches
        self._node_cache.clear()
        self._load_index()
        self._load_edges()
        
        return True
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        backup_base = self.base_dir / "backups"
        
        if backup_base.exists():
            for backup_dir in backup_base.iterdir():
                if backup_dir.is_dir():
                    metadata_file = backup_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            backups.append(json.load(f))
        
        return sorted(backups, key=lambda x: x.get("created_at", ""), reverse=True)
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics"""
        # Calculate storage size
        total_size = 0
        for f in self.base_dir.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size
        
        return {
            "base_dir": str(self.base_dir),
            "node_count": self._index.get("node_count", 0),
            "edge_count": self._index.get("edge_count", 0),
            "patient_count": len(self._index.get("patients", {})),
            "storage_size_bytes": total_size,
            "storage_size_mb": round(total_size / (1024 * 1024), 2),
            "last_updated": self._index.get("last_updated"),
            "backup_count": len(self.list_backups())
        }
    
    def clear(self) -> None:
        """Clear all data (use with caution!)"""
        import shutil
        
        # Clear nodes
        if self.nodes_dir.exists():
            shutil.rmtree(self.nodes_dir)
            self.nodes_dir.mkdir()
        
        # Clear edges
        self._edge_cache = []
        self._save_edges()
        
        # Reset index
        self._index = {
            "patients": {},
            "node_count": 0,
            "edge_count": 0,
            "last_updated": None
        }
        self._save_index()
        
        # Clear cache
        self._node_cache.clear()
