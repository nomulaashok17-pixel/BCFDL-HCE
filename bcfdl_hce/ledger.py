from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import time
import requests
from .utils import ensure_dir, sha256_bytes


@dataclass
class AuthenticationTransaction:
    tx_id: str
    client_id: str
    event: str
    model_hash: str
    content_id: str
    timestamp: float
    metadata: dict


@dataclass
class Block:
    index: int
    previous_hash: str
    timestamp: float
    leader: str
    term: int
    transactions: list[dict]
    nonce: int = 0
    block_hash: str = ""

    def finalize(self):
        payload = {k:v for k,v in asdict(self).items() if k != "block_hash"}
        self.block_hash = sha256_bytes(json.dumps(payload, sort_keys=True).encode())
        return self


class LimitedRaftCoordinator:
    """Deterministic crash-fault-oriented experimental coordinator for a permissioned ledger.

    This is a lightweight reproducibility emulator, not a production Raft implementation.
    It models leader election, majority quorum and leader replacement under crashes.
    """
    def __init__(self, nodes: int = 5):
        if nodes < 3: raise ValueError("At least 3 validator nodes required")
        self.nodes = [f"validator-{i}" for i in range(nodes)]
        self.alive = set(self.nodes)
        self.term = 1
        self.leader = self.nodes[0]
    @property
    def quorum(self): return len(self.nodes)//2 + 1
    def crash(self, node: str):
        self.alive.discard(node)
        if node == self.leader: self.elect()
    def recover(self, node: str): self.alive.add(node)
    def elect(self):
        if len(self.alive) < self.quorum: raise RuntimeError("No Limited-Raft majority quorum")
        self.term += 1
        self.leader = sorted(self.alive)[self.term % len(self.alive)]
        return self.leader
    def commit(self) -> tuple[str,int]:
        if self.leader not in self.alive: self.elect()
        if len(self.alive) < self.quorum: raise RuntimeError("Consensus unavailable: majority quorum lost")
        return self.leader, self.term


class PermissionedLedger:
    def __init__(self, directory: str | Path, validators: int = 5):
        self.dir = ensure_dir(directory)
        self.path = self.dir / "ledger.jsonl"
        self.raft = LimitedRaftCoordinator(validators)
        self.chain = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip(): self.chain.append(Block(**json.loads(line)))
        if not self.chain: self._append_block([])

    def _append_block(self, txs):
        leader, term = self.raft.commit()
        prev = self.chain[-1].block_hash if self.chain else "0"*64
        b = Block(len(self.chain), prev, time.time(), leader, term, txs).finalize()
        self.chain.append(b)
        with self.path.open("a", encoding="utf-8") as f: f.write(json.dumps(asdict(b))+"\n")
        return b
    def append(self, tx: AuthenticationTransaction): return self._append_block([asdict(tx)])
    def verify(self) -> bool:
        prev = "0"*64
        for i,b in enumerate(self.chain):
            if b.index != i or b.previous_hash != prev: return False
            old = b.block_hash
            copy = Block(**asdict(b)); copy.block_hash=""; copy.finalize()
            if old != copy.block_hash: return False
            prev = old
        return True


class LocalCAS:
    def __init__(self, directory: str | Path): self.dir=ensure_dir(directory)
    def add_bytes(self, data: bytes) -> str:
        cid=hashlib.sha256(data).hexdigest(); p=self.dir/cid
        if not p.exists(): p.write_bytes(data)
        return cid
    def get_bytes(self, cid: str) -> bytes: return (self.dir/cid).read_bytes()


class IPFSAdapter:
    def __init__(self, api_url: str): self.api_url=api_url.rstrip("/")
    def add_bytes(self, data: bytes) -> str:
        r=requests.post(self.api_url+"/api/v0/add", files={"file":("model.bin",data)}, timeout=30); r.raise_for_status()
        return r.json()["Hash"]
    def get_bytes(self, cid: str) -> bytes:
        r=requests.post(self.api_url+"/api/v0/cat", params={"arg":cid}, timeout=30); r.raise_for_status(); return r.content
