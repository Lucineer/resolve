"""fleet_equipment.py — Shared equipment types ported from cuda-equipment (Rust)

Every vessel in the Lucineer fleet shares these common types:
- Confidence: universal 0-1 certainty propagation
- Tile/TileGrid: tiling patterns for weights, pheromones, thermal, FPGA
- FleetMessage: A2A communication protocol
- Agent protocol: deliberative agent interface
- EquipmentRegistry: sensor/actuator capabilities

This module bridges cuda-equipment (Rust) with resolve (Python).
"""

from __future__ import annotations
from typing import Any, Optional, List, Dict, Tuple, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy


# ============================================================
# CONFIDENCE — universal 0-1 certainty (mirrors cuda-equipment Confidence)
# ============================================================

class Confidence:
    """Universal confidence value. Every value in the fleet carries this.
    Ported from cuda-equipment/src/lib.rs Confidence struct."""
    
    __slots__ = ('_value',)
    
    ZERO = None      # set after class definition
    SURE = None
    HALF = None
    LIKELY = None
    UNLIKELY = None
    
    def __init__(self, value: float = 0.5):
        self._value = max(0.0, min(1.0, float(value)))
    
    @property
    def value(self) -> float:
        return self._value
    
    def combine(self, other: Confidence) -> Confidence:
        """Bayesian combination of independent confidences: 1/(1/a + 1/b)"""
        a, b = self._value, other._value
        if a <= 0.0: return Confidence(b)
        if b <= 0.0: return Confidence(a)
        return Confidence(1.0 / (1.0/a + 1.0/b))
    
    def chain(self, other: Confidence) -> Confidence:
        """Sequential: confidence that both are true."""
        return Confidence(self._value * other._value)
    
    def weighted(self, other: Confidence, w_self: float, w_other: float) -> Confidence:
        """Weighted average."""
        total = w_self + w_other
        if total <= 0.0: return Confidence.ZERO
        return Confidence((self._value * w_self + other._value * w_other) / total)
    
    def discount(self, factor: float) -> Confidence:
        """Discount confidence (entropy)."""
        return Confidence(self._value * max(0.0, min(1.0, factor)))
    
    def decay(self, rounds: int, rate: float = 0.95) -> Confidence:
        """Multi-round decay."""
        return Confidence(self._value * (rate ** rounds))
    
    @property
    def is_certain(self) -> bool: return self._value >= 0.95
    @property
    def is_likely(self) -> bool: return self._value >= 0.5
    @property
    def is_uncertain(self) -> bool: return self._value < 0.3
    
    def to_bits(self) -> int:
        return int(self._value * 255.0 + 0.5)
    
    @classmethod
    def from_bits(cls, b: int) -> Confidence:
        return Confidence(b / 255.0)
    
    def __repr__(self) -> str: return f"Confidence({self._value:.3f})"
    def __str__(self) -> str: return f"{self._value * 100:.1f}%"
    def __float__(self) -> float: return self._value
    def __eq__(self, other) -> bool:
        if isinstance(other, Confidence): return abs(self._value - other._value) < 1e-9
        return NotImplemented
    def __lt__(self, other) -> bool:
        if isinstance(other, (int, float)): return self._value < other
        if isinstance(other, Confidence): return self._value < other._value
        return NotImplemented
    def __le__(self, other) -> bool:
        return self == other or self < other
    def __gt__(self, other) -> bool:
        if isinstance(other, (int, float)): return self._value > other
        if isinstance(other, Confidence): return self._value > other._value
        return NotImplemented
    def __ge__(self, other) -> bool:
        return self == other or self > other

# Set class constants after class definition
Confidence.ZERO = Confidence(0.0)
Confidence.SURE = Confidence(1.0)
Confidence.HALF = Confidence(0.5)
Confidence.LIKELY = Confidence(0.75)
Confidence.UNLIKELY = Confidence(0.25)


# ============================================================
# TILE — rectangular data chunk (mirrors cuda-equipment Tile<T>)
# ============================================================

class TileId:
    """Unique tile identifier."""
    __slots__ = ('_id',)
    def __init__(self, id_: int): self._id = id_
    def __repr__(self) -> str: return f"TileId({self._id})"
    def __eq__(self, other) -> bool: return isinstance(other, TileId) and self._id == other._id
    def __hash__(self) -> int: return hash(self._id)
    def __int__(self) -> int: return self._id


class Tile:
    """A rectangular chunk of data — weights, pheromones, thermal values, FPGA BRAM.
    Ported from cuda-equipment Tile<T>."""
    
    def __init__(self, id_: TileId, row: int, col: int, rows: int, cols: int,
                 data: Optional[List[Any]] = None, confidence: Optional[Confidence] = None):
        self.id = id_
        self.row = row
        self.col = col
        self.rows = rows
        self.cols = cols
        self.data = data if data is not None else [None] * (rows * cols)
        self.confidence = confidence or Confidence.SURE
        self.last_accessed = 0
    
    @property
    def width(self) -> int: return self.cols
    @property
    def height(self) -> int: return self.rows
    @property
    def size(self) -> int: return len(self.data)
    @property
    def is_empty(self) -> bool: return not self.data
    
    def get(self, r: int, c: int) -> Optional[Any]:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.data[r * self.cols + c]
        return None
    
    def set(self, r: int, c: int, value: Any) -> bool:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.data[r * self.cols + c] = value
            self.last_accessed = _now()
            return True
        return False
    
    def touches(self, other: Tile) -> bool:
        return (self.row < other.row + other.rows and other.row < self.row + self.rows
                and self.col < other.col + other.cols and other.col < self.col + self.cols)
    
    def region_slice(self, r_start: int, c_start: int, r_end: int, c_end: int) -> List[Any]:
        """Extract a sub-region as flat list."""
        result = []
        for r in range(max(0, r_start), min(self.rows, r_end)):
            for c in range(max(0, c_start), min(self.cols, c_end)):
                result.append(self.data[r * self.cols + c])
        return result
    
    def __repr__(self) -> str:
        return f"Tile(id={self.id}, pos=({self.row},{self.col}), size={self.rows}x{self.cols})"


class TileGrid:
    """Manages a tiled space — shared by weight streaming, FPGA memory, swarm pheromones.
    Ported from cuda-equipment TileGrid<T>."""
    
    def __init__(self, total_rows: int, total_cols: int, tile_rows: int, tile_cols: int):
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.grid_rows = (total_rows + tile_rows - 1) // tile_rows
        self.grid_cols = (total_cols + tile_cols - 1) // tile_cols
        self.tiles: List[Tile] = []
        next_id = 0
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                actual_rows = min(tile_rows, total_rows - r * tile_rows)
                actual_cols = min(tile_cols, total_cols - c * tile_cols)
                self.tiles.append(Tile(TileId(next_id), r, c, actual_rows, actual_cols))
                next_id += 1
    
    @property
    def total_tiles(self) -> int: return len(self.tiles)
    
    def get_tile(self, grid_row: int, grid_col: int) -> Optional[Tile]:
        if 0 <= grid_row < self.grid_rows and 0 <= grid_col < self.grid_cols:
            return self.tiles[grid_row * self.grid_cols + grid_col]
        return None
    
    def tiles_in_region(self, row: int, col: int, rows: int, cols: int) -> List[Tile]:
        gr_start = row // self.tile_rows
        gc_start = col // self.tile_cols
        gr_end = (row + rows) // self.tile_rows + 1
        gc_end = (col + cols) // self.tile_cols + 1
        result = []
        for gr in range(gr_start, min(gr_end, self.grid_rows)):
            for gc in range(gc_start, min(gc_end, self.grid_cols)):
                t = self.get_tile(gr, gc)
                if t: result.append(t)
        return result
    
    def lru_order(self) -> List[Tile]:
        return sorted(self.tiles, key=lambda t: t.last_accessed)
    
    def lru_evict(self, count: int) -> List[Tile]:
        """Get the `count` least recently used tiles."""
        return self.lru_order()[:count]
    
    def __repr__(self) -> str:
        return f"TileGrid({self.grid_rows}x{self.grid_cols}, {self.total_tiles} tiles)"


# ============================================================
# FLEET MESSAGE — A2A protocol (mirrors cuda-equipment FleetMessage)
# ============================================================

class VesselId:
    __slots__ = ('_id',)
    def __init__(self, id_: int): self._id = id_
    def __repr__(self) -> str: return f"VesselId({self._id})"
    def __eq__(self, other) -> bool: return isinstance(other, VesselId) and self._id == other._id
    def __hash__(self) -> int: return hash(self._id)


class MessageType(Enum):
    CONSIDER = "consider"
    RESOLVE = "resolve"
    FORFEIT = "forfeit"
    CAPABILITY_QUERY = "capability_query"
    CAPABILITY_RESPONSE = "capability_response"
    PING = "ping"
    PONG = "pong"
    TILE_TRANSFER = "tile_transfer"
    CONFIDENCE_UPDATE = "confidence_update"


class FleetMessage:
    """A2A message between fleet vessels. Ported from cuda-equipment FleetMessage."""
    
    _next_id = 0
    
    def __init__(self, from_vessel: VesselId, to_vessel: VesselId,
                 msg_type: MessageType, payload: Any = None,
                 confidence: Optional[Confidence] = None,
                 ttl: int = 5, in_reply_to: Optional[int] = None):
        FleetMessage._next_id += 1
        self.id = FleetMessage._next_id
        self.from_vessel = from_vessel
        self.to_vessel = to_vessel
        self.msg_type = msg_type
        self.payload = payload
        self.confidence = confidence or Confidence.SURE
        self.timestamp = _now()
        self.ttl = ttl
        self.in_reply_to = in_reply_to
    
    def reply(self, msg_type: MessageType, payload: Any = None) -> FleetMessage:
        return FleetMessage(self.to_vessel, self.from_vessel, msg_type, payload,
                           self.confidence, in_reply_to=self.id)
    
    @property
    def is_expired(self) -> bool: return self.ttl <= 0
    
    def decrement_ttl(self): self.ttl = max(0, self.ttl - 1)
    
    def __repr__(self) -> str:
        return f"Msg({self.id}: {self.from_vessel}->{self.to_vessel} {self.msg_type.value})"


# ============================================================
# AGENT PROTOCOL — deliberative agent interface
# ============================================================

@runtime_checkable
class Agent(Protocol):
    """Every deliberative entity in the fleet implements this.
    Ported from cuda-equipment Agent trait."""
    
    @property
    def id(self) -> VesselId: ...
    @property
    def name(self) -> str: ...
    def receive(self, msg: FleetMessage) -> List[FleetMessage]: ...
    def capabilities(self) -> List[str]: ...
    def self_confidence(self) -> Confidence: ...
    def is_healthy(self) -> bool: ...


class BaseAgent:
    """Minimally functional agent — the engine every boat has.
    Ported from cuda-equipment BaseAgent."""
    
    def __init__(self, id_: int, name: str, capabilities: Optional[List[str]] = None):
        self._id = VesselId(id_)
        self._name = name
        self._confidence = Confidence.HALF
        self._capabilities = capabilities or []
        self.messages_sent = 0
        self.messages_received = 0
        self._inbox: List[FleetMessage] = []
    
    @property
    def id(self) -> VesselId: return self._id
    @property
    def name(self) -> str: return self._name
    
    def receive(self, msg: FleetMessage) -> List[FleetMessage]:
        self.messages_received += 1
        self._inbox.append(msg)
        
        if msg.msg_type == MessageType.PING:
            self.messages_sent += 1
            return [msg.reply(MessageType.PONG)]
        elif msg.msg_type == MessageType.CAPABILITY_QUERY:
            self.messages_sent += 1
            return [msg.reply(MessageType.CAPABILITY_RESPONSE, ",".join(self._capabilities))]
        elif msg.msg_type == MessageType.CONFIDENCE_UPDATE:
            if isinstance(msg.payload, Confidence):
                self._confidence = self._confidence.combine(msg.payload)
            return []
        return []
    
    def capabilities(self) -> List[str]: return list(self._capabilities)
    def self_confidence(self) -> Confidence: return self._confidence
    def is_healthy(self) -> bool: return self._confidence.is_likely
    
    def send_consider(self, to: VesselId, proposal_id: int) -> FleetMessage:
        msg = FleetMessage(self._id, to, MessageType.CONSIDER, {"proposal_id": proposal_id})
        self.messages_sent += 1
        return msg
    
    def send_resolve(self, to: VesselId, proposal_id: int, accepted: bool) -> FleetMessage:
        msg = FleetMessage(self._id, to, MessageType.RESOLVE,
                          {"proposal_id": proposal_id, "accepted": accepted})
        self.messages_sent += 1
        return msg
    
    def send_forfeit(self, to: VesselId, proposal_id: int, reason: str) -> FleetMessage:
        msg = FleetMessage(self._id, to, MessageType.FORFEIT,
                          {"proposal_id": proposal_id, "reason": reason})
        self.messages_sent += 1
        return msg


# ============================================================
# EQUIPMENT REGISTRY — what a vessel can perceive and act upon
# ============================================================

class SensorType(Enum):
    CAMERA = "camera"
    THERMAL = "thermal"
    LIDAR = "lidar"
    AUDIO = "audio"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"
    PRESSURE = "pressure"
    HUMIDITY = "humidity"
    LIGHT = "light"
    PROXIMITY = "proximity"
    TOUCH = "touch"
    GPS = "gps"
    RF = "rf"
    CHEMICAL = "chemical"


class ActuatorType(Enum):
    MOTOR = "motor"
    SERVO = "servo"
    LINEAR = "linear"
    STEPPER = "stepper"
    RELAY = "relay"
    SPEAKER = "speaker"
    DISPLAY = "display"
    LED = "led"
    VALVE = "valve"
    PUMP = "pump"
    HEATER = "heater"
    COOLER = "cooler"


@dataclass
class Sensor:
    name: str
    sensor_type: SensorType
    resolution: int
    confidence: Confidence = field(default_factory=lambda: Confidence.SURE)


@dataclass
class Actuator:
    name: str
    actuator_type: ActuatorType
    max_force: float = 0.0
    max_speed: float = 0.0


class EquipmentRegistry:
    """What sensors and actuators a vessel has.
    Ported from cuda-equipment EquipmentRegistry."""
    
    def __init__(self, vessel_id: int):
        self.vessel_id = VesselId(vessel_id)
        self.sensors: List[Sensor] = []
        self.actuators: List[Actuator] = []
        self.compute_units: int = 1
        self.memory_bytes: int = 0
    
    def add_sensor(self, name: str, stype: SensorType, resolution: int) -> EquipmentRegistry:
        self.sensors.append(Sensor(name, stype, resolution))
        return self
    
    def add_actuator(self, name: str, atype: ActuatorType,
                     max_force: float = 0.0, max_speed: float = 0.0) -> EquipmentRegistry:
        self.actuators.append(Actuator(name, atype, max_force, max_speed))
        return self
    
    def has_sensor_type(self, stype: SensorType) -> bool:
        return any(s.sensor_type == stype for s in self.sensors)
    
    def has_actuator_type(self, atype: ActuatorType) -> bool:
        return any(a.actuator_type == atype for a in self.actuators)
    
    def character_sheet(self) -> Dict[str, Any]:
        return {
            "vessel_id": self.vessel_id._id,
            "sensors": [{"name": s.name, "type": s.sensor_type.value, "resolution": s.resolution} for s in self.sensors],
            "actuators": [{"name": a.name, "type": a.actuator_type.value, "force": a.max_force, "speed": a.max_speed} for a in self.actuators],
            "compute_units": self.compute_units,
            "memory_bytes": self.memory_bytes,
        }


# ============================================================
# TILE SCHEDULER — shared scheduling for weight/FPGA/swarm tiles
# ============================================================

@dataclass
class ScheduledTile:
    tile_id: TileId
    layer: int
    start_cycle: int
    end_cycle: int
    bram_slot: int


class TileScheduler:
    """Shared tile scheduling logic. Ported from cuda-equipment TileScheduler."""
    
    def __init__(self, max_concurrent: int = 4, bandwidth_bytes_per_cycle: float = 32.0,
                 latency_cycles: int = 10):
        self.max_concurrent = max_concurrent
        self.bandwidth = bandwidth_bytes_per_cycle
        self.latency = latency_cycles
    
    def load_time_cycles(self, tile_bytes: int) -> int:
        transfer = int(tile_bytes / self.bandwidth) + 1
        return transfer + self.latency
    
    def schedule_layer(self, grid: TileGrid, layer: int, bram_slots: int) -> List[ScheduledTile]:
        schedule = []
        current_cycle = 0
        active = 0
        for tile in grid.tiles:
            if active >= bram_slots:
                current_cycle += self.load_time_cycles(4096)
                active = 0
            load_time = self.load_time_cycles(len(tile.data) * 4)
            schedule.append(ScheduledTile(tile.id, layer, current_cycle,
                                         current_cycle + load_time, active))
            current_cycle += 2
            active += 1
        return schedule


# ============================================================
# INTERNALS
# ============================================================

_cycle_counter = [0]

def _now() -> int:
    _cycle_counter[0] += 1
    return _cycle_counter[0]


# ============================================================
# INTEGRATION: Bridge to resolve system
# ============================================================

def payload_to_confidence(value: float) -> Confidence:
    """Convert a resolve Payload confidence to fleet Confidence."""
    return Confidence(value)

def confidence_to_payload(conf: Confidence) -> float:
    """Convert fleet Confidence back to resolve Payload confidence."""
    return conf.value

def payload_to_fleet_message(payload_data: dict, from_id: int, to_id: int) -> FleetMessage:
    """Convert a resolve Payload dict to a FleetMessage."""
    msg_type_str = payload_data.get("type", "ping")
    type_map = {
        "consider": MessageType.CONSIDER, "resolve": MessageType.RESOLVE,
        "forfeit": MessageType.FORFEIT, "ping": MessageType.PING,
        "pong": MessageType.PONG, "capability_query": MessageType.CAPABILITY_QUERY,
        "confidence_update": MessageType.CONFIDENCE_UPDATE,
    }
    msg_type = type_map.get(msg_type_str, MessageType.PING)
    conf = Confidence(payload_data.get("confidence", 1.0))
    return FleetMessage(VesselId(from_id), VesselId(to_id), msg_type,
                       payload_data.get("payload"), conf)

def fleet_message_to_payload(msg: FleetMessage) -> dict:
    """Convert a FleetMessage back to a resolve Payload dict."""
    return {
        "type": msg.msg_type.value,
        "from": msg.from_vessel._id,
        "to": msg.to_vessel._id,
        "payload": msg.payload,
        "confidence": msg.confidence.value,
        "timestamp": msg.timestamp,
    }


if __name__ == "__main__":
    # Quick smoke test
    c1 = Confidence(0.8)
    c2 = Confidence(0.6)
    print(f"Combine: {c1.combine(c2)}")  # ~0.343
    
    grid = TileGrid(100, 100, 32, 32)
    print(f"Grid: {grid}")
    
    agent = BaseAgent(1, "test", ["thinking", "sensing"])
    ping = FleetMessage(VesselId(0), VesselId(1), MessageType.PING)
    resp = agent.receive(ping)
    print(f"Response: {resp}")
    
    eq = EquipmentRegistry(1).add_sensor("cam", SensorType.CAMERA, 1920)
    print(f"Has camera: {eq.has_sensor_type(SensorType.CAMERA)}")
    print(f"Character sheet: {eq.character_sheet()}")
    
    sched = TileScheduler()
    tiles = sched.schedule_layer(grid, 0, 4)
    print(f"Scheduled {len(tiles)} tiles")
    
    print("\nAll fleet equipment tests passed!")
