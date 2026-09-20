"""Frozen experiment constants, ported verbatim from pipeline.py SECTIONS 2-4."""

EXPECTED_COLUMNS = (
    "FlowID", "FlowDuration/s", "SrcAddr", "SrcPort", "DstAddr", "DstPort",
    "Protocol", "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes",
    "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s", "RxByteRate/s",
    "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps", "MeanPacketSize",
    "PacketDropRate", "AverageHopCount", "label",
)
EXPECTED_CLASSES = (
    "Normal Traffic", "Blackhole Attack", "Wormhole Attack",
    "Sybil Attack", "Flooding Attack",
)
DROP_COLUMNS = ("FlowID", "Protocol")
LABEL_COLUMN = "label"
KNOWN_PORTS = (9, 654)

C2_FEATURES = (
    "SrcAddr_last_octet", "SrcPort_9", "SrcPort_654",
    "DstAddr_last_octet", "DstPort_9", "DstPort_654", "FlowDuration/s",
)
NETWORK_FEATURES = (
    "TxPackets", "RxPackets", "LostPackets", "TxBytes", "RxBytes",
    "TxPacketRate/s", "RxPacketRate/s", "TxByteRate/s", "RxByteRate/s",
    "MeanPacketSize",
)
PHYSICAL_FEATURES = (
    "MeanDelay/s", "MeanJitter/s", "Throughput/Kbps",
    "PacketDropRate", "AverageHopCount",
)
MANIFOLDS = {"c2": C2_FEATURES, "network": NETWORK_FEATURES, "physical": PHYSICAL_FEATURES}

REFERENCE_CLOUD_SIZE = 500
MAX_EDGE_PERCENTILE = 25
MAX_HOM_DIM = {"c2": 2, "network": 2, "physical": 1}
SPARSE_RIPS_EPSILON = {"c2": 0.5, "network": 0.5, "physical": None}
THRESHOLD_PERCENTILE = 95

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DEBUG_SAMPLE_ROWS = 5000

PRIMARY_SEED = 42
PROBE_SEEDS = (42, 7, 123)          # paper §IV; NOT the monolith's SEEDS
SUPERVISED_SEEDS = (42, 7, 2024)    # monolith's supervised tables
PROBE_PER_CLASS = 200
PROBE_TOP_K = 50
PROBE_DELTA = 0.2
