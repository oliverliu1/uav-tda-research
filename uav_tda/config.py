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

# Persistence images.
PI_RESOLUTION = (20, 20)

# Names of the eight summary statistics extracted per (manifold, dim).
SUMMARY_STAT_NAMES = (
    "count", "mean_persistence", "std_persistence", "max_persistence",
    "total_persistence", "mean_birth", "mean_death", "persistence_entropy",
)

# Hyperparameter grids (Phase 5).
LR_C_GRID = (0.1, 1.0, 10.0)
RF_N_ESTIMATORS_GRID = (100, 300)
RF_MAX_DEPTH_GRID = (10, 20, None)
SVM_C_GRID = (0.1, 1.0, 10.0)
SVM_GAMMA_GRID = ("scale", "auto")
GRID_CV_FOLDS = 3

# Feature sets and model names used by the supervised pipeline.
FEATURE_SETS = ("original", "summary_only", "summary_plus_images", "combined")
MODEL_NAMES = ("logreg", "rf", "svm")

# SVC with RBF kernel scales O(N^2) to O(N^3); at N=85k it is intractable.
# Train SVM on a stratified subsample of this size (set to None to disable).
SVM_MAX_TRAIN_ROWS = 5000

# Curated RF feature subset (Phase 5, item 14).
TOP_K_RF_IMPORTANCE = 30
TOP_K_MUTUAL_INFO = 30

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

# Paper's three seeds first, then seven sequential seeds, for the
# manuscript's ten-seed bootstrap-CI evaluation (Phase 4).
MANUSCRIPT_SEEDS = (42, 7, 123, 0, 1, 2, 3, 4, 5, 6)
