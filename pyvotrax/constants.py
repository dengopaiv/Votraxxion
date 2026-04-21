"""Votrax SC-01A clock constants."""

MASTER_CLOCK = 720_000
SCLOCK = MASTER_CLOCK // 18   # 40000 Hz - analog sample rate
CCLOCK = SCLOCK // 2          # 20000 Hz - chip update rate
