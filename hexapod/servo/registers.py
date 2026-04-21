# ST3020 / ST3215 register map (SCS protocol, little-endian)

BROADCAST_ID = 0xFE

# Instructions
INST_PING        = 0x01
INST_READ        = 0x02
INST_WRITE       = 0x03
INST_REG_WRITE   = 0x04
INST_REG_ACTION  = 0x05
INST_SYNC_READ   = 0x82
INST_SYNC_WRITE  = 0x83

# Baud rate codes (written to BAUD_RATE register)
BAUD_1M     = 0
BAUD_500K   = 1
BAUD_250K   = 2
BAUD_128K   = 3
BAUD_115200 = 4
BAUD_76800  = 5
BAUD_57600  = 6
BAUD_38400  = 7

# EPROM — read only
MODEL_L = 3
MODEL_H = 4

# EPROM — read / write
ID            = 5
BAUD_RATE     = 6
MIN_ANGLE_L   = 9
MIN_ANGLE_H   = 10
MAX_ANGLE_L   = 11
MAX_ANGLE_H   = 12
CW_DEAD       = 26
CCW_DEAD      = 27
OFS_L         = 31
OFS_H         = 32
MODE          = 33

# SRAM — read / write
TORQUE_ENABLE  = 40
ACC            = 41
GOAL_POS_L     = 42
GOAL_POS_H     = 43
GOAL_TIME_L    = 44
GOAL_TIME_H    = 45
GOAL_SPEED_L   = 46
GOAL_SPEED_H   = 47
TORQUE_LIMIT_L = 48
TORQUE_LIMIT_H = 49
LOCK           = 55

# SRAM — read only
PRESENT_POS_L   = 56
PRESENT_POS_H   = 57
PRESENT_SPD_L   = 58
PRESENT_SPD_H   = 59
PRESENT_LOAD_L  = 60
PRESENT_LOAD_H  = 61
PRESENT_VOLTAGE = 62
PRESENT_TEMP    = 63
MOVING          = 66
PRESENT_CUR_L   = 69
PRESENT_CUR_H   = 70

# Error bits in status packets
ERR_VOLTAGE  = 0x01
ERR_ANGLE    = 0x02
ERR_OVERHEAT = 0x04
ERR_OVERELE  = 0x08
ERR_OVERLOAD = 0x20
