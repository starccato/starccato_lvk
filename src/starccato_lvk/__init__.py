import logging

# Mute fontTools INFO logs by setting its level to WARNING or higher
logging.getLogger('fontTools').setLevel(logging.WARNING)