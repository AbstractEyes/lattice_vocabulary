# crystal_lattice/specials.py

# First 1000 IDs are reserved for structural tokens.
# You can add/adjust slots any time; IDs remain stable.

SPECIAL_TOKEN_ORDER = [
    "<pad>",            # 0
    "<unk>",            # 1
    "<bos>",            # 2
    "<eos>",            # 3
    "<observer>",       # 4

    # Cardinals (leave room to expand cardinal systems)
    "<cardinal_0>",     # 5
    "<cardinal_1>",     # 6
    "<cardinal_2>",     # 7
    "<cardinal_3>",     # 8
    "<cardinal_4>",     # 9
    "<cardinal_5>",     # 10
    "<cardinal_6>",     # 11
    "<cardinal_7>",     # 12
    "<cardinal_8>",     # 13
    "<cardinal_9>",     # 14

    # Harmony channels
    "<analysis>",       # 15
    "<commentary>",     # 16
    "<final>",          # 17
    "<anchor_invariant>", # 18

    # Conversation roles / extra channels (optional, space reserved)
    "<system>",         # 19
    "<user>",           # 20
    "<assistant>",      # 21,
]

# Reserve the full band [0..999] for stable structural growth.
RESERVED_BAND_SIZE = 1000
