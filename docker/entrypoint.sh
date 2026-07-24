#!/usr/bin/env bash
# Injects the baked-in model paths as defaults and forwards everything
# else to sam3-instructsam-cli. Any explicit --lm / --mmproj / --grounding
# / --mask-queries in the caller's args overrides the corresponding
# default (later args win in the CLI's arg loop, so we prepend defaults).
set -euo pipefail

# Only inject defaults if the caller didn't already pass them.
prepend=()
if [[ " $* " != *" --lm "* ]];             then prepend+=(--lm "${SAM3_LM}"); fi
if [[ " $* " != *" --mmproj "* ]];         then prepend+=(--mmproj "${SAM3_MMPROJ}"); fi
if [[ " $* " != *" --grounding "* ]];      then prepend+=(--grounding "${SAM3_GROUNDING}"); fi
if [[ " $* " != *" --mask-queries "* ]];   then prepend+=(--mask-queries "${SAM3_MASK_QUERIES}"); fi

exec sam3-instructsam-cli "${prepend[@]}" "$@"
