#!/usr/bin/env bash
# One-command live Twitch/Kick stream watcher & 9:16 Shorts generator
if [ -z "$1" ]; then
    echo "Usage: ./watch.sh <twitch_or_kick_channel> [options]"
    echo "       ./watch.sh westcol --kick"
    echo "       ./watch.sh https://kick.com/westcol"
    echo "       ./watch.sh tarik --min 10"
    exit 1
fi

python -m app.live.watcher "$@"
