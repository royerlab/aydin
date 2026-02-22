#!/bin/bash
# =============================================================================
# Xpra startup script for Aydin Studio (Docker GUI target)
#
# Launches Aydin Studio inside a virtual X11 display and exposes it via
# Xpra's HTML5 client. Users access the GUI at http://localhost:${XPRA_PORT}
#
# Environment variables (with defaults):
#   XPRA_PORT   — Port for the HTML5 web client (default: 9876)
#   XPRA_SCREEN — Virtual screen resolution (default: 1920x1080x24+32)
#   DISPLAY     — X11 display number (default: :100)
# =============================================================================

set -e

export XPRA_PORT="${XPRA_PORT:-9876}"
export XPRA_SCREEN="${XPRA_SCREEN:-1920x1080x24+32}"
export DISPLAY="${DISPLAY:-:100}"

echo "=============================================="
echo "  Aydin Studio (Docker)"
echo "=============================================="
echo ""
echo "  Open in your browser:"
echo ""
echo "    http://localhost:${XPRA_PORT}"
echo ""
echo "  Resolution: ${XPRA_SCREEN%%x*}x$(echo ${XPRA_SCREEN} | cut -dx -f2)"
echo "  Press Ctrl+C to stop."
echo "=============================================="
echo ""

exec xpra start "${DISPLAY}" \
    --bind-tcp=0.0.0.0:"${XPRA_PORT}" \
    --html=on \
    --start-child="aydin" \
    --exit-with-children=yes \
    --daemon=no \
    --xvfb="/usr/bin/Xvfb ${DISPLAY} +extension Composite -screen 0 ${XPRA_SCREEN} -nolisten tcp -noreset" \
    --pulseaudio=no \
    --notifications=no \
    --bell=no \
    --mdns=no \
    --webcam=no
