# Sourced by every launch script that starts a paid box's work (GH200 at lambda rates):
#   . "$(dirname "${BASH_SOURCE[0]}")/require_driver.sh"
#
# tools/campaign_driver.py exports CAMPAIGN_DRIVER_RUN (its DRIVER_ENV) into every unit's
# environment. A script run without it was started by hand, outside the driver's go/no-go,
# per-phase caps, budget clock and state file, so it refuses: exit 78 (EX_CONFIG).
# One owner of the refusal; the launch scripts do not carry their own copy.
if [ -z "${CAMPAIGN_DRIVER_RUN:-}" ]; then
  printf '%s: refused: CAMPAIGN_DRIVER_RUN is not set, so this was not started by tools/campaign_driver.py.\n' "${0##*/}" >&2
  printf 'A paid box runs through the driver (go/no-go, caps, budget clock, state): put this as a unit in a campaign config and run tools/campaign_driver.py.\n' >&2
  exit 78
fi
