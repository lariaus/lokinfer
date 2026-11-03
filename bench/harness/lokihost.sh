#!/bin/sh
# Relaunch LokiHost (the iPhone remote executor) with runtime-variant environment variables, and restore the
# standard watchdog afterwards (the same loop scripts/ios_deploy.sh starts, server mode only). Run inside the phone
# lock. docs/benchmarking.md "CPU runtime variants".
#   bench/harness/lokihost.sh launch [VAR=VALUE ...]   # kills the watchdog, launches --serve with these vars
#   bench/harness/lokihost.sh run OUT.log ARGS...     # foreground: launch with ARGS (e.g. --bench-amx), console -> OUT.log,
#                                                      # returns when the app exits
#   bench/harness/lokihost.sh restore                  # standard watchdog (no variant vars)
cd "$(dirname "$0")/../.."
UDID=$(scripts/lk-config iphone_udid) || exit 2
BUNDLE=$(scripts/lk-config ios_bundle_id)
LOG=build/ios-console.log
mode=$1; shift
pkill -f "lokinfer-ios-watchdog" 2>/dev/null || true
pkill -f "devicectl device process launch.*lokinfer.testhost" 2>/dev/null || true
sleep 1
case $mode in
  launch)
    envs=""
    for kv in "$@"; do envs="$envs DEVICECTL_CHILD_$kv"; done
    echo "LOKIHOST: launch with$envs" >> $LOG
    nohup env $envs xcrun devicectl device process launch --device "$UDID" --console --terminate-existing \
      "$BUNDLE" -- --serve >> $LOG 2>&1 &
    ;;
  run)
    out=$1; shift
    echo "LOKIHOST: run $*" >> $LOG
    xcrun devicectl device process launch --device "$UDID" --console --terminate-existing \
      "$BUNDLE" -- "$@" > "$out" 2>&1
    ;;
  restore)
    echo "LOKIHOST: restore watchdog" >> $LOG
    nohup sh -c '
      # lokinfer-ios-watchdog
      while true; do
        xcrun devicectl device process launch --device "'"$UDID"'" --console --terminate-existing \
          '"$BUNDLE"' -- --serve >> '"$LOG"' 2>&1
        if grep -q "could not be, unlocked" '"$LOG"'; then echo "WATCHDOG: phone locked, stopping" >> '"$LOG"'; exit 0; fi
        echo "WATCHDOG: app exited, relaunching" >> '"$LOG"'
        sleep 1
      done' lokinfer-ios-watchdog > /dev/null 2>&1 &
    ;;
  *) echo "usage: $0 launch [VAR=VALUE ...] | run OUT.log ARGS... | restore" >&2; exit 2 ;;
esac
