#!/bin/sh
# Builds + installs the iOS test host, launches it ONCE: it runs the Catch2
# suite, then stays up as the remote executor (keeps the phone awake).
# Console goes to build/ios-console.log; this script waits for the test
# result line and prints the summary. Needs the phone unlocked for the launch.
# A watchdog relaunches the app in server mode if it dies (stop it with
# `pkill -f lokinfer-ios-watchdog`).
# Usage: scripts/ios_deploy.sh [catch2 args...]
set -e
cd "$(dirname "$0")/.."
# While this flag exists the phone is unavailable: nothing may touch it.
if [ -e /tmp/lokinfer-phone-disabled ]; then echo "iPhone disabled ($(cat /tmp/lokinfer-phone-disabled)); not deploying" >&2; exit 3; fi
UDID=$(xcrun devicectl list devices 2>/dev/null | grep physical | grep -v unavailable \
  | grep -oE '[0-9A-F]{8}-[0-9A-F]{16}' | head -1)
[ -n "$UDID" ] || { echo "no connected physical iPhone" >&2; exit 2; }
BUILD=build/ios-xcode
BUNDLE=$(scripts/lk-config ios_bundle_id)
# The build loads every Mac core: hold the Mac bench lock so it never lands inside a Mac timing run (2026-11-02:
# deploys during a Kokoro GPU bench made s6f read 70-195 ms). Lock order is always phone -> macbench.
scripts/lk-lock /tmp/lokinfer-macbench.lock sh -c "
cmake -S . -B $BUILD -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 >/dev/null
xcodebuild -project $BUILD/lokinfer.xcodeproj -scheme LokiHost -configuration Release \
  -destination 'id=$UDID' -allowProvisioningUpdates -quiet build 2>&1 | grep -E 'error' || true"
xcrun devicectl device install app --device "$UDID" $BUILD/Release-iphoneos/LokiHost.app >/dev/null
LOG=build/ios-console.log
pkill -f "lokinfer-ios-watchdog" 2>/dev/null || true
: > $LOG
pkill -f "devicectl device process launch.*lokinfer.testhost" 2>/dev/null || true
# Watchdog: if the app dies (crash in a kernel under test), relaunch it in
# server mode right away, while the phone is still unlocked from before.
# The first launch also runs the Catch2 suite.
nohup sh -c '
  # lokinfer-ios-watchdog
  first=1
  while true; do
    if [ $first = 1 ]; then args="--run-tests --serve"; first=0; else args="--serve"; fi
    xcrun devicectl device process launch --device "'"$UDID"'" --console --terminate-existing \
      '"$BUNDLE"' -- $args '"$*"' >> '"$LOG"' 2>&1
    if grep -q "could not be, unlocked" '"$LOG"'; then echo "WATCHDOG: phone locked, stopping" >> '"$LOG"'; exit 0; fi
    echo "WATCHDOG: app exited, relaunching" >> '"$LOG"'
    sleep 1
  done' lokinfer-ios-watchdog > /dev/null 2>&1 &
for i in $(seq 1 600); do
  if grep -q "LOKINFER_TESTS_EXIT_CODE" $LOG 2>/dev/null; then break; fi
  if grep -q "ERROR" $LOG 2>/dev/null; then cat $LOG; exit 1; fi
  sleep 1
done
grep -E "All tests passed|test cases:|assertions:|LOKINFER_TESTS_EXIT_CODE" $LOG
# Keep the console of a failing run (the next deploy truncates $LOG): the failing assertions are in it.
if grep -q "LOKINFER_TESTS_EXIT_CODE=[1-9]" $LOG; then
  keep=build/ios-console-fail-$(date +%Y%m%d-%H%M%S).log
  cp $LOG $keep
  grep -E "FAILED|failed|REQUIRE|CHECK" $LOG | head -20
  echo "failing run's console kept in $keep"
fi
echo "remote executor: LOKINFER_IPHONE=$UDID (host $UDID.coredevice.local:47800)"
