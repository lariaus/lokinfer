#!/bin/sh
# Builds the iOS test host, installs it on the connected iPhone and runs the
# Catch2 suite on device, streaming the console. Extra args go to Catch2.
# Usage: scripts/ios_test.sh [catch2 args...]
set -e
cd "$(dirname "$0")/.."
UDID=$(xcrun devicectl list devices 2>/dev/null | grep physical | grep -v unavailable \
  | grep -oE '[0-9A-F]{8}-[0-9A-F]{16}' | head -1)
[ -n "$UDID" ] || { echo "no connected physical iPhone" >&2; exit 2; }
BUILD=build/ios-xcode
BUNDLE=$(scripts/lk-config ios_bundle_id)
cmake -S . -B $BUILD -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 >/dev/null
xcodebuild -project $BUILD/lokinfer.xcodeproj -scheme LokiHost -configuration Release \
  -destination "id=$UDID" -allowProvisioningUpdates -quiet build
APP=$(find $BUILD -path "*Release-iphoneos/LokiHost.app" -maxdepth 4 | head -1)
xcrun devicectl device install app --device "$UDID" "$APP" >/dev/null
xcrun devicectl device process launch --device "$UDID" --console --terminate-existing \
  "$BUNDLE" -- --run-tests "$@"
