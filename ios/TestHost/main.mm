// Minimal iOS test host: UIKit shell + worker thread running Catch2.
// Launch args (devicectl ... -- <args>) are forwarded to Catch2.
#import <UIKit/UIKit.h>
#include <catch2/catch_session.hpp>
#include <sys/utsname.h>

#include "lokinfer/remote/server.h"
#include "../../bench/kernels/amx_bench.h"
#include "../../bench/kernels/conv_bench.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

@interface AppDelegate : UIResponder <UIApplicationDelegate>
@property(strong, nonatomic) UIWindow* window;
@end

static int run_tests(NSArray<NSString*>* args) {
  NSString* lib = [[NSBundle mainBundle] pathForResource:@"lokinfer" ofType:@"metallib"];
  setenv("LOKINFER_METALLIB", lib.UTF8String, 1);
  NSString* docs = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES)[0];
  std::vector<std::string> storage;
  storage.push_back("lokinfer_tests");
  bool want_tests = false;
  for (NSUInteger i = 1; i < args.count; ++i) {
    std::string a = args[i].UTF8String;
    if (a == "--run-tests") { want_tests = true; continue; }
    if (a == "--") continue;  // devicectl forwards its separator
    if (a == "--serve") continue;
    storage.push_back(a);
  }
  if (!want_tests) return -1;
  // Always also write a JUnit report into Documents/ for `devicectl device copy from`.
  storage.push_back("--reporter");
  storage.push_back("console::out=-::colour-mode=none");
  storage.push_back("--reporter");
  storage.push_back(std::string("junit::out=") + docs.UTF8String + "/lokinfer_tests.junit.xml");
  std::vector<char*> argv;
  for (auto& s : storage) argv.push_back(s.data());
  Catch::Session session;
  int rc = session.applyCommandLine((int)argv.size(), argv.data());
  if (rc == 0) rc = session.run();
  std::printf("LOKINFER_TESTS_EXIT_CODE=%d\n", rc);
  std::fflush(stdout);
  return rc;
}

@interface SceneDelegate : UIResponder <UIWindowSceneDelegate>
@property(strong, nonatomic) UIWindow* window;
@end

@implementation SceneDelegate
// iOS 27 requires the UIScene lifecycle (apps without it trap at launch).
- (void)scene:(UIScene*)scene willConnectToSession:(UISceneSession*)session options:(UISceneConnectionOptions*)opts {
  UIWindowScene* ws = (UIWindowScene*)scene;
  self.window = [[UIWindow alloc] initWithWindowScene:ws];
  UIViewController* vc = [UIViewController new];
  UILabel* l = [[UILabel alloc] initWithFrame:ws.coordinateSpace.bounds];
  l.text = @"lokinfer test host";
  l.textAlignment = NSTextAlignmentCenter;
  [vc.view addSubview:l];
  vc.view.backgroundColor = UIColor.systemBackgroundColor;
  self.window.rootViewController = vc;
  [self.window makeKeyAndVisible];
}
@end

std::string lokihost_build_id();  // build_id.mm

// Runtime-variant knobs this process was launched with (docs/benchmarking.md "CPU runtime variants").
static std::string runtime_env() {
  std::string r;
  for (const char* k : {"LOKINFER_THREADS", "LOKINFER_POOL_QOS", "VECLIB_MAXIMUM_THREADS", "LOKINFER_F16_GEMM",
                        "LOKINFER_F16T_KERNEL", "LOKINFER_F16T_TILE"})
    if (const char* v = std::getenv(k)) r += std::string(r.empty() ? "" : " ") + k + "=" + v;
  return r;
}

// Live device state for benchmark records (CLAUDE.md: every benchmark records LPM + thermal state).
static std::string device_state_json() {
  static const char* kThermal[] = {"nominal", "fair", "serious", "critical"};
  NSProcessInfo* pi = NSProcessInfo.processInfo;
  const NSInteger t = pi.thermalState;
  struct utsname u;
  uname(&u);
  return std::string("{\"device\": \"") + u.machine + "\", \"os\": \"" +
         UIDevice.currentDevice.systemVersion.UTF8String + "\", \"thermal\": \"" +
         (t >= 0 && t < 4 ? kThermal[t] : "unknown") + "\", \"low_power_mode\": " +
         (pi.lowPowerModeEnabled ? "true" : "false") + ", \"build\": \"" + lokihost_build_id() +
         "\", \"runtime_env\": \"" + runtime_env() + "\"}";
}

// --serve: remote executor (pytest on the Mac sends models + inputs over the
// USB tunnel). Blobs live in Application Support (survives reinstalls).
static void run_server() {
  NSString* support = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES)[0];
  NSString* lib = [[NSBundle mainBundle] pathForResource:@"lokinfer" ofType:@"metallib"];
  setenv("LOKINFER_METALLIB", lib.UTF8String, 1);
  lk::remote::ServerOptions o;
  o.blobDir = std::string(support.UTF8String) + "/lokinfer/blobs";
  o.deviceInfo = device_state_json;
  lk::Status s = lk::remote::runServer(o);
  std::printf("server stopped: %s\n", s.isOk() ? "ok" : s.message().c_str());
  std::fflush(stdout);
}

@implementation AppDelegate
- (BOOL)application:(UIApplication*)app didFinishLaunchingWithOptions:(NSDictionary*)opts {
  app.idleTimerDisabled = YES;  // keep the screen on during long benches
  NSArray* args = NSProcessInfo.processInfo.arguments;
  if ([args containsObject:@"--bench-amx"]) {
    // Multi-stream sgemm (AMX / SME) re-verification (bench/kernels/amx_bench.cpp); then serve if asked.
    [NSThread detachNewThreadWithBlock:^{
      std::printf("LOKINFER_DEVICE_STATE before %s\n", device_state_json().c_str());
      lk::bench::runAmxBench({2, 3, 4, 6});
      std::printf("LOKINFER_DEVICE_STATE after %s\n", device_state_json().c_str());
      std::fflush(stdout);
      if ([args containsObject:@"--serve"]) run_server();
      exit(0);
    }];
    return YES;
  }
  if ([args containsObject:@"--bench"]) {
    // M2 conv proof on the phone; output streams to the devicectl console.
    [NSThread detachNewThreadWithBlock:^{
      NSString* lib = [[NSBundle mainBundle] pathForResource:@"bench" ofType:@"metallib"];
      std::printf("LOKINFER_DEVICE_STATE before %s\n", device_state_json().c_str());
      lk::bench::runConvBench(lib.UTF8String, true, true);
      std::printf("LOKINFER_DEVICE_STATE after %s\n", device_state_json().c_str());
      std::fflush(stdout);
      if ([args containsObject:@"--serve"]) run_server();
      exit(0);
    }];
    return YES;
  }
  if ([args containsObject:@"--serve"]) {
    // Optionally run the Catch2 suite first (Catch2 allows one session per
    // process), then stay up as the remote executor: one launch per build,
    // and the foreground app keeps the phone from auto-locking.
    [NSThread detachNewThreadWithBlock:^{
      run_tests(args);
      run_server();
      exit(0);
    }];
    return YES;
  }
  [NSThread detachNewThreadWithBlock:^{
    int rc = run_tests(args);
    if (rc >= 0) exit(rc);  // test mode: exit so `devicectl --console` returns
  }];
  return YES;
}
- (UISceneConfiguration*)application:(UIApplication*)app
    configurationForConnectingSceneSession:(UISceneSession*)session
                                   options:(UISceneConnectionOptions*)options {
  UISceneConfiguration* c = [[UISceneConfiguration alloc] initWithName:@"Default" sessionRole:session.role];
  c.delegateClass = SceneDelegate.class;
  return c;
}
@end

int main(int argc, char* argv[]) {
  @autoreleasepool {
    return UIApplicationMain(argc, argv, nil, NSStringFromClass([AppDelegate class]));
  }
}
