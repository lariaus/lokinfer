// lokinfer-serve: the remote executor on the Mac (protocol tests, and the
// same code path the iOS test host runs with --serve).
//   lokinfer-serve [--port N] [--blobs DIR]
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "lokinfer/remote/server.h"

int main(int argc, char** argv) {
  lk::remote::ServerOptions o;
  o.blobDir = (std::filesystem::temp_directory_path() / "lokinfer-blobs").string();
  o.deviceInfoJson = R"({"device": "mac", "os": "macOS", "gpu": "local"})";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--port" && i + 1 < argc) o.port = static_cast<uint16_t>(std::atoi(argv[++i]));
    else if (a == "--blobs" && i + 1 < argc) o.blobDir = argv[++i];
  }
  lk::Status s = lk::remote::runServer(o);
  if (!s) {
    std::fprintf(stderr, "error: %s\n", s.message().c_str());
    return 1;
  }
  return 0;
}
