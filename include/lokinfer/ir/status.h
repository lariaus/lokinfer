// Exception-free error handling: Status for fallible operations without a
// value, Expected<T> for fallible operations returning one.
#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <utility>

namespace lk {

class Status {
 public:
  Status() = default;
  static Status ok() { return Status(); }
  static Status error(std::string message) {
    Status s;
    s.message_ = std::move(message);
    s.failed_ = true;
    return s;
  }

  bool isOk() const { return !failed_; }
  explicit operator bool() const { return !failed_; }
  const std::string& message() const { return message_; }

 private:
  std::string message_;
  bool failed_ = false;
};

template <typename T>
class Expected {
 public:
  Expected(T value) : value_(std::move(value)) {}
  Expected(Status status) : status_(std::move(status)) { assert(!status_.isOk()); }

  bool isOk() const { return value_.has_value(); }
  explicit operator bool() const { return isOk(); }
  const Status& status() const { return status_; }

  T& operator*() { assert(isOk()); return *value_; }
  const T& operator*() const { assert(isOk()); return *value_; }
  T* operator->() { assert(isOk()); return &*value_; }
  const T* operator->() const { assert(isOk()); return &*value_; }
  T take() { assert(isOk()); return std::move(*value_); }

 private:
  std::optional<T> value_;
  Status status_;
};

#define LK_RETURN_IF_ERROR(expr)              \
  do {                                        \
    ::lk::Status lk_status_ = (expr);         \
    if (!lk_status_.isOk()) return lk_status_; \
  } while (0)

}  // namespace lk
