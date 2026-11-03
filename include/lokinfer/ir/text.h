// Textual IR: printer and parser (see docs/impl-notes/design-ir.md §7).
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "lokinfer/ir/ir.h"
#include "lokinfer/ir/status.h"

namespace lk {

std::string printModel(const Model& model);

// Internal constants referenced by `const.internal` must be declared by the
// caller (or come from a binary file); the text form only names them. Unknown
// names are recorded as constants with no data.
Expected<std::unique_ptr<Model>> parseModel(std::string_view text);

// Formats a value of an inline constant (exactly round-trippable).
std::string formatScalar(DType t, const uint8_t* bytes);

}  // namespace lk
