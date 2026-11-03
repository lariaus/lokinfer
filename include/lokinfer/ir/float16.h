// IEEE half and bfloat16 <-> float conversions (round to nearest even).
#pragma once

#include <cstdint>
#include <cstring>

namespace lk {

inline uint32_t floatBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

inline float bitsFloat(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline float halfToFloat(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ff;
  if (exp == 0x1f) return bitsFloat(sign | 0x7f800000 | (mant << 13));  // inf / nan
  if (exp == 0) {
    if (mant == 0) return bitsFloat(sign);
    // Subnormal: value = mant * 2^-24.
    float v = static_cast<float>(mant) * (1.0f / 16777216.0f);
    return sign ? -v : v;
  }
  return bitsFloat(sign | ((exp + 112) << 23) | (mant << 13));
}

inline uint16_t floatToHalf(float f) {
  uint32_t u = floatBits(f);
  uint32_t sign = (u >> 16) & 0x8000;
  uint32_t absu = u & 0x7fffffff;
  if (absu >= 0x7f800000) {  // inf / nan
    return static_cast<uint16_t>(sign | 0x7c00 | (absu > 0x7f800000 ? 0x200 : 0));
  }
  if (absu >= 0x477ff000) return static_cast<uint16_t>(sign | 0x7c00);  // rounds to inf
  if (absu < 0x38800000) {  // subnormal half or zero
    // Value in units of 2^-24, rounded to nearest even.
    float v = bitsFloat(absu) * 16777216.0f;
    uint32_t q = static_cast<uint32_t>(v);
    float rem = v - static_cast<float>(q);
    if (rem > 0.5f || (rem == 0.5f && (q & 1))) ++q;
    return static_cast<uint16_t>(sign | q);
  }
  uint32_t exp = ((absu >> 23) - 112) << 10;
  uint32_t mant = (absu >> 13) & 0x3ff;
  uint32_t rest = absu & 0x1fff;
  uint32_t h = exp | mant;
  if (rest > 0x1000 || (rest == 0x1000 && (mant & 1))) ++h;  // may carry into exponent: correct
  return static_cast<uint16_t>(sign | h);
}

inline float bf16ToFloat(uint16_t b) { return bitsFloat(static_cast<uint32_t>(b) << 16); }

inline uint16_t floatToBf16(float f) {
  uint32_t u = floatBits(f);
  if ((u & 0x7fffffff) > 0x7f800000) return static_cast<uint16_t>((u >> 16) | 0x40);  // quiet nan
  uint32_t rounding = 0x7fff + ((u >> 16) & 1);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

}  // namespace lk
