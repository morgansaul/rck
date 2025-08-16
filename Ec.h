#pragma once

#include "defs.h"
#include <stdint.h>

// Forward declarations for ECC big integer and point used across the project.

struct EcInt {
    u64 data[4]; // 256-bit little-endian limbs
    // basic ops (implemented in Ec.cpp)
    void Assign(EcInt& val);
    void Set(u64 val);
    void SetZero();
    bool SetHexStr(const char* str);
    void GetHexStr(char* str);
    u16 GetU16(int index);
    bool Add(EcInt& val);
    bool Sub(EcInt& val);
    bool IsEqual(EcInt& val);
    bool IsZero();
    bool IsLessThanU(EcInt& val);
    bool IsLessThanI(EcInt& val);
    void AddModP(EcInt& val);
    void SubModP(EcInt& val);
    void Neg();
    void Neg256();
    void NegModP();
    void NegModN();
    void ShiftRight(int nbits);
    void ShiftLeft(int nbits);
    void MulModP(EcInt& val);
    void Mul_u64(EcInt& val, u64 multiplier);
    void Mul_i64(EcInt& val, i64 multiplier);
    void InvModP();
    void SqrtModP();
    void RndBits(int nbits);
    void RndMax(EcInt& max);
};

struct EcPoint {
    EcInt x;
    EcInt y;
    bool is_zero;
    // helpers (implemented in Ec.cpp)
    bool IsEqual(EcPoint& pnt);
    void LoadFromBuffer64(u8* buffer);
    void SaveToBuffer64(u8* buffer);
    bool SetHexStr(const char* str);
};

struct Ec {
    static EcPoint AddPoints(EcPoint& pnt1, EcPoint& pnt2);
    static EcPoint DoublePoint(EcPoint& pnt);
    static EcPoint MultiplyG(EcInt& k);
    static EcPoint MultiplyG_Fast(EcInt& k);
    static EcInt   CalcY(EcInt& x, bool is_even);
    static bool    IsValidPoint(EcPoint& pnt);
};

// Globals (defined in Ec.cpp)
extern EcInt g_P;
extern EcInt g_N;
extern EcPoint g_G;

// Lifecycle (defined in Ec.cpp)
void InitEc();
void DeInitEc();

void InitEc();
void DeInitEc();

void SetRndSeed(u64 seed);
