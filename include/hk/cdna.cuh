/**
 * @file
 * @brief What hk's translation calls that include/cdna3 and include/cdna4 do
 * not define.
 *
 * The rdna3 tree in the fork grew these alongside hk; the CDNA trees are
 * upstream HipKittens and never needed them. Kept in hk's include/ rather than
 * in the submodule, so the CDNA trees stay byte-identical to upstream and a
 * kernel written against them by hand sees no hk symbol. Included after
 * kittens.cuh, on CDNA targets only (hk/target: extra_includes).
 *
 * The scalar helpers are the rdna3 definitions with the two facts that differ
 * changed: a wave is 64 lanes, and gfx9's lgkmcnt is 4 bits (0..15), not 6.
 * The register-tile memory ops (load/store on a strided_gl, the bounded,
 * gathered and scattered rows, atomics, Philox) are rewritten against the
 * CDNA lane layouts through one coordinate function, hk_frag_coord. Not here:
 * load_w4 and the LDS granule set, which hk only uses on RDNA3.
 */

#pragma once

#if !defined(KITTENS_CDNA3) && !defined(KITTENS_CDNA4)
#error "hk/cdna.cuh is for KITTENS_CDNA3 / KITTENS_CDNA4 builds"
#endif

#include <stdexcept>
#include <string>

// Split by kind (docs/design/dsl-surface.md, S12); the order is the definitions' dependency
// order. Each part opens namespace kittens itself.
#include "cdna/core.cuh"
#include "cdna/lib.cuh"
#include "cdna/formats.cuh"
#include "cdna/special.cuh"
#include "cdna/primitives.cuh"
#include "cdna/escape.cuh"

// The AGPR-held accumulator (hk.arch.agpr), an experiment; outside namespace kittens.
#include "cdna_agpr.cuh"
