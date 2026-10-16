#ifndef HK_TYPES_TD
#define HK_TYPES_TD

include "hk/Dialect/HK/IR/HKBase.td"

def HK_RegTileType : HK_Type<"RegTile", "rt"> {
  let summary = "a register tile: kittens::rt<T, rows, cols, layout>, or a strip";
  let description = [{
    `!hk.rt<f32, 16x64, row>`. The layout is not cosmetic: it decides which
    mma variant applies and whether an LDS load is a ds_read_b128 or sixteen
    scalar reads. Extents are in units of the 16x16 WMMA fragment.

    `!hk.rt<bf16, 16x64, row, frag 16x32>` names the matrix-core fragment
    the tile is laid out in, where it is not the target's 16x16: gfx950's
    16x16x32 MFMA reads 16x32 operands, and include/cdna4 spells that tile
    rt<bf16, 16, 64, row, rt_16x32>. Unset (0x0) is the 16x16 default;
    hk-fragments sets it on the operand tiles of such a target.

    `!hk.rt<f32, 1x1024, strip<8>>` is a strip, kittens::rs<T, rows, cols, 8>:
    the layout a row kernel wants rather than the one WMMA gives, each lane
    owning `run` contiguous elements of a row per 32*run-column chunk. Rows
    are any positive count; columns a whole number of chunks. A strip is
    never an mma operand, never goes through LDS as a tile, and has no
    fragment.
  }];
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$rows,
                        "unsigned":$cols, "bool":$colLayout,
                        "unsigned":$fragRows, "unsigned":$fragCols,
                        DefaultValuedParameter<"unsigned", "0">:$run);
  let builders = [
    TypeBuilder<(ins "::mlir::Type":$elementType, "unsigned":$rows,
                     "unsigned":$cols, "bool":$colLayout,
                     "unsigned":$fragRows, "unsigned":$fragCols,
                     CArg<"unsigned", "0">:$run), [{
      return $_get($_ctxt, elementType, rows, cols, colLayout, fragRows,
                   fragCols, run);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    bool isStrip() const { return getRun() != 0; }
    ::llvm::StringRef getLayout() const {
      return isStrip() ? "strip" : getColLayout() ? "col" : "row";
    }
    unsigned getBaseTiles() const { return (getRows() / 16) * (getCols() / 16); }
    bool hasFrag() const { return getFragRows() != 0; }
    /// The fragment's rows and cols, 16x16 where unset.
    unsigned fragR() const { return hasFrag() ? getFragRows() : 16; }
    unsigned fragC() const { return hasFrag() ? getFragCols() : 16; }
    RegTileType withFrag(unsigned r, unsigned c) const {
      return RegTileType::get(getContext(), getElementType(), getRows(),
                              getCols(), getColLayout(), r, c, getRun());
    }
    RegTileType transposed() const {
      return RegTileType::get(getContext(), getElementType(), getCols(),
                              getRows(), !getColLayout(), getFragCols(),
                              getFragRows());
    }
  }];
}

def HK_SharedTileType : HK_Type<"SharedTile", "st"> {
  let summary = "an LDS tile: kittens::st<T, rows, cols>, or `count` of them";
  let description = [{
    `!hk.st<bf16, 128x64>`, or `!hk.st<bf16, 128x64, 2>` for a contiguous,
    size-aligned double buffer. Shared tiles carry no layout: the swizzle is a
    property of the C++ type.
  }];
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$rows,
                        "unsigned":$cols, "unsigned":$count);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    SharedTileType element() const {
      return SharedTileType::get(getContext(), getElementType(), getRows(),
                                 getCols(), 1);
    }
    uint64_t getNumBytes() const;
  }];
}

def HK_RegVecType : HK_Type<"RegVec", "rv"> {
  let summary = "a register vector: one value per row (col) or column (row) of a tile";
  let description = [{
    `!hk.rv<f32, 16x64, row, col>`: the col_vec of an rt<f32,16x64,row>. It
    carries the tile it came off, because an rv's lane layout is derived from
    that tile and a vector spelled independently of it reduces the wrong
    lanes. `uniform` records that every entry holds the same value. A
    strip's vectors carry the strip: `!hk.rv<f32, 1x1024, strip<8>, col>`.
  }];
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$tileRows,
                        "unsigned":$tileCols, "bool":$tileColLayout,
                        "bool":$colKind, "bool":$uniform,
                        "unsigned":$tileFragRows, "unsigned":$tileFragCols,
                        DefaultValuedParameter<"unsigned", "0">:$tileRun);
  let builders = [
    TypeBuilder<(ins "::mlir::Type":$elementType, "unsigned":$tileRows,
                     "unsigned":$tileCols, "bool":$tileColLayout,
                     "bool":$colKind, "bool":$uniform,
                     "unsigned":$tileFragRows, "unsigned":$tileFragCols,
                     CArg<"unsigned", "0">:$tileRun), [{
      return $_get($_ctxt, elementType, tileRows, tileCols, tileColLayout,
                   colKind, uniform, tileFragRows, tileFragCols, tileRun);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    ::llvm::StringRef getKind() const { return getColKind() ? "col" : "row"; }
    unsigned getLength() const { return getColKind() ? getTileRows() : getTileCols(); }
    bool isStrip() const { return getTileRun() != 0; }
    RegTileType getTile() const {
      return RegTileType::get(getContext(), getElementType(), getTileRows(),
                              getTileCols(), getTileColLayout(),
                              getTileFragRows(), getTileFragCols(), getTileRun());
    }
    RegVecType plain() const {
      return RegVecType::get(getContext(), getElementType(), getTileRows(),
                             getTileCols(), getTileColLayout(), getColKind(),
                             false, getTileFragRows(), getTileFragCols(),
                             getTileRun());
    }
    /// The same vector off a tile of fragment `t`.
    RegVecType withTile(RegTileType t) const {
      return RegVecType::get(getContext(), getElementType(), getTileRows(),
                             getTileCols(), getTileColLayout(), getColKind(),
                             getUniform(), t.getFragRows(), t.getFragCols(),
                             getTileRun());
    }
    /// Same C++ type: rt<T,R,C,L>::col_vec does not depend on C (rt.cuh:78);
    /// it does on the fragment, which include/cdna4's rv carries. A strip's
    /// col_vec is rs_cv<T, R>, its row_vec rs<T, 1, C, run>.
    bool sameCpp(RegVecType other) const {
      if (getElementType() != other.getElementType() ||
          getLength() != other.getLength() ||
          getColKind() != other.getColKind() || isStrip() != other.isStrip())
        return false;
      if (isStrip())
        return getColKind() || getTileRun() == other.getTileRun();
      return getTileColLayout() == other.getTileColLayout() &&
             getTileFragRows() == other.getTileFragRows() &&
             getTileFragCols() == other.getTileFragCols();
    }
  }];
}

def HK_SharedVecType : HK_Type<"SharedVec", "sv"> {
  let summary = "an LDS vector: kittens::sv<T, length>, or `count` of them";
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$length,
                        "unsigned":$count);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    uint64_t getNumBytes() const;
  }];
}

def HK_GlobalType : HK_Type<"Global", "gl"> {
  let summary = "a global tensor, 4D (b, d, r, c): kittens::gl<T, -1, -1, -1, -1>";
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
}

def HK_CoordType : HK_Type<"Coord", "coord"> {
  let summary = "a (b, d, r, c) index, in tiles or in elements";
  let description = [{
    `!hk.coord<tile>` indexes in whole tiles (C++ `coord<RT>`),
    `!hk.coord<element>` in elements (`coord<>`). Mixing them up reads the
    wrong memory without a diagnostic, which is why the unit is in the type.
  }];
  let parameters = (ins "bool":$element);
  let hasCustomAssemblyFormat = 1;
}

def HK_GranulesType : HK_Type<"Granules", "granules"> {
  let summary = "the per-lane LDS addresses of one shared tile, factored";
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$rows,
                        "unsigned":$cols);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    unsigned getSwizzleBytes() const;
    unsigned getNumAddresses() const { return getSwizzleBytes() / 16; }
    SharedTileType getTile() const {
      return SharedTileType::get(getContext(), getElementType(), getRows(),
                                 getCols(), 1);
    }
  }];
}

def HK_StageType : HK_Type<"Stage", "stage"> {
  let summary = "float4 buf[depth][calls]: the VGPR half of a global->LDS copy";
  let parameters = (ins "::mlir::Type":$elementType, "unsigned":$rows,
                        "unsigned":$cols, "unsigned":$threads,
                        "unsigned":$depth);
  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    unsigned getCalls() const;
    unsigned getFloats() const { return getCalls() * getDepth(); }
    SharedTileType getTile() const {
      return SharedTileType::get(getContext(), getElementType(), getRows(),
                                 getCols(), 1);
    }
  }];
}

def HK_AnyTile : AnyTypeOf<[HK_RegTileType]>;
def HK_Elementwise : AnyTypeOf<[HK_RegTileType, HK_RegVecType],
                               "register tile or vector">;

#endif // HK_TYPES_TD
