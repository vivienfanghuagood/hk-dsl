// RUN: hk-opt %s --pass-pipeline='builtin.module(func.func(hk-map-fuse))' | FileCheck %s

// A chain of one-op maps -- what `((c >> 4) & 63) * n` builds -- becomes one
// map with the chain's operands, deduplicated.

// CHECK-LABEL: func.func @chain
func.func @chain(%c: !hk.rt<i32, 1x64, strip<1>>, %n: i32, %g: !hk.gl<i32>, %idx: !hk.coord<element>) attributes {hk.kernel} {
  // CHECK-NEXT: %[[M:[a-z0-9_]+]] = hk.map(%{{.*}}, %{{.*}}) : (i32, !hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  // CHECK-NEXT: ^bb0(%[[N:[a-z0-9_]+]]: i32, %[[X:[a-z0-9_]+]]: i32):
  // CHECK-NEXT: %[[C4:[a-z0-9_]+]] = arith.constant 4
  // CHECK-NEXT: %[[S:[a-z0-9_]+]] = arith.shrsi %[[X]], %[[C4]]
  // CHECK-NEXT: %[[C63:[a-z0-9_]+]] = arith.constant 63
  // CHECK-NEXT: %[[A:[a-z0-9_]+]] = arith.andi %[[S]], %[[C63]]
  // CHECK-NEXT: %[[P:[a-z0-9_]+]] = arith.muli %[[A]], %[[N]]
  // CHECK-NEXT: %[[Q:[a-z0-9_]+]] = arith.addi %[[P]], %[[X]]
  // CHECK-NEXT: hk.yield %[[Q]] : i32
  // CHECK-NEXT: }
  // CHECK-NEXT: hk.store %[[M]]
  %s = hk.map(%c) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32):
    %c4 = arith.constant 4 : i32
    %r = arith.shrsi %x, %c4 : i32
    hk.yield %r : i32
  }
  %a = hk.map(%s) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32):
    %c63 = arith.constant 63 : i32
    %r = arith.andi %x, %c63 : i32
    hk.yield %r : i32
  }
  %p = hk.map(%a, %n, %c) : (!hk.rt<i32, 1x64, strip<1>>, i32, !hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32, %y: i32, %z: i32):
    %m = arith.muli %x, %y : i32
    %r = arith.addi %m, %z : i32
    hk.yield %r : i32
  }
  hk.store %p, %g[%idx] : !hk.rt<i32, 1x64, strip<1>>, !hk.gl<i32>, !hk.coord<element>
  return
}

// A value read twice -- by two maps, or by a store and a map -- is computed
// once and read.

// CHECK-LABEL: func.func @shared
func.func @shared(%c: !hk.rt<i32, 1x64, strip<1>>, %g: !hk.gl<i32>, %idx: !hk.coord<element>) attributes {hk.kernel} {
  // CHECK-NEXT: %[[K:[a-z0-9_]+]] = hk.map(%{{.*}}) {{.*}} {
  // CHECK: %[[D:[a-z0-9_]+]] = hk.map(%{{.*}}) {{.*}} {
  // CHECK: arith.constant 2
  // CHECK: hk.map(%[[D]]) {{.*}} {
  // CHECK-NEXT: ^bb0
  // CHECK-NEXT: arith.constant 3
  // CHECK-NEXT: arith.muli
  // CHECK-NEXT: hk.yield
  // CHECK: hk.map(%[[D]]) {{.*}} {
  // CHECK-NEXT: ^bb0
  // CHECK-NEXT: arith.constant 5
  // CHECK: hk.map(%[[K]]) {{.*}} {
  // CHECK-NEXT: ^bb0
  // CHECK-NEXT: arith.constant 7
  // CHECK: hk.store %[[K]]
  %k = hk.map(%c) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32):
    %c1 = arith.constant 1 : i32
    %r = arith.addi %x, %c1 : i32
    hk.yield %r : i32
  }
  %d = hk.map(%c) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32):
    %c2 = arith.constant 2 : i32
    %r = arith.muli %x, %c2 : i32
    hk.yield %r : i32
  }
  %u = hk.map(%d) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32):
    %c3 = arith.constant 3 : i32
    %r = arith.muli %x, %c3 : i32
    hk.yield %r : i32
  }
  %v = hk.map(%d) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32):
    %c5 = arith.constant 5 : i32
    %r = arith.addi %x, %c5 : i32
    hk.yield %r : i32
  }
  %k7 = hk.map(%k) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32):
    %c7 = arith.constant 7 : i32
    %r = arith.addi %x, %c7 : i32
    hk.yield %r : i32
  }
  hk.store %k, %g[%idx] : !hk.rt<i32, 1x64, strip<1>>, !hk.gl<i32>, !hk.coord<element>
  hk.store %k7, %g[%idx] : !hk.rt<i32, 1x64, strip<1>>, !hk.gl<i32>, !hk.coord<element>
  hk.store %u, %g[%idx] : !hk.rt<i32, 1x64, strip<1>>, !hk.gl<i32>, !hk.coord<element>
  hk.store %v, %g[%idx] : !hk.rt<i32, 1x64, strip<1>>, !hk.gl<i32>, !hk.coord<element>
  return
}

// A string body is opaque: neither inlined nor inlined into. A producer
// outside a loop stays outside it.

// CHECK-LABEL: func.func @opaque_and_loop
func.func @opaque_and_loop(%c: !hk.rt<i32, 1x64, strip<1>>, %g: !hk.gl<i32>, %idx: !hk.coord<element>) attributes {hk.kernel} {
  // CHECK-NEXT: %[[B:[a-z0-9_]+]] = hk.map(%{{.*}}) {body = "x0 + 1"}
  // CHECK-NEXT: %[[R:[a-z0-9_]+]] = hk.map(%[[B]]) {{.*}} {
  // CHECK: hk.map(%[[R]]) {body = "x0 * 2"}
  // CHECK: scf.for
  // CHECK-NEXT: hk.map(%[[R]], %{{.*}})
  %b = hk.map(%c) {body = "x0 + 1"} : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>>
  %r = hk.map(%b) : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>> {
  ^bb0(%x: i32):
    %c3 = arith.constant 3 : i32
    %y = arith.shli %x, %c3 : i32
    hk.yield %y : i32
  }
  %t = hk.map(%r) {body = "x0 * 2"} : (!hk.rt<i32, 1x64, strip<1>>) -> !hk.rt<i32, 1x64, strip<1>>
  hk.store %t, %g[%idx] : !hk.rt<i32, 1x64, strip<1>>, !hk.gl<i32>, !hk.coord<element>
  %lo = arith.constant 0 : i32
  %hi = arith.constant 4 : i32
  %st = arith.constant 1 : i32
  scf.for %i = %lo to %hi step %st : i32 {
    %w = hk.map(%r, %i) : (!hk.rt<i32, 1x64, strip<1>>, i32) -> !hk.rt<i32, 1x64, strip<1>> {
    ^bb0(%x: i32, %j: i32):
      %y = arith.addi %x, %j : i32
      hk.yield %y : i32
    }
    hk.store %w, %g[%idx] : !hk.rt<i32, 1x64, strip<1>>, !hk.gl<i32>, !hk.coord<element>
  }
  return
}
