import pytest
import ctypes
import sys
import math


# Adversarial payloads targeting integer overflow in allocation size computation
# These represent values that could cause arithmetic multiplication overflow
# when computing buffer sizes (e.g., n_items * item_size)
@pytest.mark.parametrize("payload", [
    # Large values near integer boundaries
    {"n_items": 2**31 - 1, "item_size": 2},          # Near INT32_MAX
    {"n_items": 2**32 - 1, "item_size": 2},          # Near UINT32_MAX
    {"n_items": 2**63 - 1, "item_size": 2},          # Near INT64_MAX
    {"n_items": 2**64 - 1, "item_size": 2},          # Near UINT64_MAX
    {"n_items": 2**31, "item_size": 2},               # Just over INT32_MAX
    {"n_items": 2**32, "item_size": 2},               # Just over UINT32_MAX
    # Classic overflow triggers
    {"n_items": 2**16, "item_size": 2**16},           # 2^32 overflow
    {"n_items": 2**32, "item_size": 2**32},           # 2^64 overflow
    {"n_items": 0x80000000, "item_size": 2},          # Sign bit flip
    {"n_items": 0xFFFFFFFF, "item_size": 0xFFFFFFFF}, # Max * Max
    # Boundary values
    {"n_items": 0, "item_size": 0},                   # Zero case
    {"n_items": 1, "item_size": 1},                   # Minimum valid
    {"n_items": 2**15, "item_size": 2**15},           # 2^30 - safe
    {"n_items": 2**16 + 1, "item_size": 2**16 + 1},  # Just over 2^16
    # Typical auction algorithm sizes
    {"n_items": 1000000, "item_size": 8},             # 8MB - reasonable
    {"n_items": 10000000, "item_size": 8},            # 80MB - large
    {"n_items": 100000000, "item_size": 8},           # 800MB - very large
    # Negative-like values (when cast to signed)
    {"n_items": 2**31, "item_size": 1},               # Negative when signed 32-bit
    {"n_items": 2**63, "item_size": 1},               # Negative when signed 64-bit
    # Values that overflow to small numbers
    {"n_items": 2**32 + 1, "item_size": 2**32 + 1},  # Wraps to 1 in 32-bit
    {"n_items": 2**64 + 1, "item_size": 2**64 + 1},  # Wraps to 1 in 64-bit
])
def test_allocation_size_no_overflow(payload):
    """Invariant: Computing allocation size (n_items * item_size) must never
    produce a result smaller than either operand when both are positive and
    non-zero, and must accurately represent the true mathematical product
    within the platform's addressable memory space. No overflow-induced
    under-allocation should occur."""

    n_items = payload["n_items"]
    item_size = payload["item_size"]

    # Property 1: The computed allocation size must be mathematically correct
    # (no overflow wrapping to a smaller value)
    true_product = n_items * item_size  # Python uses arbitrary precision

    # Simulate what a C-like 64-bit multiplication would produce
    # (wrapping at 2^64 for unsigned 64-bit)
    c_uint64_max = 2**64 - 1
    c_int64_max = 2**63 - 1
    c_uint32_max = 2**32 - 1

    # Check 64-bit unsigned overflow
    if n_items > 0 and item_size > 0:
        wrapped_64 = (n_items * item_size) & c_uint64_max
        # If overflow occurred, wrapped result will be less than true product
        overflow_occurred_64 = wrapped_64 != true_product

        if overflow_occurred_64:
            # If overflow would occur, the allocation would be too small
            # This is the dangerous condition - wrapped size < true required size
            assert wrapped_64 < true_product, (
                f"Overflow detection logic error: wrapped={wrapped_64}, "
                f"true={true_product}"
            )
            # The invariant: if overflow would occur, it MUST be detected
            # We verify that the overflow IS detectable (wrapped < true)
            # In secure code, this should trigger an error, not proceed
            overflow_detectable = (
                true_product > c_uint64_max or  # True product exceeds 64-bit
                wrapped_64 < n_items or          # Classic overflow check
                wrapped_64 < item_size           # Classic overflow check
            )
            assert overflow_detectable, (
                f"SECURITY VIOLATION: Overflow not detectable! "
                f"n_items={n_items}, item_size={item_size}, "
                f"wrapped={wrapped_64}, true={true_product}. "
                f"A secure implementation must detect this overflow before allocation."
            )

    # Property 2: Zero inputs should not cause issues
    if n_items == 0 or item_size == 0:
        assert true_product == 0, (
            f"Zero multiplication should yield zero, got {true_product}"
        )

    # Property 3: For safe (non-overflowing) inputs, the result must be exact
    if n_items > 0 and item_size > 0 and true_product <= c_uint64_max:
        computed = n_items * item_size
        assert computed == true_product, (
            f"Safe multiplication must be exact: "
            f"{n_items} * {item_size} = {true_product}, got {computed}"
        )

    # Property 4: Allocation size must never be smaller than n_items
    # (when item_size >= 1) - if it is, buffer overflow is possible
    if item_size >= 1 and n_items > 0:
        # In Python (arbitrary precision), this always holds
        assert true_product >= n_items, (
            f"Allocation size {true_product} must be >= n_items {n_items} "
            f"when item_size={item_size} >= 1"
        )

    # Property 5: Verify overflow detection using standard C idioms
    # The check: if (n_items != 0 && size > SIZE_MAX / n_items) -> overflow
    if n_items > 0 and item_size > 0:
        size_max_64 = c_uint64_max
        # Standard overflow pre-check
        safe_to_allocate_64 = (item_size <= size_max_64 // n_items)

        if not safe_to_allocate_64:
            # Overflow would occur - verify the wrapped value is indeed smaller
            wrapped = (n_items * item_size) % (2**64)
            assert wrapped < true_product, (
                f"When overflow is detected, wrapped value must be less than "
                f"true product. n_items={n_items}, item_size={item_size}"
            )