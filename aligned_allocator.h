#pragma once

#ifdef __cpp_aligned_new

#include <limits>
#include <new>
#include <numeric>

/**
 * Returns aligned pointers when allocations are requested. Default alignment
 * is 64B = 512b, sufficient for AVX-512 and most cache line sizes.
 *
 * @tparam ALIGNMENT_IN_BYTES Must be a positive power of 2.
 */
template <typename ElementType, std::size_t ALIGNMENT_IN_BYTES = 64>
class aligned_allocator {
 private:
  static_assert(
      ALIGNMENT_IN_BYTES >= alignof(ElementType),
      "Beware that types like int have minimum alignment requirements "
      "or access will result in crashes.");

 public:
  using value_type = ElementType;
  static constexpr std::align_val_t ALIGNMENT{ALIGNMENT_IN_BYTES};

  /**
   * This is only necessary because AlignedAllocator has a second template
   * argument for the alignment that will make the default
   * std::allocator_traits implementation fail during compilation.
   * @see https://stackoverflow.com/a/48062758/2191065
   */
  template <class OtherElementType>
  struct rebind {
    using other = aligned_allocator<OtherElementType, ALIGNMENT_IN_BYTES>;
  };

 public:
  constexpr aligned_allocator() noexcept = default;

  constexpr aligned_allocator(const aligned_allocator&) noexcept = default;

  template <typename U>
  constexpr aligned_allocator(
      aligned_allocator<U, ALIGNMENT_IN_BYTES> const&) noexcept {}

  [[nodiscard]] ElementType* allocate(std::size_t nElementsToAllocate) {
    if (nElementsToAllocate > std::numeric_limits<std::size_t>::max() / sizeof(ElementType)) {
      throw std::bad_array_new_length();
    }

    auto const nBytesToAllocate = nElementsToAllocate * sizeof(ElementType);
    return reinterpret_cast<ElementType*>(
        ::operator new[](nBytesToAllocate, ALIGNMENT));
  }

  void deallocate(ElementType* allocatedPointer,
                  [[maybe_unused]] std::size_t nBytesAllocated) {
    /* According to the C++20 draft n4868 � 17.6.3.3, the delete operator
     * must be called with the same alignment argument as the new expression.
     * The size argument can be omitted but if present must also be equal to
     * the one used in new. */
    ::operator delete[](allocatedPointer, ALIGNMENT);
  }
};
#endif