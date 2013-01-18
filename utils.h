#ifndef DYNAMICL_UTILS_H_BF56AQKP
#define DYNAMICL_UTILS_H_BF56AQKP

#include <string>
#include <type_traits>

namespace DynamiCL
{
    std::string stripExtension(std::string const& path);

    namespace detail
    {

        template <typename T>
        T* align_ptr(char* unaligned, size_t alignment)
        {
            return reinterpret_cast<T*>(alignment
                                        + ((( unsigned long long )unaligned) & ~(alignment - 1)));
        }

    }

    /**
     * Manages heap allocated array
     */
    template <typename T, std::size_t Align=std::alignment_of<T>::value>
    class array_ptr
    {

        // TODO: add static assert to ensure Align is a power of 2

        size_t size_;
        char* unalignedData_;
        T* array_;

        void dealloc()
        {
            free(unalignedData_);
            invalidate();
        }

        void invalidate()
        {
            size_ = 0;
            unalignedData_ = nullptr;
            array_ = nullptr;
        }

        /**
         * Allocates enough memory to store @a n objects
         * of type T, aligned to Align.
         */
        static char* malloc_enough(size_t n)
        {
            char* result = (char*)malloc(Align + n * sizeof(T) );
            if (!result)
            {
                throw std::bad_alloc();
            }
            return result;
        }

    public:
        typedef T* iterator;
        typedef T const* const_iterator;

        array_ptr()
            : size_(0),
              unalignedData_(nullptr),
              array_(nullptr)
        { }

        array_ptr(size_t s)
            : size_(s),
              unalignedData_(malloc_enough(s)),
              array_(detail::align_ptr<T>(unalignedData_, Align))
        { }

        array_ptr(array_ptr&& other)
            : size_(other.size_),
              unalignedData_(other.unalignedData_),
              array_(other.array_)
        {
            other.invalidate();
        }

        array_ptr& operator = (array_ptr&& other)
        {
            size_ = other.size_;

            dealloc();
            unalignedData_ = other.unalignedData_;
            array_ = other.array_;

            other.invalidate();

            return *this;
        }

        array_ptr(array_ptr const& other) = delete;
        array_ptr& operator = (array_ptr const& other) = delete;

        ~array_ptr()
        { 
            dealloc();
        }

        size_t size() const { return size_; }
        T* ptr() const { return array_; }

        const_iterator begin() const { return array_; }
        const_iterator end()   const { return array_ + size_; }

        iterator begin() { return array_; }
        iterator end()   { return array_ + size_; }

    };

}

#endif /* end of include guard: DYNAMICL_UTILS_H_BF56AQKP */
