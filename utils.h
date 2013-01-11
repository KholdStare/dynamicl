#ifndef DYNAMICL_UTILS_H_BF56AQKP
#define DYNAMICL_UTILS_H_BF56AQKP

#include <string>

namespace DynamiCL
{
    std::string stripExtension(std::string const& path);

    /**
     * Manages heap allocated array
     */
    template <typename T>
    class array_ptr
    {
        size_t size_;
        T* array_;

        void dealloc()
        {
            delete[] array_;
        }

        void invalidate()
        {
            size_ = 0;
            array_ = nullptr;
        }

    public:
        array_ptr()
            : size_(0),
              array_(nullptr)
        { }

        array_ptr(size_t s)
            : size_(s),
              array_(new T[s])
        { }

        array_ptr(array_ptr&& other)
            : size_(other.size_),
              array_(other.array_)
        {
            other.invalidate();
        }

        array_ptr& operator = (array_ptr&& other)
        {
            size_ = other.size_;

            dealloc();
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

    };

}

#endif /* end of include guard: DYNAMICL_UTILS_H_BF56AQKP */
