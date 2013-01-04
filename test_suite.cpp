#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/mpl/list.hpp>
#include <random>

#include "cl_utils.h"
#include "utils.h"

using namespace DynamiCL;

BOOST_AUTO_TEST_SUITE(utils)

BOOST_AUTO_TEST_CASE( stripExtension_test )
{
    using namespace std;

    BOOST_CHECK_EQUAL( "", stripExtension("") );
    BOOST_CHECK_EQUAL( "a", stripExtension("a") );
    BOOST_CHECK_EQUAL( "longer", stripExtension("longer") );
    BOOST_CHECK_EQUAL( "", stripExtension(".") );
    BOOST_CHECK_EQUAL( "", stripExtension(".jpg") );
    BOOST_CHECK_EQUAL( "hello", stripExtension("hello.jpg") );
    BOOST_CHECK_EQUAL( ".jpg", stripExtension(".jpg.bmp") );
    BOOST_CHECK_EQUAL( "hello.jpg", stripExtension("hello.jpg.bmp") );
}

BOOST_AUTO_TEST_SUITE_END()
// ========================================================

BOOST_AUTO_TEST_SUITE( host_image )

typedef boost::mpl::list<int,long,unsigned char> pix_types;

BOOST_AUTO_TEST_CASE_TEMPLATE( common_ops, PixType, pix_types)
{
    size_t width = 4;
    size_t height = 3;
    HostImage<PixType, 2> image(width, height);

    BOOST_CHECK_EQUAL( image.valid(), true );
    BOOST_CHECK_EQUAL( image.totalSize(), width*height );
    BOOST_CHECK_EQUAL( image.width(), width );
    BOOST_CHECK_EQUAL( image.height(), height );

    char* b = reinterpret_cast<char*>(image.begin());
    char* e = reinterpret_cast<char*>(image.end());

    BOOST_CHECK_EQUAL( e-b, width*height*sizeof(PixType) );

    // generate random data
    std::vector<PixType> input(image.totalSize());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> d(0, 255);
    std::generate(input.begin(), input.end(),
                  [&]() { return d(gen); });

    // populate with random data
    std::copy(input.begin(), input.end(), image.begin());
    BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(),
                                  image.begin(), image.end());

    // move test
    HostImage<PixType, 2> moved(std::move(image));

    BOOST_CHECK_EQUAL( image.valid(), false );
    BOOST_CHECK_EQUAL( image.totalSize(), 0 );
    BOOST_CHECK_EQUAL( moved.valid(), true );
    BOOST_CHECK_EQUAL( moved.totalSize(), width*height );
    BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(),
                                  moved.begin(), moved.end());

}

BOOST_AUTO_TEST_CASE_TEMPLATE( image_array_construction, PixType, pix_types )
{
    size_t width = 4;
    size_t height = 3;
    size_t depth = 5;

    typedef HostImage<PixType, 2> subimage_type;
    std::vector< subimage_type > subimages;

    std::generate_n(std::back_inserter(subimages), depth,
                    [&]() { return subimage_type(width, height); });

    // assert sizes
    for (auto&& image : subimages)
    {
        BOOST_CHECK_EQUAL( image.valid(), true );
        BOOST_CHECK_EQUAL( image.totalSize(), width*height );
        BOOST_CHECK_EQUAL( image.width(), width );
        BOOST_CHECK_EQUAL( image.height(), height );
    }

    // generate random data
    size_t total = width*height*depth;
    std::vector<PixType> input(total);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned char> d(0, 255);
    std::generate(input.begin(), input.end(),
                  [&]() { return d(gen); });

    // populate with random data
    auto inputIt = input.begin();
    size_t pitch = width*height;
    for (auto&& image : subimages)
    {
        std::copy(inputIt, inputIt + pitch,
                  image.begin());
        BOOST_CHECK_EQUAL_COLLECTIONS(inputIt, inputIt + pitch,
                                      image.begin(), image.end());
        inputIt += pitch;
    }

    // create array
    HostImage<PixType, 3> imageArray(subimages);

    BOOST_CHECK_EQUAL( imageArray.valid(), true );
    BOOST_CHECK_EQUAL( imageArray.totalSize(), total );
    BOOST_CHECK_EQUAL( imageArray.width(), width );
    BOOST_CHECK_EQUAL( imageArray.height(), height );
    BOOST_CHECK_EQUAL( imageArray.depth(), depth );
    BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(),
                                  imageArray.begin(), imageArray.end());


    // collapse dimension
    HostImage<PixType, 2> collapsed = collapseDimension(imageArray);
    BOOST_CHECK_EQUAL( imageArray.valid(), false );
    BOOST_CHECK_EQUAL( imageArray.totalSize(), 0 );
    BOOST_CHECK_EQUAL( collapsed.valid(), true );
    BOOST_CHECK_EQUAL( collapsed.totalSize(), total );
    BOOST_CHECK_EQUAL( collapsed.width(), width );
    BOOST_CHECK_EQUAL( collapsed.height(), height*depth );
    BOOST_CHECK_EQUAL_COLLECTIONS(input.begin(), input.end(),
                                  collapsed.begin(), collapsed.end());

}

BOOST_AUTO_TEST_SUITE_END()
