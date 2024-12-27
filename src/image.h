#ifndef IMAGE_H
#define IMAGE_H

#include "color.h"
#include <vector>
#include <ostream>

class image
{
public:
    image(int width, int height) : width_(width), height_(height)
    {
        pixels_.resize(width * height);
    }

    int width() const { return width_; }
    int height() const { return height_; }

    color get_pixel(int i, int j) const
    {
        return pixels_[j * width_ + i];
    }

    void set_pixel(int i, int j, color pixel)
    {
        pixels_[j * width_ + i] = pixel;
    }

private:
    int width_;
    int height_;
    std::vector<color> pixels_;
};

std::ostream &operator<<(std::ostream &out, const image &img)
{
    out << "P3\n"
        << img.width() << ' ' << img.height() << "\n255\n";
    for (size_t j = 0; j < img.height(); j++)
    {
        for (size_t i = 0; i < img.width(); i++)
        {
            write_color(out, img.get_pixel(i, j));
        }
    }
    return out;
}

#endif // IMAGE_H