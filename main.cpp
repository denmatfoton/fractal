#include <unistd.h>
#include <cstdlib>
#include <cstring>

#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <list>
#include <vector>

#include <png.h>

using namespace std;


constexpr int thread_num = 8;
constexpr int color_comp_num = 3;

int img_bit_depth = 8;
size_t color_comp_size = 1;
size_t pixel_size = 3;
double range = 2.;
double escape_threshold = 2.;
double julia_cx = -.79, julia_cy = .15;
double shift_x = 0.0, shift_y = 0.0;
bool mandelbrot = false;
bool antialias = false;

struct ColorSet {
   uint32_t max_iteration;
   uint32_t color_map[8][color_comp_num];
   uint32_t color_pos[8];
};

const ColorSet color_sets[2] = {
   {
      256 * 4,
      {
         {0, 0, 0},
         {40, 40, 200},
         {250, 250, 250},
         {250, 250, 100},
         {250, 250, 20},
         {250, 140, 10},
         {250, 40, 10},
         {250, 0, 0}
      },
      {0, 128, 256, 384, 512, 640, 768, 1024}
   },
   {
      256 * 16,
      {
         {0, 0, 0},
         {2, 2, 30},
         {40, 40, 200},
         {250, 250, 250},
         {250, 250, 100},
         {250, 250, 20},
         {250, 140, 10},
         {250, 40, 10}
      },
      {0, 384, 768, 1280, 2048, 2560, 3072, 4096}
   }
};


const ColorSet* cur_cs = color_sets;


inline uint16_t SwapBytes(uint16_t val) {
   return (val >> 8) | (val << 8);
}


void CalculatePoint(png_bytep px, double zx, double zy) {
   double cy = julia_cy;
   double cx = julia_cx;
   if (mandelbrot) {
      cx = zx;
      cy = zy;
      zx = 0;
      zy = 0;
   }

   uint32_t idx = 0;
   for (; idx < cur_cs->max_iteration; ++idx) {
      double zx2 = zx * zx;
      double zy2 = zy * zy;
      if (zx2 + zy2 > escape_threshold) break;
      zy = 2 * zx * zy  + cy;
      zx = zx2 - zy2 + cx;
   }

   if (idx >= cur_cs->max_iteration) {
      memset(px, 0, pixel_size);
      return;
   }

   size_t i = 0;
   for (; ; ++i) {
      if (cur_cs->color_pos[i] > idx) break;
   }

   uint32_t d1 = idx - cur_cs->color_pos[i - 1];
   uint32_t d2 = cur_cs->color_pos[i] - idx;

   if (color_comp_size == 1) {
      for (int j = 0; j < 3; ++j) {
         px[j] = static_cast<png_byte>(
            (cur_cs->color_map[i - 1][j] * d2 + cur_cs->color_map[i][j] * d1) /
            (cur_cs->color_pos[i] - cur_cs->color_pos[i - 1]));
      }
   }
   else {
      for (int j = 0; j < 3; ++j) {
         *(reinterpret_cast<uint16_t*>(px) + j) = SwapBytes(static_cast<uint16_t>(
            ((cur_cs->color_map[i - 1][j] * d2 + cur_cs->color_map[i][j] * d1) << (img_bit_depth - 8)) /
            (cur_cs->color_pos[i] - cur_cs->color_pos[i - 1])));
      }
   }
}


static void ProcessRow(png_bytep *raw_img_bytes, int im_y, int width, int height) {
   png_bytep px = raw_img_bytes[im_y];
   double zy_s = double((height >> 1) - im_y) * range / width + shift_y;

   for (int im_x = 0; im_x < width; im_x++, px += pixel_size) {
      double zx = double(im_x - (width >> 1)) * range / width + shift_x;

      CalculatePoint(px, zx, zy_s);
   }
}


static void ProcessRowAntiAlias(png_bytep *raw_img_bytes, int im_y, int width, int height) {
   png_bytep px = raw_img_bytes[im_y];
   double zy_s = double((height >> 1) - im_y) * range / width + shift_y;
   png_byte px_ext[4][6];

   for (int im_x = 0; im_x < width; im_x++, px += pixel_size) {
      double zx = double(im_x - (width >> 1)) * range / width + shift_x;
      double zy = zy_s;

      double inc = range / (width << 1);
      CalculatePoint(px_ext[0], zx, zy);
      CalculatePoint(px_ext[1], zx + inc, zy);
      zy += inc;
      CalculatePoint(px_ext[2], zx, zy);
      CalculatePoint(px_ext[3], zx + inc, zy);
      for (int j = 0; j < 3; ++j) {
         uint32_t sum = 0;
         if (color_comp_size == 1) {
            for (auto & p : px_ext) {
               sum += p[j];
            }
            px[j] = static_cast<png_byte>(sum >> 2);
         }
         else {
            for (auto & p : px_ext) {
               sum += *(reinterpret_cast<uint16_t*>(p) + j);
            }
            *(reinterpret_cast<uint16_t*>(px) + j) = static_cast<uint16_t>(sum >> 2);
         }
      }
   }
}


static void CreateFractalImage(png_bytep *raw_img_bytes, int width, int height) {
   if (mandelbrot) {
      cur_cs = color_sets + 1;
   }

   atomic_int im_y_a(0);
   list<thread> threads;

   auto start = chrono::steady_clock::now();

   for (int i = thread_num; i > 0; i--) {
      threads.push_back(move(thread( [=, &im_y_a] {
         while (true) {
            int im_y = im_y_a.fetch_add(1, std::memory_order_relaxed);
            if (im_y >= height) break;
            if (antialias) {
               ProcessRowAntiAlias(raw_img_bytes, im_y, width, height);
            }
            else {
               ProcessRow(raw_img_bytes, im_y, width, height);
            }
         }
      })));
   }

   auto helper = thread( [=, &im_y_a] {
      int prev_percent = -1;
      while (true) {
         usleep(100000);
         int im_y = im_y_a.load(std::memory_order_relaxed) - thread_num;
         if (im_y < 0) im_y = 0;
         int percent = im_y * 100 / height;
         if (percent != prev_percent) {
            printf("\r%d%%", percent);
            cout << flush;
            prev_percent = percent;
         }
         if (im_y >= height) break;
      }
      cout << endl;
   });

   for (auto& t : threads) {
      t.join();
   }

   auto end = chrono::steady_clock::now();

   helper.join();
   cout << "Calculating time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;
}


static void ParsePoint(const string& str, double& x, double& y) {
   string::size_type end;
   x = stod(str, &end);
   auto start = str.find_first_of("-0123456789.", end);
   if (start != string::npos) {
      y = stod(str.substr(start), &end);
   }
}


int main(int argc, char *argv[])
{
   png_uint_32 out_img_width_px = 2000;
   png_uint_32 out_img_height_px = 1500;
   string out_file_name = "out.png";
   int opt;
   
   while ((opt = getopt(argc, argv, "o:w:h:r:e:s:c:d:ba")) != -1) {
      switch (opt) {
         case 'o':
            out_file_name = optarg;
            break;
         case 'w':
            out_img_width_px = static_cast<png_uint_32>(strtoul(optarg, nullptr, 10));
            break;
         case 'h':
            out_img_height_px = static_cast<png_uint_32>(strtoul(optarg, nullptr, 10));
            break;
         case 'r':
            range = strtod(optarg, nullptr);
            break;
         case 'e':
            escape_threshold = strtod(optarg, nullptr);
            break;
         case 's':
            ParsePoint(optarg, shift_x, shift_y);
            break;
         case 'c':
            ParsePoint(optarg, julia_cx, julia_cy);
            break;
         case 'd':
            img_bit_depth = static_cast<int>(strtoul(optarg, nullptr, 10));
            break;
         case 'b':
            mandelbrot = true;
            break;
         case 'a':
            antialias = true;
            break;
         default: /* '?' */
            fprintf(stderr, "Usage: %s [-o output_file_name] [-w image_width] [-h image_height]\n", argv[0]);
            exit(EXIT_FAILURE);
      }
   }

   FILE *fp = fopen(out_file_name.c_str(), "wb");
   if(!fp) abort();

   png_structp out_png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
   if (!out_png) abort();

   png_infop info = png_create_info_struct(out_png);
   if (!info) abort();

   if (setjmp(png_jmpbuf(out_png))) abort();

   png_init_io(out_png, fp);

   // Output is 8bit depth, RGB format.
   png_set_IHDR(
      out_png,
      info,
      out_img_width_px, out_img_height_px,
      img_bit_depth,
      PNG_COLOR_TYPE_RGB,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT
   );
   png_write_info(out_png, info);

   pixel_size = png_get_rowbytes(out_png, info) / out_img_width_px;
   color_comp_size = pixel_size / color_comp_num;
   printf("Image bit depth: %d, pixel_size: %lu, color_comp_size: %lu\n", img_bit_depth, pixel_size, color_comp_size);

   auto *raw_img_bytes = (png_bytep*)malloc(sizeof(png_bytep) * out_img_height_px);
   for (int y = 0; y < out_img_height_px; y++) {
      raw_img_bytes[y] = (png_bytep)calloc(png_get_rowbytes(out_png, info), sizeof(png_byte));
   }

   CreateFractalImage(raw_img_bytes, static_cast<int>(out_img_width_px), static_cast<int>(out_img_height_px));

   cout << "Saving image to file: " + out_file_name << endl;
   png_write_image(out_png, raw_img_bytes);
   png_write_end(out_png, nullptr);

   for(int y = 0; y < out_img_height_px; y++) {
      free(raw_img_bytes[y]);
   }
   free(raw_img_bytes);

   fclose(fp);
   
   return 0;
}
