#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "utils.h"
#include "point3.h"
#include "sphere.h"
#include "ray.h"
#include "globals.h"
#include "interval.h"

void write_color(FILE *out, point3_t pixel_color)
{
  double r = linear_to_gamma(pixel_color.x);
  double g = linear_to_gamma(pixel_color.y);
  double b = linear_to_gamma(pixel_color.z);

  // Translate the [0,1] component values to the byte range [0,255].
  int rbyte = (int)(255.999 * clamp(r));
  int gbyte = (int)(255.999 * clamp(g));
  int bbyte = (int)(255.999 * clamp(b));

  fprintf(out, "%d %d %d\n", rbyte, gbyte, bbyte);
}

int main(void)
{

  // Image
  double aspect_ratio = 16.0 / 9.0;
  int image_width = 1200;

  // Calculate the image height, and ensure that it's at least 1.
  int image_height = (int)(image_width / aspect_ratio);
  if (image_height < 1)
  {
    image_height = 1;
  }

  // World
  // Cambiare il numero di num_s dentro globals.h per definire la grandezza dell'array
  sphere_t world[num_s];
  material mat;

  point3_t temp = {0, -100.5, -1};
  point3_t alb_temp = {0.8,0.8,0.0};
  mat.type = lambertian; mat.albedo = alb_temp;
  world[0].center = temp;
  world[0].radius = 100;
  world[0].mat = mat;

  temp.x = 0; temp.y = 0; temp.z = -1.2;
  alb_temp.x = 0.1; alb_temp.y = 0.2; alb_temp.z = 0.5;
  world[1].center = temp;
  world[1].radius = 0.5;
  mat.albedo = alb_temp;
  world[1].mat = mat;

  temp.x = -1.0; temp.y = 0; temp.z = -1.0;
  alb_temp.x = 0.8; alb_temp.y = 0.8; alb_temp.z = 0.8;
  world[2].center = temp;
  world[2].radius = 0.5;
  mat.type = metal;
  mat.albedo = alb_temp;
  world[2].mat = mat;

  temp.x = 1.0; temp.y = 0; temp.z = -1.0;
  alb_temp.x = 0.8; alb_temp.y = 0.6; alb_temp.z = 0.2;
  world[3].center = temp;
  world[3].radius = 0.5;
  mat.albedo = alb_temp;
  world[3].mat = mat;

  // Camera
  double focal_length = 1.0;
  double viewport_height = 2.0;
  double viewport_width = viewport_height * ((double)image_width / image_height);

  RT_DEBUG("viewport: %f %f", viewport_width, viewport_height);

  point3_t camera_center = {0, 0, 0}; // doveva essere un point3
  RT_DEBUG("camera_center: %f %f %f", camera_center.x, camera_center.y, camera_center.z);

  // Calculate the vectors across the horizontal and down the vertical viewport
  // edges.
  point3_t viewport_u = {viewport_width, 0, 0};
  point3_t viewport_v = {0, -viewport_height, 0};

  // Calculate the horizontal and vertical delta vectors from pixel to pixel.
  point3_t pixel_delta_u = vec3_div_sc(viewport_u, image_width);
  point3_t pixel_delta_v = vec3_div_sc(viewport_v, image_height);

  // Calculate the location of the upper left pixel.
  point3_t focal_offset = {0, 0, focal_length};
  point3_t half_viewport_u = vec3_div_sc(viewport_u, 2);
  point3_t half_viewport_v = vec3_div_sc(viewport_v, 2);

  // Calculate the location of the upper left corner of the viewport.
  point3_t viewport_upper_left = vec3_sub(camera_center, focal_offset);
  // Adjust the upper left corner by subtracting half the viewport width.
  viewport_upper_left = vec3_sub(viewport_upper_left, half_viewport_u);
  // Adjust the upper left corner by subtracting half the viewport height.
  viewport_upper_left = vec3_sub(viewport_upper_left, half_viewport_v);

  point3_t half_pixel_delta_u = vec3_mul_sc(pixel_delta_u, 0.5);
  point3_t half_pixel_delta_v = vec3_mul_sc(pixel_delta_v, 0.5);
  point3_t half_pixel_delta_sum = vec3_sum(half_pixel_delta_u, half_pixel_delta_v);

  // Calculate the location of the center of the first pixel (pixel00).
  point3_t pixel00_loc = vec3_sum(viewport_upper_left, half_pixel_delta_sum);

  RT_DEBUG("pixel00_loc: %f %f %f", pixel00_loc.x, pixel00_loc.y, pixel00_loc.z);

  // Opening output file
  char *output = "./out.ppm";

  FILE *out_fd = fopen(output, "w");
  if (out_fd == NULL)
  {
    fprintf(stderr, "Error opening file %s: %s\n", output, strerror(errno));
    exit(EXIT_FAILURE);
  }

  fprintf(out_fd, "P3\n");
  fprintf(out_fd, "%d %d\n255\n", image_width, image_height);
  
  //render logic
  point3_t pixel_color;

  for (int j = 0; j < image_height; j++)
  {
    for (int i = 0; i < image_width; i++)
    {
      pixel_color.x=0;
      pixel_color.y=0;
      pixel_color.z=0;

      for (int k = 0; k < num_samples; k++)
      {
        ray_t r = get_ray_sample(i, j, pixel00_loc, camera_center, pixel_delta_u, pixel_delta_v);
        pixel_color = vec3_sum(ray_color(r, world), pixel_color);
      }

      write_color(out_fd, vec3_div_sc(pixel_color, num_samples));
    }
  }

  /* chiude il file */
  fclose(out_fd);

  printf("\nDone.\n");
}
