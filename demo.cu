// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// ---------------------------------------------------------

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "utils.hpp"

// CUDA kernel function to integrate a TSDF voxel volume given depth images
__global__
void Integrate(float * cam_K, float * cam2base, float * depth_im,
               int im_height, int im_width, int voxel_grid_dim_x, int voxel_grid_dim_y, int voxel_grid_dim_z,
               float voxel_grid_origin_x, float voxel_grid_origin_y, float voxel_grid_origin_z, float voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight) {

  int pt_grid_z = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  for (int pt_grid_x = 0; pt_grid_x < voxel_grid_dim_x; ++pt_grid_x) {

    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = voxel_grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = voxel_grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = voxel_grid_origin_z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
    float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
    int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= im_width || pt_pix_y < 0 || pt_pix_y >= im_height)
      continue;

    float depth_val = depth_im[pt_pix_y * im_width + pt_pix_x];

    float diff = depth_val - pt_cam_z;

    if (diff <= -trunc_margin)
      continue;

    // Integrate
    int volume_idx = pt_grid_z * voxel_grid_dim_y * voxel_grid_dim_x + pt_grid_y * voxel_grid_dim_x + pt_grid_x;
    float dist = fmin(1.0f, diff / trunc_margin);
    float weight_old = voxel_grid_weight[volume_idx];
    float weight_new = weight_old + 1.0f;
    voxel_grid_weight[volume_idx] = weight_new;
    voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
  }
}

// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
int main(int argc, char * argv[]) {
  // Location of folder containing RGB-D frames and camera pose files
  // std::string data_path = "data/rgbd-frames";
  std::string data_path = "data/seq-01";
  // std::string data_path = "fountain";

  // Location of camera intrinsic file
  std::string cam_K_file = "data/camera-intrinsics.txt";
  if(data_path == "fountain"){
    cam_K_file = "fountain/camera-intrinsics.txt";
  }

  // int base_frame_idx = 150;
  // int first_frame_idx = 150;
  // float num_frames = 50;// 1
  int base_frame_idx = 0;
  int first_frame_idx = 0;
  float num_frames = 999;
  if(data_path == "fountain"){
    base_frame_idx = 2;
    first_frame_idx = 2;
    num_frames = 5;
   // base_frame_idx = 5;//2;
   //  first_frame_idx = 5;//2;
   //  num_frames = 4;//7;
   }

  float cam_K[3 * 3];
  float base2world[4 * 4];
  float cam2base[4 * 4];
  float cam2world[4 * 4];
  int im_width = 640;
  int im_height = 480;
  if(data_path == "fountain"){
    im_width = 768;//640;
    im_height = 512;//480;
  }
  float depth_im[im_height * im_width];

  // Voxel grid parameters (change these to change voxel grid resolution, etc.)
  float voxel_grid_origin_x = -1.5f; // Location of voxel grid origin in base frame camera coordinates
  float voxel_grid_origin_y = -1.5f;
  float voxel_grid_origin_z = 0.5f;
  if(data_path == "fountain"){
    voxel_grid_origin_x = -4.0f;// position in base frame
    voxel_grid_origin_y = -4.0f;
    voxel_grid_origin_z = 4.0f;
  }
  float voxel_size = 0.006f;
  float trunc_margin = voxel_size * 5;
  if(data_path == "fountain"){
    voxel_size = 0.090f;// 0.030f;
    trunc_margin = voxel_size * 5;
  }
  int voxel_grid_dim_x = 500;
  int voxel_grid_dim_y = 500;
  int voxel_grid_dim_z = 500;
  // if(data_path == "fountain"){
  //   voxel_grid_dim_x = 1000;
  //   voxel_grid_dim_y = 1000;
  //   voxel_grid_dim_z = 1000;
  // }

  // Manual parameters
  if (argc > 1) {
    cam_K_file = argv[1];
    data_path = argv[2];
    base_frame_idx = atoi(argv[3]);
    first_frame_idx = atoi(argv[4]);
    num_frames = atof(argv[5]);
    voxel_grid_origin_x = atof(argv[6]);
    voxel_grid_origin_y = atof(argv[7]);
    voxel_grid_origin_z = atof(argv[8]);
    voxel_size = atof(argv[9]);
    trunc_margin = atof(argv[10]);
  }

  // Read camera intrinsics
  std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
  std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);

  // Read base frame camera pose
  std::ostringstream base_frame_prefix;
  if(data_path == "data/rgbd-frames") base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
  else if(data_path == "data/seq-01") base_frame_prefix << std::setw(6) << std::setfill('0') << base_frame_idx;
  else if(data_path == "fountain") base_frame_prefix << base_frame_idx;
  else std::cerr << "not found data_path" << std::endl;

  std::string base2world_file = data_path + "/frame-" + base_frame_prefix.str() + ".pose.txt";
  if(data_path == "fountain"){
    // base2world_file = data_path + "/inv" + base_frame_prefix.str() + "-pose-cw.txt";// from world to base
    base2world_file = data_path + "/inv" + base_frame_prefix.str() + "-pose-wc.txt";// from base to world
  }
  std::vector<float> base2world_vec = LoadMatrixFromFile(base2world_file, 4, 4);
  std::copy(base2world_vec.begin(), base2world_vec.end(), base2world);

  // Invert base frame camera pose to get world-to-base frame transform 
  float base2world_inv[16] = {0};
  invert_matrix(base2world, base2world_inv);

  // Initialize voxel grid
  float * voxel_grid_TSDF = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  float * voxel_grid_weight = new float[voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z];
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    voxel_grid_TSDF[i] = 1.0f;
  memset(voxel_grid_weight, 0, sizeof(float) * voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z);

  // Load variables to GPU memory
  float * gpu_voxel_grid_TSDF;
  float * gpu_voxel_grid_weight;
  cudaMalloc(&gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  cudaMalloc(&gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float));
  checkCUDA(__LINE__, cudaGetLastError());
  cudaMemcpy(gpu_voxel_grid_TSDF, voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_voxel_grid_weight, voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyHostToDevice);
  checkCUDA(__LINE__, cudaGetLastError());
  float * gpu_cam_K;
  float * gpu_cam2base;
  float * gpu_depth_im;
  cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
  cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
  cudaMalloc(&gpu_depth_im, im_height * im_width * sizeof(float));
  checkCUDA(__LINE__, cudaGetLastError());

  // Loop through each depth frame and integrate TSDF voxel grid
  for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; ++frame_idx) {

    std::ostringstream curr_frame_prefix;
    if(data_path == "data/rgbd-frames") curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
    else if(data_path == "data/seq-01") curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;
    else if(data_path == "fountain") curr_frame_prefix << frame_idx;
    else std::cerr << "not found data_path" << std::endl;

    // // Read current frame depth
    std::string depth_im_file = data_path + "/frame-" + curr_frame_prefix.str() + ".depth.png";
    if(data_path == "fountain"){
      // depth_im_file = data_path + "/inv" + curr_frame_prefix.str() + ".png";
      depth_im_file = data_path + "/000" + curr_frame_prefix.str() + "/Depth0001.exr";
    }
    ReadDepth(depth_im_file, im_height, im_width, depth_im);

    // Read base frame camera pose
    std::string cam2world_file = data_path + "/frame-" + curr_frame_prefix.str() + ".pose.txt";
    if(data_path == "fountain"){
      // cam2world_file = data_path + "/inv" + curr_frame_prefix.str() + "-pose-cw.txt";// from world to camera
      cam2world_file = data_path + "/inv" + curr_frame_prefix.str() + "-pose-wc.txt";// from camera to world
    }
    std::vector<float> cam2world_vec = LoadMatrixFromFile(cam2world_file, 4, 4);
    // for (auto i: cam2world_vec){
    //   std::cout << "pose: " << i << std::endl;
    // }
    std::copy(cam2world_vec.begin(), cam2world_vec.end(), cam2world);

    // Compute relative camera pose (camera-to-base frame) // cam2base: from camera to base
    multiply_matrix(base2world_inv, cam2world, cam2base);
    // // debug
    // for(int d=0; d < 16; d++){
    //   std::cout << "debug pose: " << cam2base[d] << std::endl;
    // }


    cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_depth_im, depth_im, im_height * im_width * sizeof(float), cudaMemcpyHostToDevice);
    checkCUDA(__LINE__, cudaGetLastError());

    std::cout << "Fusing: " << depth_im_file << std::endl;

    Integrate <<< voxel_grid_dim_z, voxel_grid_dim_y >>>(gpu_cam_K, gpu_cam2base, gpu_depth_im,
                                                         im_height, im_width, voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z,
                                                         voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z, voxel_size, trunc_margin,
                                                         gpu_voxel_grid_TSDF, gpu_voxel_grid_weight);
  }

  // Load TSDF voxel grid from GPU to CPU memory
  cudaMemcpy(voxel_grid_TSDF, gpu_voxel_grid_TSDF, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(voxel_grid_weight, gpu_voxel_grid_weight, voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z * sizeof(float), cudaMemcpyDeviceToHost);
  checkCUDA(__LINE__, cudaGetLastError());

  // Compute surface points from TSDF voxel grid and save to point cloud .ply file
  std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
  SaveVoxelGrid2SurfacePointCloud("tsdf.ply", voxel_grid_dim_x, voxel_grid_dim_y, voxel_grid_dim_z, 
                                  voxel_size, voxel_grid_origin_x, voxel_grid_origin_y, voxel_grid_origin_z,
                                  voxel_grid_TSDF, voxel_grid_weight, 0.2f, 0.0f);

  // Save TSDF voxel grid and its parameters to disk as binary file (float array)
  std::cout << "Saving TSDF voxel grid values to disk (tsdf.bin)..." << std::endl;
  std::string voxel_grid_saveto_path = "tsdf.bin";
  std::ofstream outFile(voxel_grid_saveto_path, std::ios::binary | std::ios::out);
  float voxel_grid_dim_xf = (float) voxel_grid_dim_x;
  float voxel_grid_dim_yf = (float) voxel_grid_dim_y;
  float voxel_grid_dim_zf = (float) voxel_grid_dim_z;
  outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
  outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
  outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_x, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_y, sizeof(float));
  outFile.write((char*)&voxel_grid_origin_z, sizeof(float));
  outFile.write((char*)&voxel_size, sizeof(float));
  outFile.write((char*)&trunc_margin, sizeof(float));
  for (int i = 0; i < voxel_grid_dim_x * voxel_grid_dim_y * voxel_grid_dim_z; ++i)
    outFile.write((char*)&voxel_grid_TSDF[i], sizeof(float));
  outFile.close();

  return 0;
}


