// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-FileCopyrightText: Copyright 2026 ROBOTIS CO., LTD.
// SPDX-License-Identifier: Apache-2.0
//
// This file is derived from NVIDIA Isaac ROS cuMotion:
// https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_cumotion
//
// Original work Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Modifications Copyright 2026 ROBOTIS CO., LTD.
//
// Original Author: Karanbir Chahal
// Modification Author: Yeonguk Kim

#include "cyclo_cumotion_robot_segmenter/robot_segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"
#include "cumotion/kinematics.h"

namespace nvidia
{
namespace isaac_ros
{
namespace manipulator
{

namespace
{
constexpr const char kDefaultQoS[] = "DEFAULT";
constexpr int kTfLookupTimeoutSeconds = 30;
constexpr int64_t kDefaultTfLookupTimeoutMilliseconds = 200;
constexpr int64_t kDefaultTfPollPeriodMilliseconds = 5;
constexpr int64_t kDefaultPendingDepthQueueSize = 10;
}  // namespace

RobotSegmenter::RobotSegmenter(const rclcpp::NodeOptions & options)
: rclcpp::Node("robot_segmenter", options),
  sync_queue_size_(declare_parameter<int>("sync_queue_size", 10)),
  input_qos_{::isaac_ros::common::AddQosParameter(
      *this, kDefaultQoS, "input_qos")},
  output_qos_{::isaac_ros::common::AddQosParameter(
      *this, kDefaultQoS, "output_qos")},
  memory_pool_num_blocks_(declare_parameter<int64_t>("memory_pool_num_blocks", 40)),
  joint_limit_tolerance_(declare_parameter<double>("joint_limit_tolerance", 0.02)),
  enable_performance_logging_(declare_parameter<bool>("enable_performance_logging", false)),
  additional_buffer_distance_(declare_parameter<double>("additional_buffer_distance", 0.0)),
  robot_base_frame_(declare_parameter<std::string>("robot_base_frame", "base_link")),
  robot_description_service_name_(declare_parameter<std::string>(
      "robot_description_service_name",
      "/cumotion/get_robot_description"))
{
  if (memory_pool_num_blocks_ <= 0) {
    throw std::runtime_error("memory_pool_num_blocks must be greater than 0");
  }
  if (joint_limit_tolerance_ < 0.0) {
    throw std::runtime_error("joint_limit_tolerance must be non-negative");
  }

  // Load robot description from URDF and XRDF files
  const std::string urdf_path = declare_parameter<std::string>("urdf_path", "");
  const std::string xrdf_path = declare_parameter<std::string>("xrdf_path", "");

  if (urdf_path.empty() || xrdf_path.empty()) {
    throw std::runtime_error("urdf_path and xrdf_path parameters must be provided");
  }

  if (robot_base_frame_.empty()) {
    throw std::runtime_error("robot_base_frame parameter must be provided");
  }

  const int64_t tf_lookup_timeout_ms = declare_parameter<int64_t>(
    "tf_lookup_timeout_ms", kDefaultTfLookupTimeoutMilliseconds);
  const int64_t tf_poll_period_ms = declare_parameter<int64_t>(
    "tf_poll_period_ms", kDefaultTfPollPeriodMilliseconds);
  const int64_t pending_depth_queue_size = declare_parameter<int64_t>(
    "pending_depth_queue_size", kDefaultPendingDepthQueueSize);
  if (tf_lookup_timeout_ms <= 0 || tf_poll_period_ms <= 0 || pending_depth_queue_size <= 0) {
    throw std::runtime_error(
            "tf_lookup_timeout_ms, tf_poll_period_ms, and pending_depth_queue_size "
            "must be greater than zero");
  }
  tf_lookup_timeout_ = std::chrono::milliseconds(tf_lookup_timeout_ms);
  tf_poll_period_ = std::chrono::milliseconds(tf_poll_period_ms);
  pending_depth_queue_size_ = static_cast<size_t>(pending_depth_queue_size);

  RCLCPP_INFO(
    get_logger(), "Loading robot description from URDF: %s, XRDF: %s",
    urdf_path.c_str(), xrdf_path.c_str());

  robot_description_ = cumotion::LoadRobotFromFile(xrdf_path, urdf_path);
  if (!robot_description_) {
    throw std::runtime_error("Failed to load robot description from URDF and XRDF files");
  }

  RCLCPP_INFO(
    get_logger(), "Robot description loaded successfully with %d joints",
    robot_description_->numCSpaceCoords());

  CacheRobotCspaceMetadata();

  rclcpp::SubscriptionOptions sub_options;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  depth_sub_ = create_subscription<Nitros::NitrosImage>(
    "depth_image", input_qos_,
    std::bind(&RobotSegmenter::DepthCallback, this, std::placeholders::_1), sub_options);
  depth_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    "camera_info_depth", input_qos_,
    std::bind(&RobotSegmenter::DepthCameraInfoCallback, this, std::placeholders::_1), sub_options);

  cuda_stream_ = ::nvidia::isaac_ros::common::createCudaStream(
    "isaac_ros_cumotion_robot_segmenter_stream");

  rclcpp::SubscriptionOptions sub_options_joint_state;
  sub_options_joint_state.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  sub_options_joint_state.callback_group = create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);

  joint_state_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", input_qos_,
    std::bind(&RobotSegmenter::JointStateCallback, this, std::placeholders::_1),
    sub_options_joint_state);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  pending_depth_timer_ = create_wall_timer(
    tf_poll_period_, std::bind(&RobotSegmenter::ProcessPendingDepthFrames, this));

  rclcpp::PublisherOptions pub_options;
  pub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;
  robot_mask_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "robot_mask", output_qos_, pub_options);
  robot_depth_pub_ = create_publisher<nvidia::isaac_ros::nitros::NitrosImage>(
    "robot_depth", output_qos_, pub_options);

  // Topic name for receiving reload robot description signal
  const std::string reload_topic_name = declare_parameter<std::string>(
    "reload_robot_description_topic", "reload_robot_description");
  const std::string reload_ack_topic_name = declare_parameter<std::string>(
    "reload_robot_description_ack_topic", "reload_robot_description_ack");

  // Create callback group for service client
  service_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Create service client for getting robot description
  get_robot_description_client_ =
    create_client<isaac_ros_cumotion_interfaces::srv::GetRobotDescription>(
    robot_description_service_name_,
    rclcpp::ServicesQoS(),
    service_cb_group_);

  // Subscribe to reload robot description topic
  const auto reload_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  reload_robot_description_sub_ = create_subscription<std_msgs::msg::UInt64>(
    reload_topic_name,
    reload_qos,
    std::bind(&RobotSegmenter::ReloadRobotDescriptionCallback, this, std::placeholders::_1),
    sub_options);
  reload_ack_pub_ = create_publisher<std_msgs::msg::String>(
    reload_ack_topic_name, rclcpp::QoS(10).reliable());

  RCLCPP_INFO(
    get_logger(),
    "Robot segmenter initialized. Listening for generation reloads on topic '%s', "
    "acknowledging on '%s', and fetching descriptions from service '%s'. "
    "TF wait: %ld ms, poll: %ld ms, queue: %zu",
    reload_topic_name.c_str(), reload_ack_topic_name.c_str(),
    robot_description_service_name_.c_str(),
    static_cast<long>(tf_lookup_timeout_.count()),
    static_cast<long>(tf_poll_period_.count()), pending_depth_queue_size_);
}

void RobotSegmenter::CacheRobotCspaceMetadata()
{
  ordered_joint_names_.clear();
  ordered_joint_limits_.clear();
  auto kinematics = robot_description_->kinematics();
  const auto cspace_coords = robot_description_->numCSpaceCoords();
  ordered_joint_names_.reserve(static_cast<size_t>(cspace_coords));
  ordered_joint_limits_.reserve(static_cast<size_t>(cspace_coords));
  for (int i = 0; i < cspace_coords; ++i) {
    const std::string name = robot_description_->cSpaceCoordName(i);
    const auto limits = kinematics->cSpaceCoordLimits(i);
    RCLCPP_DEBUG(
      get_logger(), "Cspace coord %d: %s [%.6f, %.6f]", i, name.c_str(),
      limits.lower, limits.upper);
    ordered_joint_names_.push_back(name);
    ordered_joint_limits_.emplace_back(limits.lower, limits.upper);
  }
}

void RobotSegmenter::InitializeCameraPose(const sensor_msgs::msg::CameraInfo & depth_camera_info)
{
  const std::string camera_frame = depth_camera_info.header.frame_id;

  RCLCPP_INFO(
    get_logger(),
    "Waiting for transform from %s to %s...",
    camera_frame.c_str(),
    robot_base_frame_.c_str());

  // lookupTransform will block until transform is available or timeout expires
  geometry_msgs::msg::TransformStamped stamped;
  try {
    stamped = tf_buffer_->lookupTransform(
      robot_base_frame_, camera_frame,
      tf2::TimePointZero,
      std::chrono::seconds(kTfLookupTimeoutSeconds));
  } catch (const tf2::TransformException & ex) {
    throw std::runtime_error(
            "Failed to lookup transform from '" + camera_frame + "' to '" +
            robot_base_frame_ + "' after " + std::to_string(kTfLookupTimeoutSeconds) +
            " seconds: " + ex.what());
  }

  const Eigen::Isometry3d eig_transform = tf2::transformToEigen(stamped);
  Eigen::Matrix4d robot_pose_camera = eig_transform.matrix();

  std::stringstream ss;
  ss << robot_pose_camera;
  RCLCPP_INFO(get_logger(), "Camera pose in robot frame:\n%s", ss.str().c_str());
  robot_pose_camera_ = cumotion::Pose3(robot_pose_camera);
}

bool RobotSegmenter::UpdateCameraPose(
  const std::string & camera_frame,
  const tf2::TimePoint & timestamp,
  const tf2::Duration & timeout,
  std::string * error_message)
{
  geometry_msgs::msg::TransformStamped stamped;
  try {
    stamped = tf_buffer_->lookupTransform(
      robot_base_frame_, camera_frame, timestamp, timeout);
  } catch (const tf2::TransformException & ex) {
    if (error_message != nullptr) {
      *error_message = ex.what();
    }
    return false;
  }

  const Eigen::Isometry3d eig_transform = tf2::transformToEigen(stamped);
  robot_pose_camera_ = cumotion::Pose3(eig_transform.matrix());
  return true;
}

RobotSegmenter::~RobotSegmenter()
{
  CHECK_CUDA_ERROR(cudaStreamSynchronize(*cuda_stream_), "Failed to synchronize CUDA stream");
}

bool RobotSegmenter::ComputeAndPublishRobotMask(
  const Nitros::NitrosImage & depth_msg,
  const sensor_msgs::msg::CameraInfo & depth_camera_info,
  const sensor_msgs::msg::JointState & joint_state)
{
  auto start_time = std::chrono::high_resolution_clock::now();

  RCLCPP_DEBUG(get_logger(), "Starting robot segmenter computation");

  const int depth_w = static_cast<int>(depth_msg.width);
  const int depth_h = static_cast<int>(depth_msg.height);

  // Check if buffers are allocated and have correct size
  if (buffer_width_ != depth_w || buffer_height_ != depth_h) {
    RCLCPP_WARN(
      get_logger(), "Buffer size mismatch. Expected %dx%d, got %dx%d. Skipping frame.",
      buffer_width_, buffer_height_, depth_w, depth_h);
    return false;
  }

  if (!robot_segmenter_) {
    RCLCPP_WARN(get_logger(), "Robot segmenter not initialized yet, dropping frame");
    return false;
  }

  const std::string encoding = depth_msg.encoding;
  const size_t buffer_size = depth_msg.get_data_size();

  is_mono_ = encoding == sensor_msgs::image_encodings::TYPE_32FC1 ? false : true;

  const size_t num_pixels = static_cast<size_t>(depth_w) * static_cast<size_t>(depth_h);
  const size_t current_size = is_mono_ ?
    cpu_input_buffer_uint16_.size() :
    cpu_input_buffer_float_.size();

  if (num_pixels != current_size) {
    // Resize CPU buffers (element count, not bytes).
    if (is_mono_) {
      cpu_input_buffer_uint16_.resize(num_pixels);
      cpu_output_buffer_uint16_.resize(num_pixels);
      cpu_mask_buffer_uint16_.resize(num_pixels);
    } else {
      cpu_input_buffer_float_.resize(num_pixels);
      cpu_output_buffer_float_.resize(num_pixels);
      cpu_mask_buffer_float_.resize(num_pixels);
    }
    RCLCPP_INFO(
      get_logger(), "Resized CPU buffers to %zu elements (%zu bytes)",
      num_pixels, buffer_size);
  }

  // Create joint positions vector (needed for segmentation lambda)
  std::unordered_map<std::string, size_t> index_by_name;
  index_by_name.reserve(joint_state.name.size());
  for (size_t i = 0; i < joint_state.name.size(); ++i) {
    index_by_name[joint_state.name[i]] = i;
  }
  Eigen::VectorXd joint_positions(static_cast<Eigen::Index>(ordered_joint_names_.size()));
  for (size_t i = 0; i < ordered_joint_names_.size(); ++i) {
    const std::string & name = ordered_joint_names_[i];
    const auto index_it = index_by_name.find(name);
    if (index_it == index_by_name.end() || index_it->second >= joint_state.position.size()) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Dropping depth frame because joint state is missing '%s'", name.c_str());
      return false;
    }
    const double raw_position = joint_state.position[index_it->second];
    if (!std::isfinite(raw_position)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Dropping depth frame because joint '%s' is not finite", name.c_str());
      return false;
    }
    const auto [lower, upper] = ordered_joint_limits_[i];
    const double safe_position = std::clamp(raw_position, lower, upper);
    const double violation = std::abs(raw_position - safe_position);
    if (violation > joint_limit_tolerance_) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Dropping depth frame because joint '%s'=%.6f exceeds cuMotion limits "
        "[%.6f, %.6f] by %.6f", name.c_str(), raw_position, lower, upper, violation);
      return false;
    }
    if (violation > 0.0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Clamping joint '%s' from %.6f to %.6f for robot segmentation",
        name.c_str(), raw_position, safe_position);
    }
    joint_positions[static_cast<Eigen::Index>(i)] = safe_position;
  }

  // Create output buffer pool, assuming the size of the buffer doesn't change
  if (!pool_.initialized()) {
    cudaError_t err = pool_.create(
      static_cast<size_t>(buffer_size), static_cast<size_t>(memory_pool_num_blocks_),
      nvidia::isaac_ros::nitros::CUDAMemoryPool::MemoryType::Device);
    CHECK_CUDA_ERROR(err, "Failed to initialize output buffer pool");
  }

  // Generic lambda to handle both uint16_t and float types
  // This avoids code duplication - compiler instantiates two versions
  auto process_depth = [&](auto * input_buf, auto * output_buf, auto * mask_buf) {
      using T = std::remove_pointer_t<decltype(input_buf)>;
      const size_t dense_buffer_size = num_pixels * sizeof(T);

      // Copy input data from GPU to CPU buffer
      RCLCPP_DEBUG(
        get_logger(), "Copying depth image from GPU to CPU, buffer size: %zu bytes",
        dense_buffer_size);
      if (buffer_size != dense_buffer_size) {
        throw std::runtime_error("Depth image must be tightly packed");
      }

      auto depth_read_handle = depth_msg.get_read_handle(*cuda_stream_);
      CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
          input_buf, depth_read_handle.get_ptr(),
          buffer_size, cudaMemcpyDefault, *cuda_stream_),
        "Failed to copy depth image from GPU to CPU");
      CHECK_CUDA_ERROR(
        cudaStreamSynchronize(*cuda_stream_),
        "Failed to synchronize CUDA stream after copying depth image");

      const std::string output_encoding = std::is_same_v<T, uint16_t> ?
        sensor_msgs::image_encodings::TYPE_16UC1 :
        sensor_msgs::image_encodings::TYPE_32FC1;

      const size_t bytes_per_element =
        sensor_msgs::image_encodings::bitDepth(output_encoding) / CHAR_BIT;
      const size_t num_channels = sensor_msgs::image_encodings::numChannels(output_encoding);

      auto depth_output_image = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
      size_t output_step = depth_w * num_channels * bytes_per_element;
      auto depth_write_handle = depth_output_image->from_pool(
        pool_, depth_w, depth_h, output_step, output_encoding, *cuda_stream_);
      auto mask_output_image = std::make_unique<nvidia::isaac_ros::nitros::NitrosImage>();
      auto mask_write_handle = mask_output_image->from_pool(
        pool_, depth_w, depth_h, output_step, output_encoding, *cuda_stream_);

      // Create CPU DepthImage views with HOST residency
      const auto input_depth_image = cumotion::CreateDepthImageView(
        input_buf, depth_w, depth_h, std::nullopt, cumotion::BufferResidency::HOST);
      auto output_depth_image = cumotion::CreateDepthImageView(
        output_buf, depth_w, depth_h, std::nullopt, cumotion::BufferResidency::HOST);
      auto output_mask = cumotion::CreateDepthImageView(
        mask_buf, depth_w, depth_h, std::nullopt, cumotion::BufferResidency::HOST);

      // Call robot segmenter (operates on CPU buffers)
      robot_segmenter_->segmentDepthImage(
        robot_pose_camera_,
        joint_positions,
        input_depth_image,
        &output_depth_image,
        &output_mask
      );

      // Allocate GPU memory and copy output
      T * gpu_aligned_depth = reinterpret_cast<T *>(depth_write_handle.get_ptr());
      T * gpu_aligned_mask = reinterpret_cast<T *>(mask_write_handle.get_ptr());
      CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
          gpu_aligned_mask, mask_buf, buffer_size, cudaMemcpyDefault, *cuda_stream_),
        "Failed to copy segmented mask from CPU to GPU");
      CHECK_CUDA_ERROR(
        cudaMemcpyAsync(
          gpu_aligned_depth, output_buf, buffer_size, cudaMemcpyDefault, *cuda_stream_),
        "Failed to copy segmented depth from CPU to GPU");
      CHECK_CUDA_ERROR(
        cudaStreamSynchronize(*cuda_stream_),
        "Failed to synchronize CUDA stream after copying segmented outputs");

      // Build and publish NitrosImage with GPU data
      mask_output_image->frame_id = depth_camera_info.header.frame_id;
      mask_output_image->set_timestamp_sec(depth_msg.timestamp_sec);
      mask_output_image->set_timestamp_nsec(depth_msg.timestamp_nsec);
      depth_output_image->frame_id = depth_camera_info.header.frame_id;
      depth_output_image->set_timestamp_sec(depth_msg.timestamp_sec);
      depth_output_image->set_timestamp_nsec(depth_msg.timestamp_nsec);

      // Note that for uint16_t 0 = robot, max uint16_t value = background.
      // For float16 0 = robot, 1 = background.
      robot_mask_pub_->publish(std::move(mask_output_image));
      robot_depth_pub_->publish(std::move(depth_output_image));
    };

  // Call with appropriate buffer types based on encoding
  if (is_mono_) {
    process_depth(
      cpu_input_buffer_uint16_.data(),
      cpu_output_buffer_uint16_.data(),
      cpu_mask_buffer_uint16_.data());
  } else {
    process_depth(
      cpu_input_buffer_float_.data(),
      cpu_output_buffer_float_.data(),
      cpu_mask_buffer_float_.data());
  }

  RCLCPP_DEBUG(get_logger(), "Published robot mask");
  if (enable_performance_logging_) {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    RCLCPP_DEBUG(
      get_logger(), "Robot segmenter computation took %ld milliseconds", duration.count());
  }
  return true;
}

void RobotSegmenter::DepthCallback(const Nitros::NitrosImage::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(node_mutex_);

  if (!depth_camera_info_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Received depth image but don't have depth camera info !");
    return;
  } else if (!joint_state_.has_value()) {
    RCLCPP_DEBUG(get_logger(), "Have not received joint state yet !");
    return;
  } else {
    if (pending_depth_frames_.size() >= pending_depth_queue_size_) {
      pending_depth_frames_.pop_front();
      RCLCPP_DEBUG_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Dropping oldest depth frame because the TF wait queue is full "
        "(capacity=%zu)", pending_depth_queue_size_);
    }
    pending_depth_frames_.push_back(
      PendingDepthFrame{msg, joint_state_.value(), std::chrono::steady_clock::now()});
  }
}

void RobotSegmenter::ProcessPendingDepthFrames()
{
  std::lock_guard<std::mutex> lock(node_mutex_);
  const auto now = std::chrono::steady_clock::now();
  std::string oldest_transform_error;

  // Select the newest frame for which an exact transform is available. A
  // FIFO policy builds latency whenever camera input is faster than TF; that
  // stale backlog is visible as jerky nvblox updates.
  for (size_t reverse_index = 0; reverse_index < pending_depth_frames_.size(); ++reverse_index) {
    const size_t index = pending_depth_frames_.size() - 1U - reverse_index;
    PendingDepthFrame & pending = pending_depth_frames_[index];
    const int64_t depth_timestamp_ns =
      static_cast<int64_t>(pending.depth->timestamp_sec) * 1000000000LL +
      static_cast<int64_t>(pending.depth->timestamp_nsec);
    const tf2::TimePoint depth_timestamp{tf2::Duration(depth_timestamp_ns)};
    std::string transform_error;

    if (UpdateCameraPose(
        depth_camera_info_->header.frame_id, depth_timestamp,
        tf2::Duration::zero(), &transform_error))
    {
      auto depth = pending.depth;
      auto joint_state = std::move(pending.joint_state);
      pending_depth_frames_.erase(
        pending_depth_frames_.begin(), pending_depth_frames_.begin() + index + 1U);
      try {
        ComputeAndPublishRobotMask(*depth, depth_camera_info_.value(), joint_state);
      } catch (const std::exception & exception) {
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Robot segmentation failed without terminating the container: %s",
          exception.what());
      }
      return;
    }
    if (index == 0U) {
      oldest_transform_error = std::move(transform_error);
    }
  }

  while (!pending_depth_frames_.empty()) {
    PendingDepthFrame & pending = pending_depth_frames_.front();
    const auto wait_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - pending.enqueue_time);
    if (wait_time < tf_lookup_timeout_) {
      return;
    }

    RCLCPP_DEBUG_THROTTLE(
      get_logger(), *get_clock(), 5000,
      "Dropping depth frame after waiting %lld ms for transform from '%s' to '%s' "
      "at the image timestamp: %s",
      static_cast<long long>(wait_time.count()),
      depth_camera_info_->header.frame_id.c_str(), robot_base_frame_.c_str(),
      oldest_transform_error.c_str());
    pending_depth_frames_.pop_front();
  }
}

void RobotSegmenter::DepthCameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(node_mutex_);

  if (depth_camera_info_.has_value()) {
    RCLCPP_DEBUG_ONCE(get_logger(), "Received depth camera info but already have it !");
    return;
  }

  const sensor_msgs::msg::CameraInfo & depth_camera_info = *msg;

  // Create and cache cuMotion CameraIntrinsics from the ROS CameraInfo K matrix
  // K is a 3x3 matrix in row-major order: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
  const double fx = depth_camera_info.k[0];
  const double cx = depth_camera_info.k[2];
  const double fy = depth_camera_info.k[4];
  const double cy = depth_camera_info.k[5];
  depth_camera_intrinsics_ = cumotion::CameraIntrinsics(fx, fy, cx, cy);

  InitializeCameraPose(depth_camera_info);

  // Create the robot segmenter now that we have camera intrinsics
  if (!robot_segmenter_ && robot_description_) {
    RCLCPP_INFO(
      get_logger(), "Creating robot segmenter with camera intrinsics: "
      "fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", fx, fy, cx, cy);

    robot_segmenter_ = cumotion::CreateRobotSegmenter(
      *robot_description_,
      depth_camera_intrinsics_.value(),
      additional_buffer_distance_,
      cumotion::RobotSegmenter::RobotGeometryKind::WORLD_COLLISION_SPHERES);

    if (robot_segmenter_) {
      RCLCPP_INFO(get_logger(), "Robot segmenter created successfully");
    } else {
      throw std::runtime_error("Failed to create robot segmenter");
    }

    // Preallocate CPU buffers for input and output images
    // The segmenter requires HOST residency buffers
    buffer_width_ = static_cast<int>(depth_camera_info.width);
    buffer_height_ = static_cast<int>(depth_camera_info.height);
  }

  depth_camera_info_ = depth_camera_info;
}

void RobotSegmenter::JointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(node_mutex_);
  joint_state_ = *msg;
}

void RobotSegmenter::ReloadRobotDescriptionCallback(const std_msgs::msg::UInt64::SharedPtr msg)
{
  if (msg->data == 0U) {
    RCLCPP_WARN(get_logger(), "Received invalid robot-description generation 0, ignoring");
    return;
  }

  RCLCPP_INFO(
    get_logger(), "Received robot-description reload generation %llu",
    static_cast<unsigned long long>(msg->data));
  FetchAndReinitializeRobotDescription(msg->data);
}

void RobotSegmenter::FetchAndReinitializeRobotDescription(uint64_t generation)
{
  // Check if service is available (non-blocking check)
  if (!get_robot_description_client_->service_is_ready()) {
    RCLCPP_ERROR(
      get_logger(),
      "Service '%s' not available",
      robot_description_service_name_.c_str());
    return;
  }

  // Create request
  auto request =
    std::make_shared<isaac_ros_cumotion_interfaces::srv::GetRobotDescription::Request>();

  RCLCPP_INFO(get_logger(), "Sending async request to GetRobotDescription service...");

  // Call service asynchronously with a callback to avoid deadlock
  // (synchronous wait from within a callback blocks the executor)
  auto future = get_robot_description_client_->async_send_request(
    request,
    [this, generation](rclcpp::Client<isaac_ros_cumotion_interfaces::srv::GetRobotDescription>::
    SharedFuture response_future) {
      try {
        auto response = response_future.get();
        if (response->urdf.empty() || response->xrdf.empty()) {
          RCLCPP_ERROR(
            get_logger(),
            "Service returned empty URDF or XRDF, cannot reinitialize robot segmenter");
          return;
        }

        RCLCPP_INFO(
          get_logger(),
          "Received robot description from service (URDF: %zu bytes, XRDF: %zu bytes)",
          response->urdf.size(), response->xrdf.size());

        // Reinitialize with the new robot description
        if (!InitializeRobotSegmenterWithCustomDescription(response->urdf, response->xrdf)) {
          return;
        }
        std_msgs::msg::String ack;
        ack.data = std::to_string(generation) + "|" + get_fully_qualified_name();
        reload_ack_pub_->publish(ack);
        RCLCPP_INFO(
          get_logger(), "Acknowledged robot-description generation %llu",
          static_cast<unsigned long long>(generation));
      } catch (const std::exception & e) {
        RCLCPP_ERROR(get_logger(), "Service call failed: %s", e.what());
      }
    });
}

bool RobotSegmenter::InitializeRobotSegmenterWithCustomDescription(
  const std::string & urdf,
  const std::string & xrdf)
{
  std::lock_guard<std::mutex> lock(node_mutex_);

  RCLCPP_INFO(get_logger(), "Reinitializing robot description from strings");

  // Load robot description from URDF and XRDF strings
  auto new_robot_description = cumotion::LoadRobotFromMemory(xrdf, urdf);
  if (!new_robot_description) {
    RCLCPP_ERROR(get_logger(), "Failed to load robot description from provided URDF and XRDF");
    return false;
  }

  // Update robot description
  robot_description_ = std::move(new_robot_description);

  RCLCPP_INFO(
    get_logger(), "Robot description reloaded successfully with %d joints",
    robot_description_->numCSpaceCoords());

  CacheRobotCspaceMetadata();

  // Recreate robot segmenter if we have camera intrinsics
  if (depth_camera_intrinsics_.has_value()) {
    RCLCPP_INFO(get_logger(), "Recreating robot segmenter with existing camera intrinsics");

    robot_segmenter_ = cumotion::CreateRobotSegmenter(
      *robot_description_,
      depth_camera_intrinsics_.value(),
      additional_buffer_distance_,
      cumotion::RobotSegmenter::RobotGeometryKind::WORLD_COLLISION_SPHERES);

    if (robot_segmenter_) {
      RCLCPP_INFO(get_logger(), "Robot segmenter recreated successfully");
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to recreate robot segmenter");
      return false;
    }
  } else {
    RCLCPP_INFO(
      get_logger(),
      "Camera intrinsics not yet available, robot segmenter will be created when they arrive");
    robot_segmenter_.reset();
  }
  return true;
}

}  // namespace manipulator
}  // namespace isaac_ros
}  // namespace nvidia

// Register as component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::manipulator::RobotSegmenter)
