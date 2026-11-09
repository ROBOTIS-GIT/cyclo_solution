# The base tag provides native linux/amd64 and linux/arm64 variants.
ARG BASE_IMAGE=robotis/ai-worker:2.2.7
FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-c"]
WORKDIR /root/ros2_ws
ARG CMAKE_BUILD_PARALLEL_LEVEL=2
# Inference dependencies belong to the image, not a mutable runtime volume.
# Host networking also works on Jetson hosts without Docker's iptables raw table.
COPY scripts/setup_pick_inference.sh /tmp/setup_pick_inference.sh
RUN --network=host bash /tmp/setup_pick_inference.sh && rm /tmp/setup_pick_inference.sh
COPY . /root/ros2_ws/src/demo_1f
# The Jetson base contains an empty DLA compiler placeholder; TensorRT links it
# even for GPU-only inference. Extract the matching L4T 36.4.3 library privately.
RUN --network=none if [[ "$(uname -m)" == aarch64 ]]; then \
      echo '928170c9e743257c063c235a610473cfec4be917a36f91a61a890ba2bfc47c94  /root/ros2_ws/src/demo_1f/docker/vendor/nvidia-l4t-dla-compiler.deb' | sha256sum -c - && \
      dpkg-deb -x /root/ros2_ws/src/demo_1f/docker/vendor/nvidia-l4t-dla-compiler.deb /tmp/demo-dla && \
      mkdir -p /opt/demo-trt/lib && \
      cp /tmp/demo-dla/usr/lib/aarch64-linux-gnu/nvidia/libnvdla_compiler.so /opt/demo-trt/lib/ && \
      echo /opt/demo-trt/lib > /etc/ld.so.conf.d/00-demo-trt.conf && ldconfig && \
      rm -rf /tmp/demo-dla; fi
# ROS dependencies are already in the base image; compilation stays offline.
# A derived demo image may contain old generated interfaces. Rebuild this package
# cleanly and normalize copied timestamps inside the image (host clocks can differ).
RUN --network=none source /opt/ros/jazzy/setup.bash && \
    source /root/ros2_ws/install/setup.bash && \
    rm -rf /root/ros2_ws/build/demo_1f /root/ros2_ws/install/demo_1f && \
    find /root/ros2_ws/src/demo_1f -type f -exec touch {} + && \
    CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL} colcon build --base-paths /root/ros2_ws/src/demo_1f /root/ros2_ws/src/demo_1f/modules/dynamixel_pick/ffw_* --packages-up-to demo_1f --cmake-args -DBUILD_TESTING=OFF && \
    chmod +x /root/ros2_ws/src/demo_1f/docker/entrypoint.sh && \
    printf '\nsource /root/ros2_ws/install/setup.bash\n' >> /root/.bashrc
# Override the base image's /init: do not start s6 or robot hardware services.
ENTRYPOINT ["/root/ros2_ws/src/demo_1f/docker/entrypoint.sh"]
# An interactive `docker compose exec demo_1f bash` loads the ROS overlay above.
CMD ["sleep", "infinity"]
