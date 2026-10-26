FROM chiyunnn/trail-odom:latest
# mesa-vulkan-drivers provides lavapipe (software Vulkan ICD) -- without it the Vulkan
# loader has no driver to load at all ("Unable to create Vulkan instance", a hard crash
# in Open3D's Filament-based GUI, not a tolerated fallback like rviz2's OGRE renderer
# gets via llvmpipe/OpenGL). Same category of software-rendering fix, different API.
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir open3d==0.20.0
