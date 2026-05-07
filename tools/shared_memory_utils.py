# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
A simplified multi-image shared memory tool module
When writing, concatenate three images (head, left, right) horizontally and write them
When reading, split the concatenated image into three independent images
"""

import ctypes
import time
import numpy as np
import cv2
from multiprocessing import shared_memory, resource_tracker
from typing import Optional, Dict, List
import struct
import os


def _open_shm_no_track(name: str) -> shared_memory.SharedMemory:
    """Open an existing SHM by name, opting out of resource_tracker.

    Python's multiprocessing.shared_memory auto-registers every open
    (creator or reader) with resource_tracker, which calls shm_unlink()
    on the SHM when this process exits — destroying it for any other
    process that's still using it. That's correct for the creator, but
    catastrophic for short-lived reader scripts (debug viewers, probes)
    that happen to open the SHM by name.

    Calling this helper instead of SharedMemory(name=...) on the open
    path means a reader can exit without nuking the producer's SHM.
    Creators (SharedMemory(create=True, ...)) intentionally still
    register so the SHM gets cleaned up if the creator dies abnormally.
    """
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return shm

# shared memory configuration
# Use separate shared memory for each image
def get_shm_name(image_name: str) -> str:
    """Get shared memory name for a specific image"""
    return f"isaac_{image_name}_image_shm"

SHM_SIZE_PER_IMAGE = 1920 * 1080 * 3 + 128  # ~6MB per image + header (supports up to 1080p)

# Backward compatibility
SHM_NAME = "isaac_multi_image_shm"  # Kept for backward compatibility
SHM_SIZE = SHM_SIZE_PER_IMAGE * 3   # Kept for backward compatibility


# Encoding tags written into SimpleImageHeader.encoding.
# 0 / 1 are pre-existing (raw 8-bit BGR / JPEG). 2 was added for the
# G1 depth pipeline so a single SHM region per image-name can carry
# either RGB or 16-bit metric depth without a parallel writer class.
ENC_RAW_U8_BGR = 0
ENC_JPEG = 1
ENC_RAW_U16_MONO = 2  # depth in millimetres, 1-channel, uint16 little-endian

# define the simplified header structure
class SimpleImageHeader(ctypes.LittleEndianStructure):  # Use little-endian for cross-platform compatibility
    """Simplified image header structure for individual images"""
    _fields_ = [
        ('timestamp', ctypes.c_uint64),    # timestamp
        ('height', ctypes.c_uint32),       # image height
        ('width', ctypes.c_uint32),        # image width
        ('channels', ctypes.c_uint32),     # number of channels
        ('image_name', ctypes.c_char * 16), # image name (e.g., 'head', 'left', 'right')
        ('data_size', ctypes.c_uint32),    # data size
        ('encoding', ctypes.c_uint32),     # 0=raw BGR, 1=JPEG
        ('quality', ctypes.c_uint32),      # JPEG quality (valid if encoding=1)
    ]


class MultiImageWriter:
    """A simplified multi-image shared memory writer using separate SHM for each image"""

    def __init__(self, enable_jpeg: bool = False, jpeg_quality: int = 85, skip_cvtcolor: bool = False):
        """Initialize the multi-image shared memory writer

        Args:
            enable_jpeg: whether to enable JPEG compression
            jpeg_quality: JPEG quality (0-100)
            skip_cvtcolor: whether to skip color conversion
        """
        # 50 FPS 限速（避免高频阻塞主循环）
        self._min_interval_sec = 1.0 / 50.0
        self._last_write_ts_ms = 0

        # 压缩与颜色空间配置（由主进程注入）
        self._enable_jpeg = bool(enable_jpeg)
        self._jpeg_quality = int(jpeg_quality)
        self._skip_cvtcolor = bool(skip_cvtcolor)

        # 为每个图像维护独立的共享内存
        self.shms = {}  # image_name -> SharedMemory
        print(f"[MultiImageWriter] Initialized with separate SHM per image")

    def set_options(self, *, enable_jpeg: Optional[bool] = None, jpeg_quality: Optional[int] = None, skip_cvtcolor: Optional[bool] = None):
        if enable_jpeg is not None:
            self._enable_jpeg = bool(enable_jpeg)
        if jpeg_quality is not None:
            self._jpeg_quality = int(jpeg_quality)
        if skip_cvtcolor is not None:
            self._skip_cvtcolor = bool(skip_cvtcolor)

    def write_images(self, images: Dict[str, np.ndarray]) -> bool:
        """Write multiple images to separate shared memories

        Args:
            images: the image dictionary, the key is the image name ('head', 'left', 'right'), the value is the image array

        Returns:
            bool: whether the writing is successful
        """
        if not images:
            return False

        # 轻量限速：最多 50 FPS，直接跳过多余写入，避免阻塞主循环
        now_ms = int(time.time() * 1000)
        if self._last_write_ts_ms and (now_ms - self._last_write_ts_ms) < int(self._min_interval_sec * 1000):
            return True

        success_count = 0

        for image_name, image in images.items():
            try:
                # 为每个图像获取独立的共享内存
                shm_name = get_shm_name(image_name)
                if shm_name not in self.shms:
                    try:
                        # 尝试打开现有的共享内存
                        self.shms[shm_name] = _open_shm_no_track(shm_name)
                    except FileNotFoundError:
                        # 如果不存在，创建新的共享内存
                        self.shms[shm_name] = shared_memory.SharedMemory(create=True, size=SHM_SIZE_PER_IMAGE, name=shm_name)

                shm = self.shms[shm_name]

                # 确保连续内存布局，尽量减少拷贝
                if not image.flags['C_CONTIGUOUS']:
                    image = np.ascontiguousarray(image)

                # Depth path: a uint16 single-channel image (mm) is
                # written as ENC_RAW_U16_MONO. We never JPEG-encode
                # depth — that would destroy the per-pixel metric
                # value, and the consumer (player/operator overlay)
                # expects exact mm reads.
                is_depth_u16 = (image.dtype == np.uint16 and image.ndim == 2)

                if not is_depth_u16:
                    # OpenCV 期望 BGR 格式；可通过配置跳过转换
                    if image.ndim == 3 and image.shape[2] == 3:
                        if not self._skip_cvtcolor:
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # get the image information
                if is_depth_u16:
                    height, width = image.shape
                    channels = 1
                else:
                    height, width, channels = image.shape

                # 准备头部
                header = SimpleImageHeader()
                header.timestamp = now_ms  # millisecond timestamp
                header.height = height
                header.width = width
                header.channels = channels
                header.image_name = image_name.encode('utf-8')[:15].ljust(16, b'\x00')  # truncate and pad to 16 bytes

                # 计算数据
                if is_depth_u16:
                    data_bytes = image.tobytes()
                    header.encoding = ENC_RAW_U16_MONO
                    header.quality = 0
                elif self._enable_jpeg:
                    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self._jpeg_quality)]
                    ok, buffer = cv2.imencode('.jpg', image, encode_params)
                    if not ok:
                        print(f"[MultiImageWriter] Failed to encode {image_name} as JPEG")
                        continue
                    data_bytes = buffer.tobytes()
                    header.encoding = ENC_JPEG
                    header.quality = int(self._jpeg_quality)
                else:
                    data_bytes = image.tobytes()
                    header.encoding = ENC_RAW_U8_BGR
                    header.quality = 0

                header.data_size = len(data_bytes)

                # 检查空间是否足够
                header_size = ctypes.sizeof(SimpleImageHeader)
                total_size = header_size + header.data_size
                if total_size > shm.size:
                    print(f"[MultiImageWriter] Not enough space for {image_name}: need {total_size}, available {shm.size}")
                    continue

                # 写入头部
                header_bytes = ctypes.string_at(ctypes.byref(header), header_size)
                shm.buf[0:header_size] = header_bytes

                # 写入数据
                data_start = header_size
                data_end = data_start + header.data_size
                shm.buf[data_start:data_end] = data_bytes

                success_count += 1

            except Exception as e:
                print(f"[MultiImageWriter] Error writing {image_name}: {e}")
                continue

        self._last_write_ts_ms = now_ms
        return success_count > 0

    def close(self):
        """Close all shared memories"""
        for shm_name, shm in self.shms.items():
            try:
                shm.close()
                print(f"[MultiImageWriter] Shared memory closed: {shm_name}")
            except Exception as e:
                print(f"[MultiImageWriter] Error closing {shm_name}: {e}")
        self.shms.clear()


class MultiImageReader:
    """A simplified multi-image shared memory reader using separate SHM per image"""

    def __init__(self):
        """Initialize the multi-image shared memory reader"""
        self.last_timestamps = {}  # image_name -> last_timestamp
        self.buffer = {}  # image_name -> cached_image
        self.shms = {}  # image_name -> SharedMemory

    def read_images(self) -> Optional[Dict[str, np.ndarray]]:
        """Read images from all available separate shared memories

        Returns:
            Dict[str, np.ndarray]: the image dictionary, the key is the image name, the value is the image array
        """
        images = {}
        image_names = ['head', 'left', 'right']  # Standard image names

        for image_name in image_names:
            try:
                shm_name = get_shm_name(image_name)

                # Open shared memory if not already open
                if shm_name not in self.shms:
                    try:
                        self.shms[shm_name] = _open_shm_no_track(shm_name)
                    except FileNotFoundError:
                        continue  # Skip if shared memory doesn't exist

                shm = self.shms[shm_name]
                header_size = ctypes.sizeof(SimpleImageHeader)

                # Read header
                header_data = bytes(shm.buf[:header_size])
                header = SimpleImageHeader.from_buffer_copy(header_data)

                # Check timestamp
                last_ts = self.last_timestamps.get(image_name, 0)
                if header.timestamp <= last_ts:
                    # Return cached image if available
                    if image_name in self.buffer:
                        images[image_name] = self.buffer[image_name]
                    continue

                # Read payload
                data_start = header_size
                data_end = data_start + header.data_size
                payload = bytes(shm.buf[data_start:data_end])

                # Decode image
                if header.encoding == ENC_JPEG:
                    encoded = np.frombuffer(payload, dtype=np.uint8)
                    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                elif header.encoding == ENC_RAW_U16_MONO:
                    image = np.frombuffer(payload, dtype=np.uint16)
                    expected = header.height * header.width
                    if image.size != expected:
                        print(f"[MultiImageReader] Depth data size mismatch for {image_name}: expected {expected}, got {image.size}")
                        continue
                    image = image.reshape(header.height, header.width)
                else:  # ENC_RAW_U8_BGR
                    image = np.frombuffer(payload, dtype=np.uint8)
                    expected_size = header.height * header.width * header.channels
                    if image.size != expected_size:
                        print(f"[MultiImageReader] Data size mismatch for {image_name}: expected {expected_size}, got {image.size}")
                        continue
                    image = image.reshape(header.height, header.width, header.channels)

                # Cache and return
                self.buffer[image_name] = image
                self.last_timestamps[image_name] = header.timestamp
                images[image_name] = image

            except Exception as e:
                print(f"[MultiImageReader] Error reading {image_name}: {e}")
                continue

        return images if images else None

    def read_concatenated_image(self) -> Optional[np.ndarray]:
        """Read all images and concatenate them horizontally (for backward compatibility)

        Returns:
            np.ndarray: the concatenated image array; if the reading fails, return None
        """
        images = self.read_images()
        if images is None or not images:
            return None

        try:
            # Concatenate images in order: head, left, right
            image_order = ['head', 'left', 'right']
            frames_to_concat = []

            for image_name in image_order:
                if image_name in images:
                    frames_to_concat.append(images[image_name])

            if not frames_to_concat:
                return None

            if len(frames_to_concat) > 1:
                concatenated_image = cv2.hconcat(frames_to_concat)
            else:
                concatenated_image = frames_to_concat[0]

            return concatenated_image

        except Exception as e:
            print(f"[MultiImageReader] Error concatenating images: {e}")
            return None

    def read_single_image(self, image_name: str) -> Optional[np.ndarray]:
        """Read a single specific image from its dedicated shared memory.

        Args:
            image_name: Name of the image to read ("head", "left", or "right")

        Returns:
            np.ndarray: The requested image array, or None if not found or error
        """
        try:
            shm_name = get_shm_name(image_name)

            # Open shared memory if not already open
            if shm_name not in self.shms:
                try:
                    self.shms[shm_name] = _open_shm_no_track(shm_name)
                except FileNotFoundError:
                    print(f"[MultiImageReader] Shared memory {shm_name} not found")
                    return None

            shm = self.shms[shm_name]
            header_size = ctypes.sizeof(SimpleImageHeader)

            # Read header
            header_data = bytes(shm.buf[:header_size])
            header = SimpleImageHeader.from_buffer_copy(header_data)

            # Check if there is new data
            last_ts = self.last_timestamps.get(image_name, 0)
            if header.timestamp <= last_ts:
                # Return cached image if available
                return self.buffer.get(image_name)

            # Read payload
            data_start = header_size
            data_end = data_start + header.data_size
            payload = bytes(shm.buf[data_start:data_end])

            # Decode image
            if header.encoding == ENC_JPEG:
                encoded = np.frombuffer(payload, dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if image is None:
                    return None
            elif header.encoding == ENC_RAW_U16_MONO:
                image = np.frombuffer(payload, dtype=np.uint16)
                expected = header.height * header.width
                if image.size != expected:
                    print(f"[MultiImageReader] Depth data size mismatch for {image_name}: expected {expected}, got {image.size}")
                    return None
                image = image.reshape(header.height, header.width)
            else:  # ENC_RAW_U8_BGR
                image = np.frombuffer(payload, dtype=np.uint8)
                expected_size = header.height * header.width * header.channels
                if image.size != expected_size:
                    print(f"[MultiImageReader] Data size mismatch for {image_name}: expected {expected_size}, got {image.size}")
                    return None
                image = image.reshape(header.height, header.width, header.channels)

            # Update buffer and timestamp
            self.buffer[image_name] = image
            self.last_timestamps[image_name] = header.timestamp
            return image

        except Exception as e:
            print(f"[MultiImageReader] Error reading single image {image_name}: {e}")
            return None

    def read_encoded_frame(self, image_name: str = "head") -> Optional[bytes]:
        """Read encoded payload for a specific image if available (e.g., JPEG). Returns bytes or None."""
        if self.shm is None:
            return None

        try:
            # Scan through all images in shared memory
            header_size = ctypes.sizeof(SimpleImageHeader)
            current_offset = 0

            while current_offset < self.shm.size - header_size:
                # Read header
                header_data = bytes(self.shm.buf[current_offset:current_offset + header_size])
                header = SimpleImageHeader.from_buffer_copy(header_data)

                # Check if this is the image we want and it's encoded
                current_image_name = header.image_name.decode('utf-8').rstrip('\x00')
                if current_image_name == image_name and header.encoding == 1:
                    # Check if there is new data
                    if header.timestamp <= self.last_timestamp:
                        return None

                    # Read the payload
                    data_start = current_offset + header_size
                    data_end = data_start + header.data_size
                    payload = bytes(self.shm.buf[data_start:data_end])

                    self.last_timestamp = header.timestamp
                    return payload

                # Move to next image
                current_offset += header_size + header.data_size

            return None

        except Exception as e:
            print(f"[MultiImageReader] Error reading encoded frame for {image_name}: {e}")
            return None

    def close(self):
        """Close all shared memories"""
        for shm_name, shm in self.shms.items():
            try:
                shm.close()
                print(f"[MultiImageReader] Shared memory closed: {shm_name}")
            except Exception as e:
                print(f"[MultiImageReader] Error closing {shm_name}: {e}")
        self.shms.clear()
        self.buffer.clear()
        self.last_timestamps.clear()


# ============================================================
# Point cloud SHM (for the simulated MID-360 lidar). Lives in its
# own SHM region so the existing image reader doesn't have to care
# about clouds. Cloud is written as raw float32 [N,3] in the
# sensor's own frame (lidar local) — the consumer applies the
# extrinsic transform.
# ============================================================
SHM_NAME_LIDAR = "isaac_lidar_pointcloud_shm"
# 200k pts/s × 0.1s × 12 bytes = 240 KB; allow 4× headroom.
SHM_SIZE_LIDAR = 1024 * 1024


class PointCloudHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ('timestamp', ctypes.c_uint64),    # ms since epoch
        ('n_points', ctypes.c_uint32),     # number of points in this frame
        ('stride_floats', ctypes.c_uint32),  # 3 = xyz, 4 = xyz + intensity
        ('frame_id', ctypes.c_char * 16),  # e.g. "lidar" or "torso"
        ('reserved', ctypes.c_uint32),
    ]


class PointCloudWriter:
    """Single-region SHM writer for one lidar frame."""

    def __init__(self, shm_name: str = SHM_NAME_LIDAR, shm_size: int = SHM_SIZE_LIDAR):
        self._shm_name = shm_name
        self._shm_size = shm_size
        self._shm: Optional[shared_memory.SharedMemory] = None

    def _ensure(self):
        if self._shm is not None:
            return
        try:
            self._shm = _open_shm_no_track(self._shm_name)
        except FileNotFoundError:
            self._shm = shared_memory.SharedMemory(create=True, size=self._shm_size,
                                                   name=self._shm_name)

    def write(self, points: np.ndarray, frame_id: str = "lidar") -> bool:
        """Write an [N, K] float32 array (K=3 xyz, K=4 xyz+intensity)."""
        if points is None:
            return False
        if points.ndim != 2 or points.shape[1] not in (3, 4):
            print(f"[PointCloudWriter] bad shape {points.shape}; expected [N,3] or [N,4]")
            return False
        if points.dtype != np.float32:
            points = points.astype(np.float32, copy=False)
        if not points.flags['C_CONTIGUOUS']:
            points = np.ascontiguousarray(points)

        self._ensure()
        n, k = points.shape
        header = PointCloudHeader()
        header.timestamp = int(time.time() * 1000)
        header.n_points = int(n)
        header.stride_floats = int(k)
        header.frame_id = frame_id.encode('utf-8')[:15].ljust(16, b'\x00')

        header_size = ctypes.sizeof(PointCloudHeader)
        payload = points.tobytes()
        total = header_size + len(payload)
        if total > self._shm.size:
            # Drop oldest points to fit. 200k×0.1s = 20k pts × 12B = 240KB,
            # well under the 1 MB region; we only hit this if someone
            # accumulates >1s, which the writer caller controls.
            max_pts = (self._shm.size - header_size) // (k * 4)
            points = points[:max_pts]
            n = max_pts
            header.n_points = int(n)
            payload = points.tobytes()

        self._shm.buf[0:header_size] = ctypes.string_at(ctypes.byref(header), header_size)
        self._shm.buf[header_size:header_size + len(payload)] = payload
        return True

    def close(self):
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None


class PointCloudReader:
    """Reads the latest cloud written by PointCloudWriter."""

    def __init__(self, shm_name: str = SHM_NAME_LIDAR):
        self._shm_name = shm_name
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._last_ts = 0

    def _ensure(self) -> bool:
        if self._shm is not None:
            return True
        try:
            self._shm = _open_shm_no_track(self._shm_name)
            return True
        except FileNotFoundError:
            return False

    def read(self):
        """Returns (points [N,K] float32, frame_id str, timestamp_ms) or None."""
        if not self._ensure():
            return None
        header_size = ctypes.sizeof(PointCloudHeader)
        header = PointCloudHeader.from_buffer_copy(bytes(self._shm.buf[:header_size]))
        if header.timestamp <= self._last_ts:
            return None
        n = int(header.n_points)
        k = int(header.stride_floats)
        if n == 0 or k not in (3, 4):
            return None
        nbytes = n * k * 4
        payload = bytes(self._shm.buf[header_size:header_size + nbytes])
        pts = np.frombuffer(payload, dtype=np.float32).reshape(n, k)
        self._last_ts = int(header.timestamp)
        frame_id = header.frame_id.decode('utf-8').rstrip('\x00')
        return pts.copy(), frame_id, int(header.timestamp)

    def close(self):
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None


# backward compatible class (single image)
class SharedMemoryWriter:
    """Backward compatible single image writer"""
    
    def __init__(self, shm_name: str = SHM_NAME, shm_size: int = SHM_SIZE):
        self.multi_writer = MultiImageWriter(shm_name, shm_size)
    
    def write_image(self, image: np.ndarray) -> bool:
        """Write a single image (as the head image)"""
        return self.multi_writer.write_images({'head': image})
    
    def close(self):
        self.multi_writer.close()


class SharedMemoryReader:
    """Backward compatible single image reader"""
    
    def __init__(self, shm_name: str = SHM_NAME):
        self.multi_reader = MultiImageReader(shm_name)
    
    def read_image(self) -> Optional[np.ndarray]:
        """Read a single image (the head image)"""
        images = self.multi_reader.read_images()
        return images.get('head') if images else None
    
    def close(self):
        self.multi_reader.close() 