from dataclasses import dataclass

CAMERA_IDS = ('head', 'left_wrist', 'right_wrist')
STREAM_TYPES = ('raw', 'rectified')
FIELDS = ('image_topic', 'stream_type', 'camera_info_topic', 'depth_topic', 'frame_id')
REQUIRED_FIELDS = ('image_topic', 'stream_type', 'camera_info_topic', 'frame_id')


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    image_topic: str
    stream_type: str
    camera_info_topic: str
    depth_topic: str
    frame_id: str


def make_camera_config(camera_id, values):
    if camera_id not in CAMERA_IDS:
        raise ValueError(f'unknown camera_id: {camera_id}')
    missing = [field for field in REQUIRED_FIELDS if not values.get(field)]
    if missing:
        raise ValueError(f'camera {camera_id} has empty fields: {", ".join(missing)}')
    if values['stream_type'] not in STREAM_TYPES:
        raise ValueError(f'camera {camera_id} has unknown stream_type: {values["stream_type"]}')
    return CameraConfig(camera_id, **{field: values.get(field, '') for field in FIELDS})


def declare_camera_parameters(node):
    node.declare_parameter('camera_id', 'head')
    for camera_id in CAMERA_IDS:
        node.declare_parameter(f'cameras.{camera_id}.enabled', False)
        for field in FIELDS:
            node.declare_parameter(f'cameras.{camera_id}.{field}', '')


def load_camera_config(node):
    camera_id = node.get_parameter('camera_id').value
    if camera_id not in CAMERA_IDS:
        raise ValueError(f'unknown camera_id: {camera_id}')
    prefix = f'cameras.{camera_id}.'
    if not node.get_parameter(prefix + 'enabled').value:
        raise ValueError(f'camera {camera_id} is disabled')
    values = {field: node.get_parameter(prefix + field).value for field in FIELDS}
    return make_camera_config(camera_id, values)
