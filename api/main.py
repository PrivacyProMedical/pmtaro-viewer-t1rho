from __future__ import annotations

from pathlib import Path
import struct
import zlib

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import pydicom
except ModuleNotFoundError:
    pydicom = None


def _require(module, import_name: str, pip_name: str):
    if module is None:
        raise ModuleNotFoundError(
            f"Missing dependency '{import_name}'. Install it with: python -m pip install {pip_name}"
        )
    return module


def fit_t1rho_map(
    instance1,
    tsl1_ms: float,
    instance2,
    tsl2_ms: float,
    instance3,
    tsl3_ms: float,
    instance4,
    tsl4_ms: float,
    output_dir: str = '/tmp/t1rho_outputs',
):
    np_module = _require(np, 'numpy', 'numpy')
    pydicom_module = _require(pydicom, 'pydicom', 'pydicom')

    paths = [
        _extract_file_path_from_payload(instance1, 'instance1'),
        _extract_file_path_from_payload(instance2, 'instance2'),
        _extract_file_path_from_payload(instance3, 'instance3'),
        _extract_file_path_from_payload(instance4, 'instance4'),
    ]
    tsl_ms = np_module.asarray([tsl1_ms, tsl2_ms, tsl3_ms, tsl4_ms], dtype=np_module.float64)
    tsl_s = tsl_ms / 1000.0

    series = []
    template_ds = None
    for path in paths:
      ds = pydicom_module.dcmread(path)
      if template_ds is None:
          template_ds = ds
      slope = float(getattr(ds, 'RescaleSlope', 1))
      intercept = float(getattr(ds, 'RescaleIntercept', 0))
      series.append(ds.pixel_array.astype(np_module.float64) * slope + intercept)

    signal = np_module.stack(series, axis=0)
    epsilon = 1e-6
    signal = np_module.clip(signal, epsilon, None)

    log_signal = np_module.log(signal)
    x = tsl_s[:, None]
    y = log_signal.reshape(4, -1)

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    slope = ((x - x_mean) * (y - y_mean)).sum(axis=0) / ((x - x_mean) ** 2).sum(axis=0)
    slope = np_module.where(np_module.isfinite(slope), slope, 0.0)

    t1rho_seconds = np_module.where(slope < -epsilon, -1.0 / slope, 0.0)
    t1rho_ms = t1rho_seconds * 1000.0
    t1rho_ms = np_module.clip(t1rho_ms, 0.0, 100.0)

    map_2d = t1rho_ms.reshape(series[0].shape)

    output_dir_path = Path(output_dir or '/tmp/t1rho_outputs')
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = output_dir_path / 'T1rho_Map.dcm'
    preview_path = output_dir_path / 'T1rho_Map_Preview.png'

    _write_single_map_dicom(template_ds, output_path, 'T1rho', map_2d)
    _write_png_rgb(_jet_preview_rgb(map_2d), preview_path)

    return {
        'output_dcm': _module_file_payload(str(output_path)),
        'preview_png': _module_file_payload(str(preview_path)),
    }


def _extract_file_path_from_payload(payload, label: str) -> str:
    if hasattr(payload, 'to_py'):
        payload = payload.to_py()
    if isinstance(payload, (str, Path)):
        return str(payload)
    if not isinstance(payload, dict):
        raise ValueError(f'{label} must be a payload object.')
    selection = payload.get('selection') or payload
    if not isinstance(selection, dict):
        raise ValueError(f'{label} must include a selection object.')
    file_path = selection.get('path') or selection.get('filePath')
    if not file_path:
        raise ValueError(f'{label} must provide selection.path or selection.filePath.')
    return str(file_path)


def _module_file_payload(path: str) -> dict:
    file_path = str(path)
    return {
        'from': 'module',
        'selection': {
            'name': Path(file_path).name,
            'path': file_path,
            'isFile': True,
        },
    }


def _write_single_map_dicom(template_ds, output_path: Path, map_name: str, map_array):
    pydicom_module = _require(pydicom, 'pydicom', 'pydicom')
    np_module = _require(np, 'numpy', 'numpy')

    ds = template_ds.copy()
    pixel_data = np_module.rint(map_array).astype(np_module.uint16)
    source_array = np_module.asarray(map_array)
    ww = float(np_module.max(source_array) - np_module.min(source_array))
    wl = float((np_module.max(source_array) + np_module.min(source_array)) / 2)

    ds.Rows, ds.Columns = pixel_data.shape
    ds.PixelData = np_module.ascontiguousarray(pixel_data).tobytes()
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME2'
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    base_desc = str(getattr(template_ds, 'SeriesDescription', 'T1rho'))
    ds.SeriesDescription = f'{base_desc} - {map_name}'
    ds.ImageType = ['DERIVED', 'SECONDARY', 'T1RHO']

    if hasattr(template_ds, 'SeriesNumber'):
        try:
            ds.SeriesNumber = int(template_ds.SeriesNumber) + 201
        except Exception:
            ds.SeriesNumber = 201
    else:
        ds.SeriesNumber = 201

    ds.InstanceNumber = 1
    ds.SOPInstanceUID = pydicom_module.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom_module.uid.generate_uid()
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1
    ds.WindowCenter = wl
    ds.WindowWidth = ww if ww > 0 else 1.0

    if not hasattr(ds, 'file_meta') or ds.file_meta is None:
        ds.file_meta = pydicom_module.dataset.FileMetaDataset()
    if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
        ds.file_meta.TransferSyntaxUID = pydicom_module.uid.ExplicitVRLittleEndian

    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(str(output_path), write_like_original=False)


def _jet_preview_rgb(map_2d):
    np_module = _require(np, 'numpy', 'numpy')
    normalized = np_module.clip(map_2d / 100.0, 0.0, 1.0)

    def clamp(channel):
        return np_module.clip(channel, 0.0, 1.0)

    red = clamp(1.5 - np_module.abs(4.0 * normalized - 3.0))
    green = clamp(1.5 - np_module.abs(4.0 * normalized - 2.0))
    blue = clamp(1.5 - np_module.abs(4.0 * normalized - 1.0))

    rgb = np_module.stack([red, green, blue], axis=-1)
    rgb_uint8 = np_module.rint(rgb * 255.0).astype(np_module.uint8)
    return rgb_uint8


def _write_png_rgb(rgb_uint8, output_path: Path):
    np_module = _require(np, 'numpy', 'numpy')
    image = np_module.ascontiguousarray(rgb_uint8, dtype=np_module.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError('PNG preview expects an RGB image.')

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack('>I', len(data))
            + chunk_type
            + data
            + struct.pack('>I', zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    raw_rows = b''.join(
        b'\x00' + image[row].tobytes()
        for row in range(height)
    )
    compressed = zlib.compress(raw_rows, level=9)

    png_bytes = b''.join([
        b'\x89PNG\r\n\x1a\n',
        chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)),
        chunk(b'IDAT', compressed),
        chunk(b'IEND', b''),
    ])
    output_path.write_bytes(png_bytes)
