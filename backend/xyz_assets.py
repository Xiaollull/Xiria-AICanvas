"""Read-only XYZ original verification. Strong fingerprints are captured ONLY at completion.

Legacy metadata checks intentionally cannot authenticate old pixels. No hash is ever backfilled.
"""
import hashlib
import json
import shutil
import tempfile
from PIL import Image


class AssetVerificationError(ValueError):
    pass


def request_digest(request):
    # Definition records serialize base seeds as strings; the live request uses an int.
    value = dict(request)
    value['seed'] = str(value['seed'])
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def asset_expectation(definition, planned, output, attempt):
    return {'experiment_id': definition['experiment_id'], 'cell_id': planned['cell_id'],
            'coordinate': planned['coordinate'], 'request_sha256': request_digest(planned['request']),
            'definition_digest': definition['digest'], 'attempt': attempt,
            'seed': str(planned['request']['seed']), 'width': output['width'], 'height': output['height']}


def _hash(handle):
    handle.seek(0)
    digest, size = hashlib.sha256(), 0
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
        size += len(chunk)
    handle.seek(0)
    return digest.hexdigest(), size


def verify_handle(handle, expectation, fingerprint=None, *, capture=False):
    """Verify the SAME open file used by a consumer; streaming hash, PNG CRC, authoritative identity.
    capture requires new metadata. Normal reads without a fingerprint use explicit legacy limits."""
    try:
        sha, size = _hash(handle)
        if fingerprint is not None:
            if fingerprint.get('version') != 1 or fingerprint.get('sha256') != sha or fingerprint.get('bytes') != size:
                raise AssetVerificationError('原图 SHA-256/完整字节数与完成时记录不一致')
            if fingerprint.get('expected') != expectation:
                raise AssetVerificationError('原图摘要的冻结身份与权威实验定义不一致')
        with Image.open(handle) as image:
            if image.format != 'PNG' or image.size != (expectation['width'], expectation['height']):
                raise AssetVerificationError('原图格式/最终尺寸与完成记录不一致')
            parameters = json.loads(image.info.get('parameters', '{}'))
            metadata = parameters.get('xyz_plot') or {}
            for field in ('experiment_id', 'cell_id', 'attempt'):
                if metadata.get(field) != expectation[field]:
                    raise AssetVerificationError('原图 XYZ 元数据不属于本实验/格/尝试')
            if str(parameters.get('seed')) != expectation['seed']:
                raise AssetVerificationError('原图种子与冻结请求不一致')
            if capture or fingerprint is not None:
                for field in ('coordinate', 'request_sha256', 'definition_digest'):
                    if metadata.get(field) != expectation[field]:
                        raise AssetVerificationError('原图最终 XYZ 冻结元数据不一致')
            image.verify()  # CRC/truncation check, no pixel conversion or resize.
        # Image.verify may close its wrapper but not the caller's file. Detect in-place mutation.
        if _hash(handle) != (sha, size):
            raise AssetVerificationError('原图在校验期间发生变化')
        return {'version': 1, 'sha256': sha, 'bytes': size, 'expected': expectation,
                'assurance': 'sha256' if capture or fingerprint is not None else 'legacy_metadata_only'}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        if isinstance(error, AssetVerificationError):
            raise
        raise AssetVerificationError('原图损坏或缺少可验证 XYZ 元数据') from error


def verify_path(path, expectation, fingerprint=None, *, capture=False):
    with open(path, 'rb') as handle:
        return verify_handle(handle, expectation, fingerprint, capture=capture)


def verified_snapshot(path, expectation, fingerprint=None):
    """Byte-exact critical response snapshot; bounded RAM, owned temporary disk for large originals."""
    snapshot = tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode='w+b')
    try:
        with open(path, 'rb') as source:
            shutil.copyfileobj(source, snapshot, 1024 * 1024)
        verify_handle(snapshot, expectation, fingerprint)
        snapshot.seek(0)
        return snapshot
    except BaseException:
        snapshot.close()
        raise
