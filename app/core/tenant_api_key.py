"""租户 API Key 加密/解密工具

参考 WeKnora99 实现，使用 AES-GCM 加密租户 ID。
格式: sk-{base64(nonce + ciphertext)}

与用户 API Key (app/core/api_key.py) 的区别：
- 用户 API Key: 使用 bcrypt 哈希，存储在数据库，用于用户认证
- 租户 API Key: 使用 AES-GCM 加密，自包含租户 ID，用于租户级认证
"""

import os
import struct
from typing import NamedTuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.observability.logging import get_logger

logger = get_logger(__name__)

# ============== 配置 ==============

API_KEY_PREFIX = "sk"
NONCE_SIZE = 12  # AES-GCM 推荐的 nonce 大小
KEY_SIZE = 32  # AES-256

_API_KEY_SECRET: bytes | None = None


def get_api_key_secret() -> bytes:
    """获取 API Key 加密密钥"""
    global _API_KEY_SECRET
    if _API_KEY_SECRET is None:
        secret = os.getenv("TENANT_AES_KEY")
        if not secret:
            # 从配置获取
            from app.core.config import get_settings

            settings = get_settings()
            secret = getattr(settings, "tenant_aes_key", "")

        if not secret:
            raise ValueError(
                "TENANT_AES_KEY environment variable must be set. "
                "Generate one with: openssl rand -base64 32"
            )

        # 确保密钥长度正确
        if len(secret) < KEY_SIZE:
            secret = secret.ljust(KEY_SIZE, "0")[:KEY_SIZE]

        _API_KEY_SECRET = secret.encode("utf-8")[:KEY_SIZE]

    return _API_KEY_SECRET


class DecodedAPIKey(NamedTuple):
    """解码后的 API Key"""

    tenant_id: int


def generate_api_key(tenant_id: int) -> str:
    """生成 API Key

    Args:
        tenant_id: 租户 ID

    Returns:
        格式为 sk-{encrypted} 的 API Key
    """
    # 1. 将 tenant_id 转换为 bytes (小端序)
    id_bytes = struct.pack("<Q", tenant_id)  # 8 bytes

    # 2. 生成随机 nonce
    nonce = os.urandom(NONCE_SIZE)

    # 3. 使用 AES-GCM 加密
    secret = get_api_key_secret()
    aesgcm = AESGCM(secret)
    ciphertext = aesgcm.encrypt(nonce, id_bytes, None)

    # 4. 合并 nonce 和 ciphertext，然后 base64 编码
    combined = nonce + ciphertext
    import base64

    encoded = base64.urlsafe_b64encode(combined).decode("utf-8").rstrip("=")

    # 5. 添加前缀
    return f"{API_KEY_PREFIX}-{encoded}"


def extract_tenant_id_from_api_key(api_key: str) -> int | None:
    """从 API Key 中提取租户 ID

    Args:
        api_key: API Key 字符串

    Returns:
        租户 ID，验证失败返回 None
    """
    try:
        # 1. 验证格式并提取加密部分
        if not api_key.startswith(f"{API_KEY_PREFIX}-"):
            logger.warning("invalid_api_key_format", prefix=api_key[:5])
            return None

        encrypted_part = api_key[len(API_KEY_PREFIX) + 1 :]

        # 2. Base64 解码
        import base64

        # 添加 padding
        padded = encrypted_part + "=" * (4 - len(encrypted_part) % 4)
        encrypted_data = base64.urlsafe_b64decode(padded.encode("utf-8"))

        # 3. 分离 nonce 和 ciphertext
        if len(encrypted_data) < NONCE_SIZE:
            logger.warning("invalid_api_key_length", length=len(encrypted_data))
            return None

        nonce = encrypted_data[:NONCE_SIZE]
        ciphertext = encrypted_data[NONCE_SIZE:]

        # 4. 解密
        secret = get_api_key_secret()
        aesgcm = AESGCM(secret)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)

        # 5. 转换回 tenant_id
        tenant_id = struct.unpack("<Q", plaintext)[0]

        logger.debug("api_key_decoded", tenant_id=tenant_id)
        return tenant_id

    except Exception as e:
        logger.error("api_key_decode_failed", error=str(e))
        return None


def validate_api_key(api_key: str, expected_tenant_id: int) -> bool:
    """验证 API Key 是否匹配指定租户

    Args:
        api_key: API Key 字符串
        expected_tenant_id: 期望的租户 ID

    Returns:
        是否匹配
    """
    decoded_tenant_id = extract_tenant_id_from_api_key(api_key)
    return decoded_tenant_id == expected_tenant_id


# 解决循环导入
if __name__ == "__main__":
    pass
