"""对象存储抽象层

支持多种对象存储后端：MinIO、腾讯云 COS、本地存储、Base64。
参考 WeKnora 架构设计。
"""

import io
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ============== 抽象基类 ==============

class Storage(ABC):
    """对象存储抽象基类

    定义统一的上传接口，支持不同存储后端。
    """

    @abstractmethod
    def upload_file(self, file_path: str) -> str:
        """上传文件到对象存储

        Args:
            file_path: 本地文件路径

        Returns:
            文件访问 URL，失败返回空字符串
        """
        pass

    @abstractmethod
    def upload_bytes(self, content: bytes, file_ext: str = ".png") -> str:
        """上传字节内容到对象存储

        Args:
            content: 字节内容
            file_ext: 文件扩展名（如 .png, .jpg）

        Returns:
            文件访问 URL，失败返回空字符串
        """
        pass


# ============== MinIO 存储 ==============

class MinioStorage(Storage):
    """MinIO 对象存储实现

    兼容 S3 API 的对象存储服务。
    """

    def __init__(self, storage_config: dict[str, Any] | None = None):
        """初始化 MinIO 存储

        Args:
            storage_config: 存储配置字典，可为空则从环境变量读取
        """
        self.storage_config = storage_config or {}
        self.client, self.bucket_name, self.use_ssl, self.endpoint, self.path_prefix = (
            self._init_minio_client()
        )

    def _init_minio_client(self) -> tuple:
        """初始化 MinIO 客户端

        Returns:
            (client, bucket_name, use_ssl, endpoint, path_prefix)
        """
        try:
            from minio import Minio
        except ImportError:
            logger.error("minio package not installed, run: pip install minio")
            return None, None, None, None, None

        settings = get_settings()

        # 从配置或环境变量获取参数
        access_key = self.storage_config.get("access_key_id") or getattr(
            settings, "minio_access_key_id", "minioadmin"
        )
        secret_key = self.storage_config.get("secret_access_key") or getattr(
            settings, "minio_secret_access_key", "minioadmin"
        )
        bucket_name = self.storage_config.get("bucket_name") or getattr(
            settings, "minio_bucket_name", "kiki"
        )
        path_prefix_raw = self.storage_config.get("path_prefix") or getattr(
            settings, "minio_path_prefix", ""
        )
        path_prefix = path_prefix_raw.strip().strip("/") if path_prefix_raw else ""
        endpoint = self.storage_config.get("endpoint") or getattr(
            settings, "minio_endpoint", "localhost:9000"
        )
        use_ssl = self.storage_config.get("use_ssl") or getattr(
            settings, "minio_use_ssl", False
        )

        if not endpoint:
            logger.error("MinIO endpoint not configured")
            return None, None, None, None, None

        try:
            # 初始化客户端
            client = Minio(
                endpoint, access_key=access_key, secret_key=secret_key, secure=use_ssl
            )

            # 确保桶存在
            found = client.bucket_exists(bucket_name)
            if not found:
                logger.info(f"Creating MinIO bucket: {bucket_name}")
                client.make_bucket(bucket_name)

            return client, bucket_name, use_ssl, endpoint, path_prefix

        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            return None, None, None, None, None

    def _get_download_url(self, object_key: str) -> str:
        """构建 MinIO 对象访问 URL

        Args:
            object_key: 对象键

        Returns:
            访问 URL
        """
        settings = get_settings()
        public_endpoint = getattr(settings, "minio_public_endpoint", None)

        if public_endpoint:
            return f"{public_endpoint}/{self.bucket_name}/{object_key}"

        if self.use_ssl:
            return f"https://{self.endpoint}/{self.bucket_name}/{object_key}"
        return f"http://{self.endpoint}/{self.bucket_name}/{object_key}"

    def upload_file(self, file_path: str) -> str:
        """上传文件到 MinIO

        Args:
            file_path: 本地文件路径

        Returns:
            文件访问 URL
        """
        logger.info(f"Uploading file to MinIO: {file_path}")
        if not self.client:
            logger.warning("MinIO client not initialized")
            return ""

        try:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            object_key = (
                f"{self.path_prefix}/files/{uuid.uuid4().hex}{file_ext}"
                if self.path_prefix
                else f"files/{uuid.uuid4().hex}{file_ext}"
            )

            file_size = os.path.getsize(file_path)
            with open(file_path, "rb") as file_data:
                self.client.put_object(
                    bucket_name=self.bucket_name,
                    object_name=object_key,
                    data=file_data,
                    length=file_size,
                )

            url = self._get_download_url(object_key)
            logger.info(f"Successfully uploaded to MinIO: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload file to MinIO: {e}")
            return ""

    def upload_bytes(self, content: bytes, file_ext: str = ".png") -> str:
        """上传字节内容到 MinIO

        Args:
            content: 字节内容
            file_ext: 文件扩展名

        Returns:
            文件访问 URL
        """
        logger.info(f"Uploading bytes to MinIO: {len(content)} bytes")
        if not self.client:
            logger.warning("MinIO client not initialized")
            return ""

        try:
            object_key = (
                f"{self.path_prefix}/files/{uuid.uuid4().hex}{file_ext}"
                if self.path_prefix
                else f"files/{uuid.uuid4().hex}{file_ext}"
            )

            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
                data=io.BytesIO(content),
                length=len(content),
            )

            url = self._get_download_url(object_key)
            logger.info(f"Successfully uploaded bytes to MinIO: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload bytes to MinIO: {e}")
            return ""


# ============== 腾讯云 COS 存储 ==============

class CosStorage(Storage):
    """腾讯云 COS 对象存储实现"""

    def __init__(self, storage_config: dict[str, Any] | None = None):
        """初始化 COS 存储

        Args:
            storage_config: 存储配置字典
        """
        self.storage_config = storage_config or {}
        self.client, self.bucket_name, self.region, self.prefix = (
            self._init_cos_client()
        )

    def _init_cos_client(self) -> tuple:
        """初始化腾讯云 COS 客户端

        Returns:
            (client, bucket_name, region, prefix)
        """
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError:
            logger.error("qcloud_cos package not installed, run: pip install cos-python-sdk-v5")
            return None, None, None, None

        settings = get_settings()

        secret_id = self.storage_config.get("access_key_id") or getattr(
            settings, "cos_secret_id", ""
        )
        secret_key = self.storage_config.get("secret_access_key") or getattr(
            settings, "cos_secret_key", ""
        )
        region = self.storage_config.get("region") or getattr(
            settings, "cos_region", ""
        )
        bucket_name = self.storage_config.get("bucket_name") or getattr(
            settings, "cos_bucket_name", ""
        )
        app_id = self.storage_config.get("app_id") or getattr(
            settings, "cos_app_id", ""
        )
        prefix = self.storage_config.get("path_prefix") or getattr(
            settings, "cos_path_prefix", ""
        )

        if not all([secret_id, secret_key, region, bucket_name, app_id]):
            logger.warning("Incomplete COS configuration")
            return None, None, None, None

        try:
            config = CosConfig(
                Appid=app_id,
                Region=region,
                SecretId=secret_id,
                SecretKey=secret_key,
            )
            client = CosS3Client(config)
            return client, bucket_name, region, prefix

        except Exception as e:
            logger.error(f"Failed to initialize COS client: {e}")
            return None, None, None, None

    def _get_download_url(self, object_key: str) -> str:
        """构建 COS 对象访问 URL"""
        return f"https://{self.bucket_name}.cos.{self.region}.myqcloud.com/{object_key}"

    def upload_file(self, file_path: str) -> str:
        """上传文件到腾讯云 COS"""
        logger.info(f"Uploading file to COS: {file_path}")
        if not self.client:
            logger.warning("COS client not initialized")
            return ""

        try:
            file_ext = os.path.splitext(file_path)[1]
            object_key = f"{self.prefix}/files/{uuid.uuid4().hex}{file_ext}"

            self.client.upload_file(
                Bucket=self.bucket_name,
                LocalFilePath=file_path,
                Key=object_key,
            )

            url = self._get_download_url(object_key)
            logger.info(f"Successfully uploaded to COS: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload file to COS: {e}")
            return ""

    def upload_bytes(self, content: bytes, file_ext: str = ".png") -> str:
        """上传字节内容到腾讯云 COS"""
        logger.info(f"Uploading bytes to COS: {len(content)} bytes")
        if not self.client:
            logger.warning("COS client not initialized")
            return ""

        try:
            object_key = f"{self.prefix}/files/{uuid.uuid4().hex}{file_ext}"

            self.client.put_object(
                Bucket=self.bucket_name,
                Body=content,
                Key=object_key,
            )

            url = self._get_download_url(object_key)
            logger.info(f"Successfully uploaded bytes to COS: {url}")
            return url

        except Exception as e:
            logger.error(f"Failed to upload bytes to COS: {e}")
            return ""


# ============== 本地存储 ==============

class LocalStorage(Storage):
    """本地文件系统存储实现"""

    def __init__(self, storage_config: dict[str, Any] | None = None):
        """初始化本地存储

        Args:
            storage_config: 存储配置，可指定 base_dir
        """
        self.storage_config = storage_config or {}
        settings = get_settings()
        base_dir = self.storage_config.get("base_dir") or getattr(
            settings, "local_storage_base_dir", "./data/files"
        )
        self.files_dir = os.path.join(base_dir, "files")
        os.makedirs(self.files_dir, exist_ok=True)

    def upload_file(self, file_path: str) -> str:
        """本地存储直接返回原路径"""
        logger.info(f"Local storage: {file_path}")
        return file_path

    def upload_bytes(self, content: bytes, file_ext: str = ".png") -> str:
        """保存字节到本地文件"""
        logger.info(f"Saving bytes to local storage: {len(content)} bytes")
        file_name = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(self.files_dir, file_name)

        try:
            with open(file_path, "wb") as f:
                f.write(content)
            return file_path
        except Exception as e:
            logger.error(f"Failed to save bytes locally: {e}")
            return ""


# ============== Base64 存储 ==============

class Base64Storage(Storage):
    """Base64 编码存储（用于返回数据 URI）"""

    def upload_file(self, file_path: str) -> str:
        """读取文件并转换为 base64 URI"""
        logger.info(f"Base64 storage: {file_path}")
        return file_path

    def upload_bytes(self, content: bytes, file_ext: str = ".png") -> str:
        """转换为 base64 data URI"""
        logger.info(f"Converting bytes to base64: {len(content)} bytes")
        import base64

        file_ext = file_ext.lstrip(".")
        b64 = base64.b64encode(content).decode("utf-8")
        return f"data:image/{file_ext};base64,{b64}"


# ============== 空存储 ==============

class DummyStorage(Storage):
    """空存储实现

    用于测试或禁用对象存储的场景。
    """

    def upload_file(self, file_path: str) -> str:
        return ""

    def upload_bytes(self, content: bytes, file_ext: str = ".png") -> str:
        return ""


# ============== 工厂函数 ==============

def create_storage(storage_config: dict[str, Any] | None = None) -> Storage:
    """创建存储实例

    Args:
        storage_config: 存储配置字典，包含 provider 字段指定存储类型

    Returns:
        Storage 实例

    Examples:
        >>> storage = create_storage({"provider": "minio", "endpoint": "localhost:9000"})
        >>> url = storage.upload_bytes(b"hello", ".txt")
    """
    settings = get_settings()
    storage_type = (
        storage_config.get("provider", "").lower()
        if storage_config
        else getattr(settings, "storage_type", "local").lower()
    )

    logger.info(f"Creating {storage_type} storage instance")

    if storage_type == "minio":
        return MinioStorage(storage_config)
    elif storage_type == "cos":
        return CosStorage(storage_config)
    elif storage_type == "local":
        return LocalStorage(storage_config or {})
    elif storage_type == "base64":
        return Base64Storage()

    logger.warning(f"Unknown storage type: {storage_type}, using DummyStorage")
    return DummyStorage()


# ============== 全局单例 ==============

_global_storage: Storage | None = None


def get_storage() -> Storage:
    """获取全局存储实例（单例模式）

    Returns:
        Storage 实例
    """
    global _global_storage
    if _global_storage is None:
        _global_storage = create_storage()
    return _global_storage
