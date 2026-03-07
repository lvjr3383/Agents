from __future__ import annotations

import importlib.util
import logging
from functools import lru_cache

from .config import Settings, get_settings
from .storage import (
    AwsObjectStore,
    AwsRecordStore,
    LocalObjectStore,
    LocalRecordStore,
    ObjectStore,
    RecordStore,
)

logger = logging.getLogger(__name__)


def _has_boto3() -> bool:
    return importlib.util.find_spec("boto3") is not None


def _aws_requested(settings: Settings) -> bool:
    if settings.aws_mode == "aws":
        return True
    if settings.aws_mode != "auto":
        return False

    if not _has_boto3():
        return False

    try:
        import boto3

        session = boto3.Session(region_name=settings.aws_region)
        creds = session.get_credentials()
        return bool(creds and settings.dynamodb_table and settings.s3_bucket)
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_record_store() -> RecordStore:
    settings = get_settings()
    if _aws_requested(settings):
        try:
            logger.info("Using AWS DynamoDB record store")
            return AwsRecordStore(settings.dynamodb_table, settings.aws_region)
        except Exception as exc:
            logger.warning("Falling back to local record store: %s", exc)

    logger.info("Using local JSON record store at %s", settings.local_dynamo_path)
    return LocalRecordStore(settings.local_dynamo_path)


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    settings = get_settings()
    if _aws_requested(settings):
        try:
            logger.info("Using AWS S3 object store")
            return AwsObjectStore(settings.s3_bucket, settings.aws_region)
        except Exception as exc:
            logger.warning("Falling back to local object store: %s", exc)

    logger.info("Using local object store rooted at %s", settings.local_s3_dir)
    return LocalObjectStore(settings.local_s3_dir)
